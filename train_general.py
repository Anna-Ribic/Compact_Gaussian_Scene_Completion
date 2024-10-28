#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import numpy as np
from torch.utils.data import Dataset

import subprocess

cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')

import torch
import torchvision
import json
import wandb
import time
from os import makedirs
import shutil, pathlib
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
# from lpipsPyTorch import lpips
import lpips
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import prefilter_voxel, render, network_gui
import sys
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

import torch.nn as nn

# torch.set_num_threads(32)
lpips_fn = lpips.LPIPS(net='vgg').to('cuda')

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")

class SceneGaussianDataset(Dataset):
    def __init__(self, scenes, dataset, opt, exp_name, init_mlp_path=None, load_iteration=-1, feat_only=False):
        """
        Args:
            scenes (list): A list of scene configurations or data.
            mlp_params (dict): Parameters for initializing shared MLPs.
            gaussian_params (dict): Parameters for initializing Gaussian models.
            load_iteration (int): Iteration to load (default: -1, which means latest).
        """
        self.exp_name = exp_name
        self.dataset = dataset
        self.scenes = scenes
        feat_dim = dataset.feat_dim
        n_offsets = dataset.n_offsets
        appearance_dim = dataset.appearance_dim

        self.feat_only = feat_only

        self.add_cov_dist = dataset.add_cov_dist
        self.cov_dist_dim = 1 if self.add_cov_dist else 0
        self.mlp_cov = nn.Sequential(
            nn.Linear(feat_dim + 3 + self.cov_dist_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 7 * n_offsets),
        ).cuda()


        self.add_color_dist = dataset.add_color_dist
        self.color_dist_dim = 1 if self.add_color_dist else 0
        self.mlp_color = nn.Sequential(
            nn.Linear(feat_dim + 3 + self.color_dist_dim + appearance_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 3 * n_offsets),
            nn.Sigmoid()
        ).cuda()


        self.add_opacity_dist = dataset.add_opacity_dist
        self.opacity_dist_dim = 1 if self.add_opacity_dist else 0
        self.mlp_opacity = nn.Sequential(
            nn.Linear(feat_dim + 3 + self.opacity_dist_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, n_offsets),
            nn.Tanh()
        ).cuda()


        self.load_iteration = load_iteration

        self.feat_dim = feat_dim
        self.appearance_dim = appearance_dim
        self.opt = opt

        if init_mlp_path is not None:
            self.load(init_mlp_path)

    def save(self, path):
        self.mlp_opacity.eval()
        opacity_mlp = torch.jit.trace(self.mlp_opacity,
                                      (torch.rand(1, self.feat_dim + 3 + self.opacity_dist_dim).cuda()))
        opacity_mlp.save(os.path.join(path, 'opacity_mlp.pt'))
        self.mlp_opacity.train()

        self.mlp_cov.eval()
        cov_mlp = torch.jit.trace(self.mlp_cov, (torch.rand(1, self.feat_dim + 3 + self.cov_dist_dim).cuda()))
        cov_mlp.save(os.path.join(path, 'cov_mlp.pt'))
        self.mlp_cov.train()

        self.mlp_color.eval()
        color_mlp = torch.jit.trace(self.mlp_color, (
            torch.rand(1, self.feat_dim + 3 + self.color_dist_dim + self.appearance_dim).cuda()))
        color_mlp.save(os.path.join(path, 'color_mlp.pt'))
        self.mlp_color.train()

    def load(self, path):
        print(f'Loading mlps from {path}')
        self.mlp_opacity = torch.jit.load(os.path.join(path, 'opacity_mlp.pt')).cuda()
        self.mlp_cov = torch.jit.load(os.path.join(path, 'cov_mlp.pt')).cuda()
        self.mlp_color = torch.jit.load(os.path.join(path, 'color_mlp.pt')).cuda()

        self.mlp_color.train()
        self.mlp_cov.train()
        self.mlp_opacity.train()


    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        """
        For a given index, returns the corresponding Gaussian model and Scene object.
        """
        model_path, source_path = self.scenes[idx]

        gaussians = GaussianModel(self.dataset.feat_dim, self.dataset.n_offsets, self.dataset.voxel_size, self.dataset.update_depth,
                                  self.dataset.update_init_factor, self.dataset.update_hierachy_factor, self.dataset.use_feat_bank,
                                  self.dataset.appearance_dim, self.dataset.ratio, self.dataset.add_opacity_dist, self.dataset.add_cov_dist,
                                  self.dataset.add_color_dist, mlp_opacity = self.mlp_opacity, mlp_cov = self.mlp_cov, mlp_color = self.mlp_color,)
        #scene = Scene(self.dataset, gaussians, ply_path=None, shuffle=False)scene = Scene(self.dataset, gaussians, shuffle=False, model_path=model_path, source_path=source_path , no_load_mlps=True)
        if not os.path.isfile(os.path.join(model_path,"point_cloud",self.exp_name,"point_cloud.ply")):
            print('Loading from input.ply\n')
            scene = Scene(self.dataset, gaussians, ply_path=None,shuffle=False, model_path=model_path, source_path=source_path,
                      no_load_mlps=True)
        else:
            scene = Scene(self.dataset, gaussians, load_iteration = -1, shuffle=False, model_path=model_path, source_path=source_path , loaded_path = self.exp_name,no_load_mlps=True)

        gaussians.training_setup(self.opt, feat_only=self.feat_only)

        return gaussians, scene

def custom_collate_fn(batch):
    return batch


def training(dataset, opt, pipe, lis, checkpoint_iterations, writer, exp_name, feat_only=False):

    mlp_path = os.path.join('mlps', exp_name)
    mlp_init_path = None
    if os.path.exists(mlp_path):
        mlp_init_path = mlp_path
    else:
        os.makedirs(os.path.join('mlps', exp_name))

    scenes_dataset = SceneGaussianDataset(lis, dataset, opt, exp_name, init_mlp_path=mlp_init_path, feat_only=feat_only)
    dataloader = DataLoader(scenes_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)

    outer_it = 100 #Number of epochs
    inner_it = 500 #Number of optimization steps per scene


    for k in range(outer_it):
        print('K:', k)

        for scene_pack in dataloader:

            for gaussians, scene in scene_pack:
                print('source', scene.source_path)

                first_iter = 0

                viewpoint_stack = None
                ema_loss_for_log = 0.0
                running_loss = 0.0
                running_psnr = 0.0
                first_iter += 1
                for iteration in range(inner_it):

                    gaussians.update_learning_rate(iteration)

                    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
                    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

                    # Pick a random Camera
                    if not viewpoint_stack:
                        viewpoint_stack = scene.getTrainCameras().copy()
                    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

                    voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background)
                    retain_grad = True #(iteration < opt.update_until and iteration >= 0)
                    render_pkg = render(viewpoint_cam, gaussians, pipe, background, is_training=True, visible_mask=voxel_visible_mask,
                                        retain_grad=retain_grad)

                    image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = render_pkg[
                        "render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], \
                    render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]

                    gt_image = viewpoint_cam.original_image.cuda()


                    Ll1 = l1_loss(image, gt_image)

                    ssim_loss = (1.0 - ssim(image, gt_image))

                    scaling_reg = scaling.prod(dim=1).mean()
                    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + 0.01 * scaling_reg

                    psnr_v = psnr(image, gt_image).mean().double()

                    running_loss += loss.item() / inner_it
                    running_psnr += psnr_v / inner_it

                    loss.backward()

                    with torch.no_grad():
                        if k < 30  and k > 1:
                            # add statis
                            gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter,
                                                      offset_selection_mask, voxel_visible_mask)

                            # densification
                            if k > 3 and iteration % 100 == 0:
                                print('Anchor Growing')
                                gaussians.adjust_anchor(check_interval=opt.update_interval,
                                                        success_threshold=opt.success_threshold,
                                                        grad_threshold=opt.densify_grad_threshold,
                                                        min_opacity=opt.min_opacity)
                        elif k ==30 and iteration == 0:
                            del gaussians.opacity_accum
                            del gaussians.offset_gradient_accum
                            del gaussians.offset_denom
                            torch.cuda.empty_cache()

                        # Optimizer step
                        gaussians.optimizer.step()
                        gaussians.optimizer.zero_grad(set_to_none=True)

                        if (iteration in checkpoint_iterations):
                            torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

                if k %10 == 0:
                    print('Evaluate:')
                    render_sets_train(gaussians, scene, exp_name, pipe)
                    evaluate_train(scene.model_path, method=exp_name, writer=writer, k=k,
                                   log_path=scene.source_path)
                    gaussians.train()

                scene.save(exp_name,no_iter=True, no_mlps=True)
                scenes_dataset.save(os.path.join('mlps', exp_name))

                print('Running loss', running_loss)
                print(f'Logging {k}, {scene.source_path}, {running_psnr}, {running_loss}')
                writer.add_scalar(f'Loss/{scene.source_path}', running_loss, k)
                writer.add_scalar(f'PSNR/{scene.source_path}', running_psnr, k)


        if k % 25 == 0:
            scenes_dataset.save(os.path.join('mlps',exp_name))

    for scene_pack in dataloader:

        for gaussians, scene in scene_pack:
            print('Evaluate:')
            render_sets_train(gaussians, scene, exp_name, pipe)
            evaluate_train(scene.model_path, method=exp_name, writer=writer, k=outer_it, log_path=scene.source_path)



def render_set_train(model_path, name, exp_name, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, exp_name, "renders")
    error_path = os.path.join(model_path, name, exp_name, "errors")
    gts_path = os.path.join(model_path, name, exp_name, "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(error_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    t_list = []
    visible_count_list = []
    name_list = []
    per_view_dict = {}
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        torch.cuda.synchronize();
        t_start = time.time()

        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask)
        torch.cuda.synchronize();
        t_end = time.time()

        t_list.append(t_end - t_start)

        # renders
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = (render_pkg["radii"] > 0).sum()
        visible_count_list.append(visible_count)

        # gts
        gt = view.original_image[0:3, :, :]

        # error maps
        errormap = (rendering - gt).abs()

        name_list.append('{0:05d}'.format(idx) + ".png")
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(errormap, os.path.join(error_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        per_view_dict['{0:05d}'.format(idx) + ".png"] = visible_count.item()

    with open(os.path.join(model_path, name, exp_name, "per_view_count.json"), 'w') as fp:
        json.dump(per_view_dict, fp, indent=True)

    return t_list, visible_count_list

def render_sets_train(gaussians, scene,exp_name, pipeline: PipelineParams):
    with torch.no_grad():
        gaussians.eval()

        bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        t_test_list, visible_count = render_set_train(scene.model_path,"test",exp_name,
                                                scene.getTestCameras(), gaussians, pipeline, background)


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate_train(model_paths, method=None, writer=None, k =0 , log_path=''):
    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    scene_dir = model_paths
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}

    test_dir = Path(scene_dir) / "test"

    full_dict[scene_dir][method] = {}
    per_view_dict[scene_dir][method] = {}
    full_dict_polytopeonly[scene_dir][method] = {}
    per_view_dict_polytopeonly[scene_dir][method] = {}

    method_dir = test_dir / method
    gt_dir = method_dir / "gt"
    renders_dir = method_dir / "renders"
    renders, gts, image_names = readImages(renders_dir, gt_dir)

    ssims = []
    psnrs = []
    lpipss = []

    for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
        ssims.append(ssim(renders[idx], gts[idx]))
        psnrs.append(psnr(renders[idx], gts[idx]))
        lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())

    if writer is not None:
        writer.add_scalar(f'EVAL_SSIM/{log_path}', torch.stack(ssims).mean().item(), k)
        writer.add_scalar(f'EVAL_PSNR/{log_path}', torch.stack(psnrs).mean().item(), k)


    print(f"model_paths: \033[1;35m{model_paths}\033[0m")
    print("  SSIM : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(ssims).mean(), ".5"))
    print("  PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(psnrs).mean(), ".5"))
    print("  LPIPS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(lpipss).mean(), ".5"))
    print("")

    full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                         "PSNR": torch.tensor(psnrs).mean().item(),
                                         "LPIPS": torch.tensor(lpipss).mean().item()})
    per_view_dict[scene_dir][method].update(
        {"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
         "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
         "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})
         #"VISIBLE_COUNT": {name: vc for vc, name in zip(torch.tensor(visible_count).tolist(), image_names)}})

    with open(scene_dir + "/results.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--warmup', action='store_true', default=False)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[3_000, 7_000, 30_000])
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[3_000, 7_000, 30_000])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--gpu", type=str, default='-1')

    parser.add_argument("--exp_name", type=str, default='')
    parser.add_argument("--writer_name", type=str, default='')
    parser.add_argument("--opt_file", type=str, default='')
    parser.add_argument("--feat_only", action='store_true', default=False)


    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)

    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        os.system("echo $CUDA_VISIBLE_DEVICES")
        logger.info(f'using GPU {args.gpu}')


    exp_name = args.exp_name
    writer_name = args.writer_name
    opt_file = args.opt_file

    # Initialize system state (RNG)
    safe_state(args.quiet)

    writer = SummaryWriter(f'{writer_name}/{exp_name}')
    exp_name = 'featnofix_overfit-60-500-vs005-v2-withfixedmlp-50'

    with open(opt_file, 'r') as file:
        opt_scenes = [line.strip() for line in file]

    training(lp.extract(args), op.extract(args), pp.extract(args), opt_scenes, args.checkpoint_iterations, writer, exp_name, args.feat_only)

    # All done
    print("\nTraining complete.")
