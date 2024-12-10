# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import os
import sys
import subprocess
import argparse
import logging
from html.parser import incomplete
from pyexpat import features

import numpy as np
from time import time, sleep
import urllib
from plyfile import PlyData, PlyElement

# Must be imported before large libs
try:
    import open3d as o3d
except ImportError:
    raise ImportError("Please install open3d with `pip install open3d`.")

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim

from random import randint
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from utils.general_utils import inverse_sigmoid, get_expon_lr_func

#from simple_knn._C import distCUDA2




####################################################

# Dataset

####################################################
import numpy as np
import torch
from plyfile import PlyData
import MinkowskiEngine as ME
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import random
from scipy.spatial import KDTree



def center_around_z(coords, given_centroid=None):
    """
    Center the coordinates around the Z-axis by shifting the centroid to (0, 0).

    Parameters:
    - coords: A torch tensor of shape (n, 3) representing the coordinates (can be integer or float).
    - given_centroid: Optional (x, y) tuple or tensor. If provided, will center around this centroid.
                      If None, the centroid is calculated from the coordinates.

    Returns:
    - centered_coords: Coordinates shifted to center around the Z-axis.
    - centroid_xy: The (x, y) centroid used for centering (as integer values if coords are discrete).
    """
    if given_centroid is None:
        # Calculate the centroid in the X-Y plane if not provided
        centroid_xy = torch.mean(coords[:, :2].float(), dim=0)  # Calculate mean in float
        centroid_xy = centroid_xy.round().long()  # Round to nearest integer and convert to long
    else:
        # Use the provided centroid (ensure it's a torch tensor and integer type)
        centroid_xy = torch.tensor(given_centroid, dtype=torch.long, device=coords.device)

    print('Centroid', centroid_xy)

    # Shift the coordinates so that the centroid is at (0, 0)
    centered_coords = coords.clone()  # Create a copy of the original tensor
    centered_coords[:, 0] -= centroid_xy[0]  # Subtract the x-coordinate of the centroid
    centered_coords[:, 1] -= centroid_xy[1]  # Subtract the y-coordinate of the centroid

    # Return the centered coordinates and the centroid used
    return centered_coords, centroid_xy


class RandomRotation:

    def _M(self, axis, theta):
        return expm(np.cross(np.eye(3), axis / norm(axis) * theta))

    def __call__(self, coords, feats):
        R = self._M(
            np.random.rand(3) - 0.5, 2 * np.pi * (np.random.rand(1) - 0.5))
        return coords @ R, feats


class RotationZ90:

    def __init__(self):
        # Randomly choose one of the angles (90°, 180°, 270°, or 360°)
        self.angle = np.random.choice([n * np.pi for n in [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75]])
        # self.angle = np.random.choice([0, 0.5 *np.pi, np.pi, 1.5 * np.pi])

    def resample(self):
        # self.angle = np.random.choice([0, 0.5 *np.pi, np.pi, 1.5 * np.pi])
        self.angle = np.random.choice([n * np.pi for n in [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75]])

    def _rotation_matrix(self, angle):
        """
        Create a rotation matrix for rotating around the Z axis by the given angle.
        """
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        # Rotation matrix around the Z axis
        R_z = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
        return R_z

    def __call__(self, coords, feats):
        # Get the rotation matrix for the selected angle around the Z axis
        R = self._rotation_matrix(self.angle)

        # Apply the rotation to the coordinates
        rotated_coords = coords @ R

        # Return the rotated coordinates and the original features
        return rotated_coords, feats


import torch


class RandomMirror:
    def __init__(self):
        self.mirror_x = True

    def resample(self):
        self.mirror_x = torch.rand(1).item() > 0.5

    def _mirror_x(self, coords):
        """
        Mirror coordinates across the YZ plane (negate the X coordinates).
        """
        mirrored_coords = coords.clone()
        mirrored_coords[:, 0] = -mirrored_coords[:, 0]  # Negate the X coordinates
        return mirrored_coords

    def _mirror_y(self, coords):
        """
        Mirror coordinates across the XZ plane (negate the Y coordinates).
        """
        mirrored_coords = coords.clone()
        mirrored_coords[:, 1] = -mirrored_coords[:, 1]  # Negate the Y coordinates
        return mirrored_coords

    def __call__(self, coords, feats):
        # Randomly choose to mirror either across X or Y
        if self.mirror_x:
            # Mirror across the YZ plane (negate X)
            coords = self._mirror_x(coords)
        else:
            # Mirror across the XZ plane (negate Y)
            coords = self._mirror_y(coords)

        # Return the transformed coordinates and original features
        return coords, feats


def extract_random_chunk(min_coords, max_coords, chunk_size=60):
    """
    Extract a random 60x60 chunk of coordinates from the scene.
    """

    # Extract X and Y min/max from the input coordinates
    x_min, y_min = min_coords[0], min_coords[1]
    x_max, y_max = max_coords[0], max_coords[1]

    # Calculate the available range along the X and Y axes
    x_range = x_max - x_min
    y_range = y_max - y_min

    # Determine the chunk size for X and Y (if scene size < chunk_size, use the full range)
    x_chunk_size = min(chunk_size, x_range)
    y_chunk_size = min(chunk_size, y_range)

    # Randomly choose a start point, ensuring the chunk fits within the range

    x_start = np.random.uniform(x_min, x_max - x_chunk_size)
    y_start = np.random.uniform(y_min, y_max - y_chunk_size)

    # Calculate the end points for X and Y based on the chunk size
    x_end = x_start + x_chunk_size
    y_end = y_start + y_chunk_size

    # Apply the "keep" condition to filter the coordinates

    return (int(x_start), int(x_end), int(y_start), int(y_end))


from scene.dataset_readers import sceneLoadTypeCallbacks
from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import prefilter_voxel, render, network_gui
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON



class PointCloudDatasetBatched(Dataset):
    def __init__(self, files,config, cache=None):
        self.dirs = files  # List of (incomplete_path, ground_truth_path) tuples
        self.data_path = config['data_path']
        self.files =  [f'{config["data_path"]}/{p}' for p in files]
        self.voxel_size =  config['voxel_size'] # Size of each voxel in the grid
        self.init_threshold = config['threshold']
        self.normalize = config['normalize']
        self.z_threshold = None
        self.mean = None
        self.std = None
        self.border = None
        if cache is None:
            self.cache = {}
            print('New empty cache')
        else:
            self.cache = cache
            print('Cache', sorted(cache.keys()))

        self.rotator = RotationZ90()
        self.flipper = RandomMirror()

        self.augmentation = config['augmentation']
        self.rotate = config['rotate']
        self.flip = config['flip']

        self.load_scene_info = config['load_scene_info']
        self.center = config['center']

        parser = argparse.ArgumentParser(description="Process resolution input")

        # Add the '--resolution' argument
        parser.add_argument('--resolution', default=-1,type=int, help='Set the resolution (e.g., 1080)')
        parser.add_argument('--data_device', default='cuda',type=str, help='Device')


        # Parse the command-line arguments
        self.cam_args = parser.parse_args([])


    def __len__(self):
        return len(self.files)



    def read_ply_file(self, ply_file, normalize):
        #print('Normalieze', normalize)

        ply_data = PlyData.read(ply_file)
        vertex = ply_data['vertex'].data

        dtype = [('x', np.float32), ('y', np.float32), ('z', np.float32),
                 ('nx', np.float32), ('ny', np.float32), ('nz', np.float32),
                 ('red', int), ('green', int), ('blue', int)]

        data = np.array([tuple(vertex[i]) for i in range(len(vertex))], dtype=dtype)

        coords = np.vstack((data['x'], data['y'], data['z'])).T

        # Define threshold to remove ceiling, use gt threshold if defined
        if self.z_threshold is None:
            self.z_threshold = np.percentile(coords[:, 2], self.init_threshold)
        #print(f"Z threshold (th percentile): {self.z_threshold}")

        # Filter out the points above the threshold
        mask = coords[:, 2] <= self.z_threshold
        data = data[mask]
        coords = coords[mask]

        #Discretize point coordintaes
        #discrete_coords = np.floor(coords / self.voxel_size).astype(np.int32)
        print('Num before', coords.shape)
        discrete_coords = np.unique(np.round(coords / self.voxel_size), axis=0).astype(np.int32)
        np.random.shuffle(discrete_coords)

        print('Num after', discrete_coords.shape)

        coordinates_tensor = torch.from_numpy(discrete_coords)

        return coordinates_tensor

    def __getitem__(self, idx):

        res = {}
        data_path = self.files[idx]
        ply_path = os.path.join(data_path, 'sparse', '0', 'points3D.ply')

        scene_info = sceneLoadTypeCallbacks["Colmap"](data_path, 'images', True, 0)
        print("Loading Training Cameras")
        train_cameras = cameraList_from_camInfos(scene_info.train_cameras, 1.0, self.cam_args)
        print("Loading Test Cameras")
        test_cameras = cameraList_from_camInfos(scene_info.test_cameras, 1.0, self.cam_args)

        disc_coords = self.read_ply_file(ply_path, normalize=self.normalize)

        res.update({'train_cameras': train_cameras, 'test_cameras': test_cameras, 'scene_info': scene_info})


        self.z_threshold = None
        self.mean = None
        self.std = None
        self.border = None

        res.update({
            'coordinates': disc_coords,
        })

        return res


def custom_collate_fn_batched_temp(batch):
    coords = batch[0]['coordinates']

    # Create sparse tensors for incomplete and ground truth data
    sparse_tensor = ME.SparseTensor(
        features = torch.ones(coords.shape[0], 1),
        coordinates=ME.utils.batched_coordinates([coords]),
        quantization_mode=ME.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE,
        device='cuda'
    )

    batch[0].update({'sparse_tensor': sparse_tensor})
    return batch[0]


def get_bounding_box_and_voxel_count(discrete_coords):
    # Find the minimum and maximum coordinates along each axis
    min_coords = np.min(discrete_coords, axis=0)
    max_coords = np.max(discrete_coords, axis=0)

    # Calculate the number of voxels per dimension
    num_voxels_per_dim = (max_coords - min_coords) + 1

    return min_coords, max_coords, num_voxels_per_dim


def save_sparse_tensor_as_ply(sparse_tensor, ply_file_path, pred=True, voxel_size=0.05, denormalize=False, mean=0,
                              std=1):
    # Extract features and coordinates from the sparse tensor
    features = sparse_tensor.F[:, 1:].cpu().detach().numpy()

    if denormalize:
        print('Denorm features')
        features = features * std + mean
        print(
            f'Normalize Features mean: {features.mean(axis=0)} featues std: {features.std(axis=0)} features_min: {features.min(axis=0)} features_max: {features.max(axis=0)}')

    coordinates = sparse_tensor.C.cpu().detach().numpy()

    # The coordinates will need to be dequantized (multiplied by the voxel size) to get original positions
    original_coords = coordinates[:, 1:].astype(np.float32) * voxel_size

    dtype = [('x', np.float32), ('y', np.float32), ('z', np.float32),
             ('nx', np.float32), ('ny', np.float32), ('nz', np.float32),
             *[(f'f_offset_{i}', np.float32) for i in range(30)],
             *[(f'f_anchor_feat_{i}', np.float32) for i in range(32)],
             ('opacity', np.float32),
             *[(f'scale_{i}', np.float32) for i in range(6)],
             *[(f'rot_{i}', np.float32) for i in range(4)]]

    num_points = original_coords.shape[0]
    nx_ny_nz = np.zeros((num_points, 3), dtype=np.float32)  # nx, ny, nz = 0, 0, 0
    rot = np.tile([1, 0, 0, 0], (num_points, 1)).astype(np.float32)  # rot_0 = 1, rot_1 = 0, rot_2 = 0, rot_3 = 0

    # Combine original coordinates, nx_ny_nz, features, and rot
    data = np.hstack((original_coords, nx_ny_nz, features, rot))
    # Combine original coordinates and features
    # data = np.hstack((original_coords, features))

    # Create a structured array
    structured_array = np.empty(data.shape[0], dtype=dtype)
    for i, field in enumerate(structured_array.dtype.names):
        structured_array[field] = data[:, i]

    # Create a PlyElement and PlyData
    ply_element = PlyElement.describe(structured_array, 'vertex')
    ply_data = PlyData([ply_element])

    # Write the PlyData to a file
    ply_data.write(ply_file_path)


####################################################

####################################################

# Visualization

####################################################

from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('agg')


def subsample_features(features, fraction=0.2):
    num_samples = features.shape[0]
    num_to_sample = int(num_samples * fraction)
    indices = np.random.choice(num_samples, num_to_sample, replace=False)
    return features[indices]


def fit_pca(features, n_components=2):
    pca = PCA(n_components=n_components)
    pca.fit(features.cpu().detach().numpy())
    return pca


def apply_pca(pca, features):
    return pca.transform(features)


def visualize_pca(pca, predicted_features, ground_truth_features_pca, epoch, save_path='pca_visualization.png',
                  subsample_fraction=0.01):
    # Apply the same PCA transformation
    predicted_features_pca = apply_pca(pca, predicted_features)

    plt.figure(figsize=(10, 10))
    plt.scatter(predicted_features_pca[:, 0], predicted_features_pca[:, 1], alpha=0.5, label='Predicted Features',
                c='blue')
    plt.scatter(ground_truth_features_pca[:, 0], ground_truth_features_pca[:, 1], alpha=0.5,
                label='Ground Truth Features', c='red')
    plt.title(f'PCA Visualization of Features (Epoch {epoch})')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def PointCloud(points, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def capture_image(pcd, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)  # Do not display the window
    vis.add_geometry(pcd)
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(filename)
    vis.destroy_window()


def compute_average_squared_distances(points):
    # Ensure the input is a NumPy array
    points_np = points.detach().cpu().numpy()

    # Build a KDTree for efficient neighbor search
    kdtree = KDTree(points_np)

    # Query the tree for the 4 nearest neighbors (including the point itself)
    distances, _ = kdtree.query(points_np, k=4)

    # Compute the mean of the squared distances to the 3 nearest neighbors
    mean_squared_distances = np.mean(distances[:, 1:] ** 2, axis=1)

    # Convert the result back to a torch tensor
    return torch.tensor(mean_squared_distances, dtype=points.dtype, device=points.device)


from PIL import Image
from pathlib import Path
import torchvision.transforms.functional as tf
import lpips
lpips_fn = lpips.LPIPS(net='vgg').to('cuda')
import json

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

def log_heatmap(diff, step, writer, feature_names):
    print('logging heatmap')
    diff_np = diff.detach().cpu().numpy()
    plt.figure(figsize=(10, 8))
    sns.heatmap(diff_np.T, annot=True, xticklabels=False, yticklabels=feature_names, cmap="coolwarm", cbar=True)
    plt.title('Feature Differences Heatmap')
    writer.add_figure('Heatmap/Feature_Differences', plt.gcf(), step)
    plt.close()


def log_feature_distribution(pred, gt, step, writer, feature_names):
    print('logging feat dist')
    pred_np = pred.detach().cpu().numpy()
    gt_np = gt.detach().cpu().numpy()

    print('Numer of anchors:', gt_np.shape[0])

    for i, feature_name in enumerate(feature_names):
        plt.figure(figsize=(8, 6))
        # sns.histplot(pred_np[:, i], color='blue', label='Predicted', kde=True, stat="density")
        # sns.histplot(gt_np[:, i], color='orange', label='Ground Truth', kde=True, stat="density")
        print(feature_name, ':')
        #print(f'Pred Min: {pred_np[:, i].min()}, Max: {pred_np[:, i].max()}, Mean: {pred_np[:, i].mean()}')
        #Pred Minprint(f'GT Min: {gt_np[:, i].min()}, Max: {gt_np[:, i].max()}, Mean: {gt_np[:, i].mean()}\n')

        sns.kdeplot(pred_np[:, i], color='blue', label='Predicted', fill=True, alpha=0.3)
        sns.kdeplot(gt_np[:, i], color='orange', label='Ground Truth', fill=True, alpha=0.3)
        plt.title(f'Feature Distribution: {feature_name}')
        plt.legend()

        # Save the plot to a numpy array
        plt.tight_layout()
        plt.savefig(f'distribution_{feature_name}.png', bbox_inches='tight', pad_inches=0.1)
        plt.close()

        # Load the image and convert to a format TensorBoard can handle
        image = Image.open(f'distribution_{feature_name}.png')
        image = image.convert('RGB')
        image_np = np.array(image)

        # Normalize and transpose the image for TensorBoard
        image_np = image_np.transpose((2, 0, 1))  # Change from HWC to CHW
        writer.add_image(f'Distribution/{feature_name}', image_np, step)


from tqdm import tqdm
import torchvision

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
        writer.add_image(f'EVAL_GT/{log_path}', gts[0].squeeze(), k)
        writer.add_image(f'EVAL_PRED/{log_path}', renders[0].squeeze(), k)
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

    with open(str(method_dir.resolve()) + "/results.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(str(method_dir.resolve()) + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)


def render_set_train(model_path, name, exp_name, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, exp_name, "renders")
    error_path = os.path.join(model_path, name, exp_name, "errors")
    gts_path = os.path.join(model_path, name, exp_name, "gt")
    os.makedirs(render_path, exist_ok=True)
    os.makedirs(error_path, exist_ok=True)
    os.makedirs(gts_path, exist_ok=True)

    t_list = []
    visible_count_list = []
    name_list = []
    per_view_dict = {}
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        torch.cuda.synchronize();
        t_start = time.time()

        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background, no_scaffold = True)
        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask, is_training=True, no_scaffold = True)
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

def save_point_cloud(model, file_path):
    data = {
        "xyz": model.xyz.detach().cpu(),
        "offset": model.offset.detach().cpu(),
        "color": model.color.detach().cpu(),
        "opacity": model.opacity.detach().cpu(),
        "scaling": model.scaling.detach().cpu(),
        "rotation": model.rotation.detach().cpu(),
    }
    torch.save(data, file_path)
    print(f"Point cloud saved to {file_path}")

class GaussianSmall():
    def __init__(self,
                 xyz=None,  # Dimension [N, 3]
                 color=None,  # Dimension [N, 3]
                 opacity=None,  # Dimension [N, 1]
                 scaling=None,  # Dimension [N, 3]
                 rotation=None,  # Dimension [N, 4] (Quaternion)
                 ):
        self.N = xyz.shape[0] if xyz is not None else 0

        # Directly store Gaussian attributes
        self.xyz = xyz  # Center positions of Gaussians [N, 3]
        self.color = color  # Color of each Gaussian [N, 3]
        self.opacity = opacity  # Opacity of each Gaussian [N, 1]
        self.scaling = scaling  # Scaling factors [N, 3]
        self.rotation = rotation  # Rotation represented as a quaternion [N, 4]
        self.offset = None #torch.zeros_like(self.xyz)

        # Activation functions for attributes
        #self.scaling_activation = torch.exp
        self.opacity_activation = torch.sigmoid
        self.rotation_activation = torch.nn.functional.normalize



    @property
    def get_scaling(self):
        return self.scaling

    @property
    def get_color(self):
        return self.color

    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity

    @property
    def get_cov_mlp(self):
        return self.mlp_cov

    @property
    def get_color_mlp(self):
        return self.mlp_color

    @property
    def get_rotation(self):
        return self.rotation_activation(self.rotation)

    @property
    def get_xyz(self):
        return self.xyz + self.offset


    @property
    def get_opacity(self):
        return self.opacity_activation(self.opacity)

    def get_parameters(self):
        l = [
            {'params': [self.offset], "name": "offset"},
            {'params': [self.color], "name": "color"},
            {'params': [self.opacity], "name": "opacity"},
            {'params': [self.scaling], "name": "scaling"},
            {'params': [self.rotation], "name": "rotation"},
        ]
        return l

    def training_setup(self, training_args):

        l = self.get_parameters()

        # Optimizer
        self.optimizer = torch.optim.Adam(l, lr=0.0005, eps=1e-15)

        # Learning rate schedulers

        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init,
                                                       lr_final=training_args.offset_lr_final,
                                                       lr_delay_mult=training_args.offset_lr_delay_mult,
                                                       max_steps=training_args.offset_lr_max_steps)

    def check_gradients(self):
        parameters = self.get_parameters()
        for param_dict in parameters:
            param_name = param_dict['name']
            param_tensor = param_dict['params'][0]
            if param_tensor.grad is None:
                print(f"No gradient for parameter: {param_name}")
            else:
                grad_norm = param_tensor.grad.norm().item()
                print(f"Gradient for parameter '{param_name}': Norm = {grad_norm}")
        print('xyzgrad', self.xyz.grad)

    def create_from_xyz(self, xyz):
        # Initialize attributes from given xyz locations
        self.N = xyz.shape[0]
        print(self.N)
        self.xyz = xyz.float().cuda().detach()  # Set xyz directly from input
        self.offset = torch.zeros_like(self.xyz).float().cuda().requires_grad_(True)
        self.color = torch.zeros((xyz.shape[0], 3)).float().cuda().requires_grad_(True)  # Initialize colors to zero
        self.opacity = inverse_sigmoid(
            0.1 * torch.ones((xyz.shape[0], 1), dtype=torch.float, device="cuda")).requires_grad_(True)  # Set initial opacity
        self.scaling = torch.zeros((xyz.shape[0], 3)).float() + 0.05
        self.scaling = self.scaling.cuda().requires_grad_(True)
        print(self.scaling.shape, self.scaling, self.scaling.max(), self.scaling.min())
        self.rotation = torch.zeros((xyz.shape[0], 4), device="cuda")  # Initialize rotation as identity quaternion
        self.rotation[:, 0] = 1
        self.rotation = self.rotation.requires_grad_(True)


    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr



####################################################

####################################################

# Training

####################################################

import time
import seaborn as sns
from IPython.display import display
import torch.multiprocessing as mp
import gc
import pickle


def image_gradients(image):
    dx = image[:, :, :-1, 1:] - image[:, :, :-1, :-1]
    dy = image[:, :, 1:, :-1] - image[:, :, :-1, :-1]
    return dx, dy

def gradient_loss(pred, gt):
    pred_dx, pred_dy = image_gradients(pred)
    gt_dx, gt_dy = image_gradients(gt)
    return torch.mean(torch.abs(pred_dx - gt_dx) + torch.abs(pred_dy - gt_dy))


def train(train_files=None,dataset_config=None,exp_name="CompletionNet_0.05_sep", writer_name=None, save_path='single_gaussians-test'):
    mp.set_start_method('spawn')
    print(torch.cuda.is_available())
    print(exp_name)
    # Load Model

    cache = {}

    dataset = PointCloudDatasetBatched(train_files, dataset_config, cache)
    dataloader = DataLoader(dataset, batch_size=dataset_config['batch_size'], shuffle=False,
                            collate_fn=custom_collate_fn_batched_temp)


    # Define experiment writer for tensorboard
    writer = SummaryWriter(f'{writer_name}/{exp_name}')

    parser = argparse.ArgumentParser(description="Training script parameters")
    pp = PipelineParams(parser)
    op = OptimizationParams(parser)
    args = parser.parse_args([])
    pipe = pp.extract(args)
    opt = op.extract(args)

    num_samples = len(dataset)


    for i, batch in enumerate(dataloader):
        print('Sample {}/{}'.format(i, num_samples))

        sparse_coordinates = batch['sparse_tensor']
        xyz = sparse_coordinates.C[:, 1:].float() * dataset_config['voxel_size']

        scene_path =batch['scene_info'].ply_path.split('/')
        scene_path = os.path.join(save_path, scene_path[-5], scene_path[-4])
        print(scene_path)

        gaussians = GaussianSmall()
        gaussians.create_from_xyz(xyz)
        gaussians.training_setup(op)

        for iteration in range(0, 30000):
            loss = 0

            train_cameras = batch['train_cameras']
            # Pick a random Camera
            viewpoint_stack = None
            if not viewpoint_stack:
                viewpoint_stack = train_cameras.copy()


            bg_color = [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

            voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background, no_scaffold=True)

            retain_grad = True  # (iteration < opt.update_until and iteration >= 0)
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, is_training=True,
                                visible_mask=voxel_visible_mask,
                                retain_grad=retain_grad, no_scaffold =True)

            image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = \
                render_pkg[
                    "render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg[
                    "selection_mask"], \
                    render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]

            gt_image = viewpoint_cam.original_image.cuda()

            Ll1 = l1_loss(image, gt_image)

            ssim_loss = (1.0 - ssim(image, gt_image))
            psnr_v = psnr(image.detach().cuda(), gt_image.detach().cuda()).mean().double()

            if iteration % 100 == 0:
                writer.add_scalar(f'{scene_path}/PSNR', psnr_v, iteration)
                writer.add_image(f'{scene_path}LOSS/pred', image.detach().cpu(), iteration)
                writer.add_image(f'{scene_path}LOSS/gt', gt_image.detach().cpu(), iteration)

            scaling_reg = scaling.prod(dim=1).mean()
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + 0.01 * scaling_reg
            #print('Render loss', loss.item())

            # Backward pass and optimize
            loss.backward()

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration == 15000 or iteration == 29999:

                bg_color = [0, 0, 0]
                background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

                render_set_train(scene_path, "test", exp_name,
                                                              batch['test_cameras'], gaussians, pp, background)
                evaluate_train(scene_path, method=exp_name, writer=writer, k=iteration,
                               log_path=scene_path)

        ply_dir = os.path.join(scene_path, exp_name)
        os.makedirs(ply_dir, exist_ok=True)
        save_point_cloud(gaussians, os.path.join(ply_dir, 'gaussians.pth') )


    writer.close()



import json
import argparse
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train a model with specified configurations.")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()

    # Load configuration from file
    with open(args.config, 'r') as config_file:
        config = json.load(config_file)

    # Extract configurations
    train_files_path = config.get('train_files', '')
    exp_name = config.get('exp_name', 'default_exp')
    writer_name = config.get('writer_name', 'default_writer')
    save_path = config.get('save_path', 'default_save_path')
    dataset_config = config.get('dataset_config', {})

    # Load train files from the specified text file
    with open(train_files_path, 'r') as train_file:
        train_files = [line.strip() for line in train_file]

    print('Train_file:', train_files)

    # Call the train function with parsed configurations
    train(train_files=train_files,
          dataset_config=dataset_config,
          exp_name=exp_name,
          writer_name=writer_name,
          save_path=save_path)
