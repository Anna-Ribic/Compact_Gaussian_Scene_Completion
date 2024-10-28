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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
#from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    def __init__(self, args : ModelParams, gaussians, load_iteration=None, shuffle=True, resolution_scales=[1.0], ply_path=None, model_path=None, source_path=None, loaded_path=None, no_load_mlps=False):
        """b
        :param path: Path to colmap scene main folder.
        """
        if model_path is None:
            self.model_path = args.model_path
        else:
            self.model_path = model_path

        if source_path is None:
            self.source_path = args.source_path
        else:
            self.source_path = source_path

        self.loaded_iter = None
        self.gaussians = gaussians

        if not loaded_path:
            if load_iteration:
                if load_iteration == -1:
                    self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
                else:
                    self.loaded_iter = load_iteration
                
                print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        print('sourcepath',self.source_path, 'modelpath', model_path, 'im', args.images, 'eval', args.eval)
        if os.path.exists(os.path.join(self.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](self.source_path, args.images, args.eval, args.lod)
        elif os.path.exists(os.path.join(self.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](self.source_path, args.white_background, args.eval, ply_path=ply_path)
        else:
            assert False, "Could not recognize scene type!"

        self.gaussians.set_appearance(len(scene_info.train_cameras))
        
        if not self.loaded_iter:
            print('Load from input expect pointcloud.ply\n')
            if ply_path is not None:
                with open(ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                    dest_file.write(src_file.read())
            else:
                print(f'LOAAAAAADING, {self.model_path}')
                with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                    dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # print(f'self.cameras_extent: {self.cameras_extent}')

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if loaded_path:
            print(f'Loading model from {loaded_path}')
            self.gaussians.load_ply_sparse_gaussian(os.path.join(self.model_path,
                                                                 "point_cloud",
                                                                 loaded_path,
                                                                 "point_cloud.ply"))
        elif self.loaded_iter:
            print(f'Loading model from iteration {self.loaded_iter}')
            self.gaussians.load_ply_sparse_gaussian(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            if not no_load_mlps:
                print('Loading MLPS')
                self.gaussians.load_mlp_checkpoints(os.path.join(self.model_path,
                                                               "point_cloud",
                                                               "iteration_" + str(self.loaded_iter)))
        else:
            print(f'Loading from pcd {scene_info.point_cloud}')
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration, no_iter=False, no_mlps=False):
        if no_iter:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/{}".format(iteration))
        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        print(f'Saving scene to {point_cloud_path}')
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        if not no_mlps:
            self.gaussians.save_mlp_checkpoints(point_cloud_path)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]