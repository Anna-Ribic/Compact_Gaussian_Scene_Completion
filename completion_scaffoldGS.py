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
from ctypes.wintypes import tagPOINT
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

import MinkowskiEngine as ME
from random import randint
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from utils.general_utils import inverse_sigmoid, get_expon_lr_func
from torch.optim.lr_scheduler import StepLR
import json


ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%m/%d %H:%M:%S",
    handlers=[ch],
)

parser = argparse.ArgumentParser()
parser.add_argument("--resolution", type=int, default=128)
parser.add_argument("--max_iter", type=int, default=30000)
parser.add_argument("--val_freq", type=int, default=1000)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--lr", default=1e-2, type=float)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--stat_freq", type=int, default=50)
parser.add_argument("--weights", type=str, default="modelnet_completion.pth")
parser.add_argument("--load_optimizer", type=str, default="true")
parser.add_argument("--eval", action="store_true")
parser.add_argument("--max_visualization", type=int, default=4)


from completion_models import CompletionNetSmaller

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
        mirrored_coords = coords.copy()
        mirrored_coords[:, 0] = -mirrored_coords[:, 0]  # Negate the X coordinates
        return mirrored_coords

    def _mirror_y(self, coords):
        """
        Mirror coordinates across the XZ plane (negate the Y coordinates).
        """
        mirrored_coords = coords.copy()
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



import copy
class PointCloudDatasetBatched(Dataset):
    def __init__(self, files,config, cache_dir='cache', use_cache=False):
        self.dirs = files  # List of (incomplete_path, ground_truth_path) tuples
        self.data_path = config['data_path']
        self.file_pairs =  [(f'{config["source_path"]}/{p}_pent/point_cloud/full_100-ou_60_in_500-v2/point_cloud.ply', f'{config["source_path"]}/{p}/point_cloud/{config["mlps_name"]}/point_cloud.ply') for p in files]
        self.voxel_size =  config['voxel_size'] # Size of each voxel in the grid
        self.init_threshold = config['threshold']
        self.normalize = config['normalize']
        self.z_threshold = None
        self.mean = None
        self.std = None
        self.border = None
        print(cache_dir, config['mlps_name'])
        self.cache_dir = os.path.join(cache_dir, config['mlps_name'])
        self.use_cache = use_cache


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

        self.mlps_name = config['mlps_name']


    def __len__(self):
        return len(self.file_pairs)

    def compute_global_mean_std(self):
        all_features = []

        for _, ground_truth_path in self.file_pairs:
            print(ground_truth_path)
            ply_data = PlyData.read(ground_truth_path)
            vertex = ply_data['vertex'].data

            dtype = [('x', np.float32), ('y', np.float32), ('z', np.float32),
                     ('nx', np.float32), ('ny', np.float32), ('nz', np.float32),
                     *[(f'f_offset_{i}', np.float32) for i in range(30)],
                     *[(f'f_anchor_feat_{i}', np.float32) for i in range(32)],
                     ('opacity', np.float32),
                     *[(f'scale_{i}', np.float32) for i in range(6)],
                     *[(f'rot_{i}', np.float32) for i in range(4)]]

            data = np.array([tuple(vertex[i]) for i in range(len(vertex))], dtype=dtype)

            coords = np.vstack((data['x'], data['y'], data['z'])).T

            # Define threshold to remove ceiling, use gt threshold if defined
            if self.z_threshold is None:
                self.z_threshold = np.percentile(coords[:, 2], self.init_threshold)
            # print(f"Z threshold (th percentile): {self.z_threshold}")

            # Filter out the points above the threshold
            mask = coords[:, 2] <= self.z_threshold
            data = data[mask]
            coords = coords[mask]

            # Do not incude normals and rotation in features since they are the same for all points (ScaffoldGs specific)
            features = np.vstack((
                # data['nx'], data['ny'], data['nz'],

                *[data[f'f_offset_{i}'] for i in range(30)],
                *[data[f'f_anchor_feat_{i}'] for i in range(32)],
                data['opacity'],
                *[data[f'scale_{i}'] for i in range(6)],
                # *[data[f'rot_{i}'] for i in range(4)],
            )).T

            all_features.append(features)

        # Concatenate all features from all ground truth files
        all_features = np.vstack(all_features)

        # Compute the global mean and std for all features
        global_mean = all_features.mean(axis=0)
        global_std = all_features.std(axis=0)

        print(f"Global mean: {global_mean}")
        print(f"Global std: {global_std}")

        global_mean = torch.from_numpy(all_features.mean(axis=0)).float()
        global_std = torch.from_numpy(all_features.std(axis=0)).float()

        # Save mean and std tensors to file
        torch.save({'mean': global_mean, 'std': global_std}, 'mean_std.pt')

    def read_ply_file(self, ply_file, normalize):
        #print('Normalieze', normalize)

        ply_data = PlyData.read(ply_file)
        vertex = ply_data['vertex'].data

        dtype = [('x', np.float32), ('y', np.float32), ('z', np.float32),
                 ('nx', np.float32), ('ny', np.float32), ('nz', np.float32),
                 *[(f'f_offset_{i}', np.float32) for i in range(30)],
                 *[(f'f_anchor_feat_{i}', np.float32) for i in range(32)],
                 ('opacity', np.float32),
                 *[(f'scale_{i}', np.float32) for i in range(6)],
                 *[(f'rot_{i}', np.float32) for i in range(4)]]

        data = np.array([tuple(vertex[i]) for i in range(len(vertex))], dtype=dtype)

        #print('Offset0\n:', data['f_offset_0'].min(), data['f_offset_0'].max())
        """for i in range(30):
            if data[f'f_offset_{i}'].min()!= 0:
                print(f'Offset_{i}\n:', data[f'f_offset_{i}'].min(), data[f'f_offset_{i}'].max())"""


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
        discrete_coords = np.floor(coords / self.voxel_size).astype(np.int32)

        #Do not incude normals and rotation in features since they are the same for all points (ScaffoldGs specific)
        features = np.vstack((
            #data['nx'], data['ny'], data['nz'],

            *[data[f'f_offset_{i}'] for i in range(30)],
            *[data[f'f_anchor_feat_{i}'] for i in range(32)],
            data['opacity'],
            *[data[f'scale_{i}'] for i in range(6)],
            #*[data[f'rot_{i}'] for i in range(4)],
        )).T

        #Normalize features
        if normalize:
            features = (features - self.global_mean) / (self.global_std + 1e-8)
            print(f'Normalize Features mean: {features.mean(axis=0)} featues std: {features.std(axis=0)} features_min: {features.min(axis=0)} features_max: {features.max(axis=0)}')

        #Define all points as present initially
        ones_column = np.ones((features.shape[0], 1))
        features_with_ones = np.hstack((ones_column, features))


        #features_tensor = torch.from_numpy(features_with_ones).float()
        #coordinates_tensor = torch.from_numpy(discrete_coords)

        features_tensor = features_with_ones
        coordinates_tensor = discrete_coords

        return features_tensor, coordinates_tensor

    def _get_cache_path(self, idx):
        return os.path.join(self.cache_dir, f'sample_{idx}.pkl')

    def __getitem__(self, idx):
        room_dir = self.dirs[idx]
        data_dir = f'{self.data_path}/{room_dir}'
        res={}


        cache_path = self._get_cache_path(idx)
        if self.use_cache and os.path.exists(cache_path):
            print(f'Loading from cache: {cache_path}')
            with open(cache_path, 'rb') as f:
                res = pickle.load(f)
            incomplete_features, incomplete_coords = res['incomplete']
            ground_truth_features, ground_truth_coords = res['ground_truth']

        else:
            print('incomplete path NOT found in cache\n:', cache_path)
            incomplete_path, ground_truth_path = self.file_pairs[idx]

            ground_truth_features, ground_truth_coords = self.read_ply_file(ground_truth_path, normalize=self.normalize)
            incomplete_features, incomplete_coords = self.read_ply_file(incomplete_path, normalize=self.normalize)

            if self.center:
                ground_truth_coords, centroid = center_around_z(ground_truth_coords)
                incomplete_coords, _ = center_around_z(incomplete_coords, centroid)

            scene_info = sceneLoadTypeCallbacks["Colmap"](data_dir, 'images', True, 0)
            print("Loading Training Cameras")
            train_cameras = cameraList_from_camInfos(scene_info.train_cameras, 1.0, self.cam_args)
            print("Loading Test Cameras")
            test_cameras = cameraList_from_camInfos(scene_info.test_cameras, 1.0, self.cam_args)

            res.update({'train_cameras': train_cameras, 'test_cameras': test_cameras, 'scene_info': scene_info})

            res.update({
                'incomplete': (incomplete_features, incomplete_coords),
                'ground_truth': (ground_truth_features, ground_truth_coords),
                'incomplete_path': incomplete_path,
                'ground_truth_path': ground_truth_path,
            })

            if self.use_cache:
                # Save to cache
                with open(cache_path, 'wb') as f:
                    pickle.dump(res, f)

        if self.augmentation:
            if self.rotate:
                # print('Random rotate')
                ground_truth_coords, ground_truth_features = self.rotator(ground_truth_coords, ground_truth_features)
                incomplete_coords, incomplete_features = self.rotator(incomplete_coords, incomplete_features)
                self.rotator.resample()
            if self.flip and random.random() < 0.2:
                # print('Random flip')
                ground_truth_coords, ground_truth_features = self.flipper(ground_truth_coords, ground_truth_features)
                incomplete_coords, incomplete_features = self.flipper(incomplete_coords, incomplete_features)
                self.flipper.resample()

        self.z_threshold = None
        self.mean = None
        self.std = None
        self.border = None

        res.update({
            'incomplete': (incomplete_features, incomplete_coords),
            'ground_truth': (ground_truth_features, ground_truth_coords)
        })

        return res


def custom_collate_fn_batched_temp(batch):
    incomplete_features, incomplete_coords = batch[0]['incomplete']
    ground_truth_features, ground_truth_coords = batch[0]['ground_truth']

    # Create sparse tensors for incomplete and ground truth data
    incomplete_sparse_tensor = ME.SparseTensor(
        features=incomplete_features,
        coordinates=ME.utils.batched_coordinates([incomplete_coords]),
        quantization_mode=ME.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE,
        device='cuda'
    )

    ground_truth_sparse_tensor = ME.SparseTensor(
        features=ground_truth_features,
        coordinates=ME.utils.batched_coordinates([ground_truth_coords]),
        quantization_mode=ME.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE,
        device='cuda'
    )

    batch[0]['incomplete'] = incomplete_sparse_tensor
    batch[0]['ground_truth'] = ground_truth_sparse_tensor

    return batch[0]


def custom_collate_fn_batched_multiple(batch):
    # Separate all incomplete and ground truth features and coordinates from the batch
    incomplete_features_list = [item['incomplete'][0] for item in batch]
    incomplete_coords_list = [item['incomplete'][1] for item in batch]

    ground_truth_features_list = [item['ground_truth'][0] for item in batch]
    ground_truth_coords_list = [item['ground_truth'][1] for item in batch]

    # Create sparse tensors for incomplete and ground truth data
    incomplete_sparse_tensor = ME.SparseTensor(
        features=torch.cat(incomplete_features_list, dim=0),
        coordinates=ME.utils.batched_coordinates(incomplete_coords_list),
        quantization_mode=ME.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE,
        device='cuda'
    )

    ground_truth_sparse_tensor = ME.SparseTensor(
        features=torch.cat(ground_truth_features_list, dim=0),
        coordinates=ME.utils.batched_coordinates(ground_truth_coords_list),
        quantization_mode=ME.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE,
        device='cuda'
    )

    # Replace the original batch data with the sparse tensors
    for i in range(len(batch)):
        batch[i]['incomplete'] = incomplete_sparse_tensor
        batch[i]['ground_truth'] = ground_truth_sparse_tensor

    return batch[0]


def custom_collate_fn(batch):
    return batch


def custom_collate_fn_batched(batch):
    batch_incomplete_coords = []
    batch_incomplete_features = []
    batch_ground_truth_coords = []
    batch_ground_truth_features = []

    # Collect features and coordinates for each item in the batch
    for item in batch:
        incomplete_features, incomplete_coords = item['incomplete']
        ground_truth_features, ground_truth_coords = item['ground_truth']

        incomplete_features, incomplete_coords = torch.from_numpy(incomplete_features).float(), torch.from_numpy(incomplete_coords)
        ground_truth_features, ground_truth_coords = torch.from_numpy(ground_truth_features).float(), torch.from_numpy(ground_truth_coords)

        # Append the coordinates and features for each sample
        batch_incomplete_coords.append(incomplete_coords)
        batch_incomplete_features.append(incomplete_features)

        batch_ground_truth_coords.append(ground_truth_coords)
        batch_ground_truth_features.append(ground_truth_features)

    # Create batched coordinates for incomplete and ground truth point clouds
    batch_incomplete_coords = ME.utils.batched_coordinates(batch_incomplete_coords)

    batch_ground_truth_coords = ME.utils.batched_coordinates(batch_ground_truth_coords)

    # Stack all features into a single tensor for incomplete and ground truth
    batch_incomplete_features = torch.cat(batch_incomplete_features, dim=0)
    batch_ground_truth_features = torch.cat(batch_ground_truth_features, dim=0)

    # Create sparse tensors for incomplete and ground truth data
    incomplete_sparse_tensor = ME.SparseTensor(
        features=batch_incomplete_features,
        coordinates=batch_incomplete_coords,
        quantization_mode=ME.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE,
        device='cuda'
    )

    ground_truth_sparse_tensor = ME.SparseTensor(
        features=batch_ground_truth_features,
        coordinates=batch_ground_truth_coords,
        quantization_mode=ME.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE,
        device='cuda'
    )

    return incomplete_sparse_tensor, ground_truth_sparse_tensor, batch

def custom_collate_fn_batched_multi(batch):
    batch_incomplete_coords = []
    batch_incomplete_features = []
    batch_ground_truth_coords = []
    batch_ground_truth_features = []

    # Collect features and coordinates for each item in the batch
    for item in batch:
        incomplete_features, incomplete_coords = item['incomplete']
        ground_truth_features, ground_truth_coords = item['ground_truth']

        incomplete_features, incomplete_coords = torch.from_numpy(incomplete_features).float(), torch.from_numpy(incomplete_coords)
        ground_truth_features, ground_truth_coords = torch.from_numpy(ground_truth_features).float(), torch.from_numpy(ground_truth_coords)

        # Append the coordinates and features for each sample
        batch_incomplete_coords.append(incomplete_coords)
        batch_incomplete_features.append(incomplete_features)

        batch_ground_truth_coords.append(ground_truth_coords)
        batch_ground_truth_features.append(ground_truth_features)

    # Create batched coordinates for incomplete and ground truth point clouds
    batch_incomplete_coords = ME.utils.batched_coordinates(batch_incomplete_coords)

    batch_ground_truth_coords = ME.utils.batched_coordinates(batch_ground_truth_coords)

    # Stack all features into a single tensor for incomplete and ground truth
    batch_incomplete_features = torch.cat(batch_incomplete_features, dim=0)
    batch_ground_truth_features = torch.cat(batch_ground_truth_features, dim=0)


    return batch_incomplete_features, batch_incomplete_coords, batch_ground_truth_features, batch_ground_truth_coords, batch  #incomplete_sparse_tensor, ground_truth_sparse_tensor, batch



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


from PIL import Image


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


class GaussianSmall():
    def __init__(self,
                 features,  # Dimension [N, feat_dim]
                 anchors,  # Dimension [N, 3]
                 mlps
                 ):
        self.N = anchors.shape[0]
        self.anchors = anchors
        print('FET', features.shape)
        self._anchor_feat = features[:, 30:62] #.detach().clone().requires_grad_(True).cuda()
        self._offset = features[:, :30].view(self.N, 10, 3) #.detach().clone().requires_grad_(True).cuda()
        # self._offset.retain_grad()
        self.n_offsets = 10
        self.scaling = features[:, 63:] #.detach().clone().requires_grad_(True).cuda()
        self._opacity = features[:, 62] #.detach().clone().requires_grad_(True).cuda()
        self.rotation = torch.tensor([1, 0, 0, 0]).repeat(self.N, 1).float().cuda()

        print(f'rotation: {self.rotation.shape}')

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.opacity_activation = torch.sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self.mlp_opacity = mlps['mlp_opacity']
        self.mlp_cov = mlps['mlp_cov']
        self.mlp_color = mlps['mlp_color']

        self.use_feat_bank = False
        self.appearance_dim = 0
        self.add_opacity_dist = False
        self.add_color_dist = False
        self.add_cov_dist = False

        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")


    @property
    def get_scaling(self):
        return 1.0 * self.scaling_activation(self.scaling)

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
    def get_anchor(self):
        return self.anchors.float()

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_parameters(self):
        l = [
            {'params': [self._offset], "name": "offset"},
            {'params': [self._anchor_feat],  "name": "anchor_feat"},
            {'params': [self._opacity],  "name": "opacity"},
            {'params': [self.scaling], "name": "scaling"},
        ]

        return l

    def training_setup(self, training_args):
        self.opacity_accum = torch.zeros((self.anchors.shape[0], 1), device="cuda")

        # Accumulators for gradients and normalization
        self.offset_gradient_accum = torch.zeros((self.anchors.shape[0] * self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.anchors.shape[0] * self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.anchors.shape[0], 1), device="cuda")


        """     l = [
            {'params': [self._offset], 'lr': training_args.offset_lr_init, "name": "offset"},
            {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self.scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
        ]"""

        l = [
            {'params': [self._offset], 'lr': 0.01, "name": "offset"},
            {'params': [self._anchor_feat], 'lr': 0.01, "name": "anchor_feat"},
            {'params': [self._opacity], 'lr': 0.01, "name": "opacity"},
            {'params': [self.scaling], 'lr': 0.01, "name": "scaling"},
        ]

        # Optimizer
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        # Learning rate schedulers

        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init,
                                                       lr_final=training_args.offset_lr_final,
                                                       lr_delay_mult=training_args.offset_lr_delay_mult,
                                                       max_steps=training_args.offset_lr_max_steps)

    def get_updated_features(self):
        # Step 1: Flatten _offset back to the original shape (N, 30)
        updated_offset = self._offset.view(self.N, 30)

        # Step 2: Gather _anchor_feat (already in the correct shape)
        updated_anchor_feat = self._anchor_feat

        # Step 3: Gather _opacity and scaling (both in the correct shapes)
        updated_opacity = self._opacity.unsqueeze(1)  # Make sure it has the shape [N, 1]
        updated_scaling = self.scaling

        # Step 4: Concatenate all parts to match the input feature vector's structure
        updated_features = torch.cat((updated_offset, updated_anchor_feat, updated_opacity, updated_scaling), dim=1)

        return updated_features



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

def get_cuda_tensors():
    return {id(obj): obj for obj in gc.get_objects() if isinstance(obj, torch.Tensor) and obj.is_cuda}


def clear_cuda_tensors(ids):
    # Find all CUDA tensors
    cuda_tensors = [obj for obj in gc.get_objects() if isinstance(obj, torch.Tensor) and obj.is_cuda]

    print(f"Found {len(cuda_tensors)} CUDA tensors.")

    # Move tensors to CPU and delete
    for tensor in cuda_tensors:
        tensor.cpu()  # Move tensor to CPU to free GPU memory
        del tensor

    # Trigger garbage collection and clear GPU cache
    torch.cuda.empty_cache()
    gc.collect()
    print("Cleared CUDA tensors.")


from torch.utils.viz._cycles import warn_tensor_cycles


def train(train_files=None, eval_files=None, dataset_config=None, mlps=None, loss_config=None,  model_path='weights_005_sep.pth',
          save_path='weights_005_add.pth',
          exp_name="CompletionNet_0.05_sep", num_epochs=200, lr=0.1, visualize=False, writer_name=None):
    mp.set_start_method('spawn')

    print('Experiment:', exp_name)
    # Load Model
    net = CompletionNetSmaller(feat_size=70)

    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total number of learnable parameters: {total_params}")

    start_epoch = 0
    print(model_path)
    if os.path.exists(model_path):
        print('Loading model from {}'.format(model_path))
        net.load_state_dict(torch.load(model_path)['state_dict'])
        start_epoch = torch.load(model_path)['curr_iter']
        print('Restarting from epoch {}'.format(start_epoch))
    else:
        print('Model path not found!')

    net.to('cuda')
    net.train()

    # Define optimizer

    net_optimizer = optim.Adam(net.parameters(),  lr=lr, eps=1e-15)
    scheduler = StepLR(net_optimizer, step_size=50, gamma=0.5)

    # Define presence and feature loss
    pos_weight = torch.tensor([3.0]).to('cuda')
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    dataset = PointCloudDatasetBatched(train_files, dataset_config)
    dataset_preloaded = [dataset[i] for i in range(len(dataset))]
    dataloader = DataLoader(dataset_preloaded, batch_size=dataset_config['batch_size'], shuffle=True, collate_fn=custom_collate_fn_batched_multi)

    eval_dataset = PointCloudDatasetBatched(eval_files, dataset_config)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn_batched)

    gc.collect()
    torch.cuda.empty_cache()

    # Define experiment writer for tensorboard
    writer = SummaryWriter(f'{writer_name}/{exp_name}')

    pruning = ME.MinkowskiPruning()

    parser = argparse.ArgumentParser(description="Training script parameters")
    pp = PipelineParams(parser)
    op = OptimizationParams(parser)
    args = parser.parse_args([])
    pipe = pp.extract(args)
    opt = op.extract(args)

    losses = loss_config['losses']
    loss_pres_flag = 'anchor' in losses
    loss_feat_flag = '3dfeat' in losses
    loss_render_flag = 'render' in losses #batch size should be 1 for render loss
    if loss_render_flag:
        assert dataset_config['batch_size'] == 1

    pres_eval = loss_pres_flag
    feat_eval = loss_feat_flag and not loss_render_flag
    render_eval = loss_feat_flag or loss_render_flag

    n_it = 1
    mini_size = loss_config['mini_batch_size']

    # Number of overall epochs
    num_epochs = num_epochs
    # Epoch when feat loss is included
    vis_frequency = 1
    logging_frequency = 10
    store_frequency = 10
    eval_frequency = 10

    first=True
    viewpoint_cam = None

    num_samples = len(dataset)


    for epoch in range(start_epoch, num_epochs):
        diff_v = True
        print('Epoch {}/{}'.format(epoch, num_epochs))
        epoch_loss = 0
        epoch_loss_feat = 0
        epoch_loss_pres = 0
        epoch_loss_render = 0
        pred_var = 0
        epoch_psnr = 0
        epoch_ssim = 0

        iou_computed = False

        start_epoch_time = time.time()
        start_sample_load_time = time.time()

        if epoch % store_frequency == 0:
            torch.save(
                {
                    "state_dict": net.state_dict(),
                    "optimizer": net_optimizer.state_dict(),
                    "curr_iter": epoch,
                },
                save_path,
            )


        for i, batch in enumerate(dataloader):
            print('Sample {}/{}'.format(i, num_samples))

            #incomplete_tensor, ground_truth_tensor = batch['incomplete'], batch['ground_truth']
            incomplete_features, incomplete_coords, ground_truth_features, ground_truth_coords, scenes_info = batch

            # Create sparse tensors for incomplete and ground truth data
            incomplete_tensor = ME.SparseTensor(
                features=incomplete_features,
                coordinates=incomplete_coords,
                quantization_mode=ME.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE,
                device='cuda'
            ).detach()

            ground_truth_tensor = ME.SparseTensor(
                features=ground_truth_features,
                coordinates=ground_truth_coords,
                quantization_mode=ME.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE,
                device='cuda'
            ).detach()
            cm = incomplete_tensor.coordinate_manager
            target_key, _ = cm.insert_and_map(
                ground_truth_tensor.C,
                string_id="target",
            )

            print(scenes_info[0]['incomplete_path'])

            for iteration in range(epoch*n_it, (epoch+1)*n_it):
                loss = 0

                out_cls, targets, _ = net(incomplete_tensor, target_key)


                # Identify new tensors
                # Identify new tensors

                if loss_pres_flag:

                    print('Compute Pres Loss')

                    num_layers, pres_loss = len(out_cls), 0
                    # Optional weighting scheme for layer losses

                    # Compute presence loss across all layers
                    for out_cl, target in zip(out_cls, targets):
                        curr_loss = crit(out_cl.F[:, 0].squeeze(), target.type(out_cl.F.dtype).to('cuda').detach())
                        print(f'press loss,{curr_loss.item()}')
                        pres_loss += curr_loss / num_layers

                    epoch_loss_pres += pres_loss.detach().cpu().item()
                    loss += 2 * pres_loss

                    del curr_loss, pres_loss, num_layers
                    torch.cuda.empty_cache()

                    print('after loss')
                    print(
                        f"{(torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory) * 100:.2f}%")

                    if visualize and epoch % vis_frequency == 0:
                        print('Visualize Train\n\n\n')
                        ll = []
                        for i in [2]:  # range(len(out_cls)): #Only last layer change for all layers
                            for b in range(dataset_config['batch_size']):
                                print(b)
                                keep = (out_cls[i].C[:, :1] == b).squeeze()
                                out_cls_b = pruning(out_cls[i], keep)

                                # Predicted anchors
                                keep = (out_cls_b.F[:, :1] > 0).squeeze()

                                print(keep.shape)

                                out_cls_pruned = pruning(out_cls_b, keep)
                                points_matched_t = (out_cls_pruned.C[:, 1:] + torch.tensor(
                                    [0, b * 300, 0]).cuda()).cpu().detach().numpy()
                                pc1 = PointCloud(points_matched_t)

                                # Input tensor
                                keep = (incomplete_tensor.C[:, :1] == b).squeeze()
                                points_in = incomplete_tensor.C[keep][:, 1:]
                                pc3 = PointCloud(
                                    (points_in + torch.tensor([-300, b * 300, 0]).cuda()).cpu().detach().numpy())

                                # Ground truth tensor
                                keep = (ground_truth_tensor.C[:, :1] == b).squeeze()
                                points_in = ground_truth_tensor.C[keep][:, 1:]
                                pc5 = PointCloud(
                                    (points_in + torch.tensor([300, b * 300, 0]).cuda()).cpu().detach().numpy())

                                ll.append(pc1)
                                ll.append(pc3)
                                ll.append(pc5)

                        o3d.visualization.draw_geometries(ll)
                        ll.clear()

                if loss_feat_flag:

                    print('Compute Feature Loss')

                    keep = (out_cls[-1].F[:, :1] > 0).squeeze()
                    out_cls_pruned = pruning(out_cls[-1], keep)

                    # Find matching anchors
                    A = out_cls_pruned.C[:, 1:]
                    B = ground_truth_tensor.C[:, 1:]

                    A_indices, B_indices = find_matching_indices(A, B)

                    if len(A_indices) > 0:
                        mapped_features = out_cls_pruned.F[A_indices, 1:]

                        ground_truth_features = ground_truth_tensor.F[B_indices, 1:]


                        mask = (torch.sum(torch.abs(ground_truth_features), dim=0) != 0).float()
                        mask = mask.unsqueeze(0).expand(ground_truth_features.shape)

                        mse_loss_per_element = (mapped_features - ground_truth_features) ** 2

                        masked_loss = mse_loss_per_element * mask

                        feature_loss = masked_loss.sum() / mask.sum()

                        loss += feature_loss

                        print('Feauture_loss:', feature_loss)

                        epoch_loss_feat += feature_loss.item()

                        feature_names = [f'f_offset_{i}' for i in range(30)] + \
                                        [f'f_anchor_feat_{i}' for i in range(32)] + \
                                        ['opacity'] + \
                                        [f'scale_{i}' for i in range(6)]



                        del mse_loss_per_element, masked_loss, feature_loss, ground_truth_features, mask, A_indices, B_indices
                        torch.cuda.empty_cache()




                if (loss_feat_flag and epoch % logging_frequency and iteration==epoch*n_it) or loss_render_flag:

                    train_cameras = scenes_info[0]['train_cameras']
                    # Pick a random Camera
                    viewpoint_stack = None
                    if not viewpoint_stack:
                        viewpoint_stack = train_cameras.copy()

                    keep = (out_cls[-1].F[:, :1] > 0).squeeze()
                    out_cls_pruned = pruning(out_cls[-1], keep)

                    features = out_cls_pruned.F[:, 1:]
                    anchors = out_cls_pruned.C[:, 1:].float() * dataset_config['voxel_size']

                    gaussians = GaussianSmall(features, anchors.detach(), mlps)

                    for mini in range(mini_size):
                        bg_color = [0, 0, 0]
                        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

                        #train_cameras = batch['train_cameras']
                        # Pick a random Camera
                        #viewpoint_stack = None
                        #if not viewpoint_stack:
                        #    viewpoint_stack = train_cameras.copy()
                        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

                        voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background)

                        retain_grad = True  # (iteration < opt.update_until and iteration >= 0)
                        render_pkg = render(viewpoint_cam, gaussians, pipe, background, is_training=True,
                                            visible_mask=voxel_visible_mask,
                                            retain_grad=retain_grad)

                        image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = \
                            render_pkg[
                                "render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg[
                                "selection_mask"], \
                                render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]

                        gt_image = viewpoint_cam.original_image.cuda()

                        # writer.add_image('LOSS/pred', image, epoch)
                        # writer.add_image('LOSS/gt', gt_image, epoch)
                        if iteration % 1 == 0 and mini==0 and i==0:
                            writer.add_image('LOSS/pred', image.detach().cpu(), iteration)
                            writer.add_image('LOSS/gt', gt_image.detach().cpu(), iteration)

                        Ll1 = l1_loss(image, gt_image)

                        ssim_loss = (1.0 - ssim(image, gt_image))
                        psnr_v = psnr(image.detach().cuda(), gt_image.detach().cuda()).mean().double()
                        print('PSNR LOSS:', psnr_v)

                        del image, gt_image, render_pkg, voxel_visible_mask, visibility_filter, offset_selection_mask, radii, opacity
                        torch.cuda.empty_cache()

                        scaling_reg = scaling.prod(dim=1).mean()
                        loss_render = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + 0.01 * scaling_reg
                        print('Render loss', loss_render.item())
                        epoch_loss_render += loss_render.item()

                        #print('Render loss', loss_render.item())
                        if loss_render_flag:
                            print('Compute Render Loss')
                            loss += loss_render /mini_size


                        epoch_psnr += psnr_v.detach().cpu().item() /mini_size
                        epoch_ssim += ssim_loss.detach().cpu().item() /mini_size

                        epoch_loss += loss.item() /mini_size

                        del loss_render, psnr_v, ssim_loss, scaling_reg
                        torch.cuda.empty_cache()


                loss.backward()

                net_optimizer.step()
                net_optimizer.zero_grad(set_to_none=True)

                gc.collect()  # Forces Python's garbage collector to free memory


                for p in net.parameters():
                    if hasattr(p, 'grad') and p.grad is not None:
                        print(p, 'delete')
                        del p.grad

                print('after delete')
                print(
                    f"{(torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory) * 100:.2f}%")

                del loss
                del incomplete_tensor, ground_truth_tensor, out_cls, targets, batch, cm, target_key
                torch.cuda.empty_cache()

                gc.collect()  # Trigger garbage collectioncollectpartial


        writer.add_scalar('Loss/loss', epoch_loss/ num_samples, epoch)
        writer.add_scalar('PSNR_LOSS/train_epoch', epoch_psnr / num_samples, epoch)
        writer.add_scalar('RenderLoss/train_epoch', epoch_loss_render /num_samples , epoch)

        writer.add_scalar('SSIM_LOSS/train_epoch', epoch_ssim /num_samples , epoch)

        writer.add_scalar('FeatLoss/train_epoch', epoch_loss_feat /num_samples , epoch)

        writer.add_scalar('PresLoss/train_epoch', epoch_loss_pres /num_samples , epoch)

        current_lr = net_optimizer.param_groups[0]['lr']
        writer.add_scalar('LR', current_lr, epoch)

        scheduler.step()

        epoch_loss = 0
        epoch_loss_feat = 0
        epoch_loss_pres = 0
        epoch_loss_render = 0
        pred_var = 0
        epoch_psnr = 0
        epoch_ssim = 0
        eval_iou_loss = 0

        if epoch % eval_frequency == 0:
            print('EVAl\n:')
            eval_pres_loss = 0
            eval_feat_loss = 0
            eval_render_loss = 0
            eval_psnr_loss = 0
            eval_ssim_loss = 0
            eval_iou_loss = 0

            with torch.no_grad():
                for i, batch in enumerate(eval_dataloader):

                    #incomplete_tensor, ground_truth_tensor = batch['incomplete'], batch['ground_truth']
                    incomplete_tensor, ground_truth_tensor, scenes_info = batch

                    cm = incomplete_tensor.coordinate_manager
                    target_key, _ = cm.insert_and_map(
                        ground_truth_tensor.C,
                        string_id="target",
                    )
                    out_cls, targets, _ = net(incomplete_tensor, target_key)

                    if pres_eval:

                        num_layers, pres_loss = len(out_cls), 0
                        losses = []
                        # Optional weighting scheme for layer losses
                        weights = torch.tensor(
                            [float(1) for i in
                             range(num_layers)])  # torch.tensor([(i + 1) ** 8 for i in range(num_layers)])
                        # weights = weights / weights.sum()
                        # Compute presence loss across all layers
                        l = 0
                        for out_cl, target in zip(out_cls, targets):
                            curr_loss = crit(out_cl.F[:, 0].squeeze(), target.type(out_cl.F.dtype).to('cuda'))
                            curr_loss *= weights[l]
                            losses.append(curr_loss.item())
                            print(f'press loss {l},{curr_loss.item()}')
                            pres_loss += curr_loss / num_layers
                            l += 1
                        eval_pres_loss += pres_loss.item()

                        for b in range(1):
                            print(out_cls[-1].C.shape, out_cls[-1].C[1], b)

                            keep = (out_cls[-1].C[:, :1] == b).squeeze()
                            out_cls_b = pruning(out_cls[-1], keep)

                            print(out_cls_b.C.shape)
                            keep = (out_cls_b.F[:, :1] > 0).squeeze()
                            out_cls_pruned = pruning(out_cls_b, keep)

                            keep = (ground_truth_tensor.C[:, :1] == b).squeeze()
                            ground_truth_cls_b = pruning(ground_truth_tensor, keep)

                            A = out_cls_pruned.C[:, 1:]
                            B = ground_truth_cls_b.C[:, 1:]

                            set_A = set([tuple(coord.tolist()) for coord in A])
                            set_B = set([tuple(coord.tolist()) for coord in B])

                            num_intersection = len(set_A.intersection(set_B))

                            num_union = len(set_A.union(set_B))

                            iou = num_intersection / num_union
                            eval_iou_loss += iou
                            print('len', num_intersection, num_union)
                            print('IOU', b, iou)

                    if feat_eval:
                        keep = (out_cls[-1].F[:, :1] > 0).squeeze()
                        out_cls_pruned = pruning(out_cls[-1], keep)

                        # Find matching anchors
                        A = out_cls_pruned.C[:, 1:]
                        B = ground_truth_tensor.C[:, 1:]

                        A_indices, B_indices = find_matching_indices(A, B)

                        if len(A_indices) > 0:
                            mapped_features = out_cls_pruned.F[A_indices, 1:]

                            ground_truth_features = ground_truth_tensor.F[B_indices, 1:]

                            mask = (torch.sum(torch.abs(ground_truth_features), dim=0) != 0).float()
                            mask = mask.unsqueeze(0).expand(ground_truth_features.shape)

                            mse_loss_per_element = (mapped_features - ground_truth_features) ** 2

                            masked_loss = mse_loss_per_element * mask

                            feature_loss = masked_loss.sum() / mask.sum()

                            eval_feat_loss += feature_loss.item()

                    if render_eval:

                        train_cameras = scenes_info[0]['test_cameras']
                        viewpoint_stack = None
                        if not viewpoint_stack:
                            viewpoint_stack = train_cameras.copy()

                        keep = (out_cls[-1].F[:, :1] > 0).squeeze()
                        out_cls_pruned = pruning(out_cls[-1], keep)

                        features = out_cls_pruned.F[:, 1:]
                        anchors = out_cls_pruned.C[:, 1:].float() * dataset_config['voxel_size']

                        gaussians = GaussianSmall(features, anchors.detach(), mlps)

                        mini_size = 16
                        for mini in range(mini_size):
                            bg_color = [0, 0, 0]
                            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

                            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

                            voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background)

                            retain_grad = True  # (iteration < opt.update_until and iteration >= 0)
                            render_pkg = render(viewpoint_cam, gaussians, pipe, background, is_training=True,
                                                visible_mask=voxel_visible_mask,
                                                retain_grad=retain_grad)

                            image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = \
                                render_pkg[
                                    "render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], \
                                render_pkg[
                                    "selection_mask"], \
                                    render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]

                            gt_image = viewpoint_cam.original_image.cuda()

                            # writer.add_image('LOSS/pred', image, epoch)
                            # writer.add_image('LOSS/gt', gt_image, epoch)
                            if mini == 0:
                                writer.add_image(f'EVAL/pred_{i}', image.detach().cpu(), epoch)
                                writer.add_image(f'EVAL/gt_{i}', gt_image.detach().cpu(), epoch)

                            Ll1 = l1_loss(image, gt_image)

                            ssim_loss = (1.0 - ssim(image, gt_image))
                            psnr_v = psnr(image.detach().cuda(), gt_image.detach().cuda()).mean().double()

                            del image, gt_image, render_pkg, voxel_visible_mask, visibility_filter, offset_selection_mask, radii, opacity
                            torch.cuda.empty_cache()

                            scaling_reg = scaling.prod(dim=1).mean()
                            loss_render = (
                                                      1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + 0.01 * scaling_reg
                            epoch_loss_render += loss_render.item()

                            # print('Render loss', loss_render.item())
                            eval_render_loss += loss_render / mini_size

                            eval_psnr_loss += psnr_v.detach().cpu().item() / mini_size

                            eval_ssim_loss += ssim_loss.detach().cpu().item() / mini_size

                            del loss_render, psnr_v, ssim_loss, scaling_reg
                            torch.cuda.empty_cache()

                num_evals = len(eval_dataloader)
                writer.add_scalar('EVAL/pres', eval_pres_loss / num_evals , epoch)
                writer.add_scalar('EVAL/feats', eval_feat_loss / num_evals, epoch)
                writer.add_scalar('EVAL/render', eval_render_loss / num_evals, epoch)
                writer.add_scalar('EVAL/psnr', eval_psnr_loss / num_evals, epoch)
                writer.add_scalar('EVAL/ssim', eval_ssim_loss / num_evals, epoch)
                writer.add_scalar(f'EVAL/IOU', eval_iou_loss / num_evals, epoch)


                del eval_pres_loss, eval_feat_loss, eval_render_loss, eval_psnr_loss, eval_ssim_loss
                torch.cuda.empty_cache()




    writer.close()

def find_matching_indices(tensor1, tensor2):
    # Convert tensor1 to tuple format and store indices in a dictionary
    coord_dict = {tuple(coord.tolist()): i for i, coord in enumerate(tensor1)}

    indices1 = []
    indices2 = []

    # Iterate over tensor2 and find matching coordinates
    for i, coord in enumerate(tensor2):
        coord_tuple = tuple(coord.tolist())
        if coord_tuple in coord_dict:
            indices1.append(coord_dict[coord_tuple])  # Index from tensor1
            indices2.append(i)  # Index from tensor2

    return torch.tensor(indices1), torch.tensor(indices2)


def evaluate(eval_files=None, dataset_config=None,mlps=None,  model_path='weights_005_sep.pth', exp_name='', eval_iou=True, eval_img=True, render_video=False, split='test', visualize=False, writer_name=None):
    print(exp_name)
    # Load Model
    net = CompletionNetSmaller(feat_size=70)

    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    print(f"Total number of learnable parameters: {total_params}")

    start_epoch = 0
    print(model_path)
    if os.path.exists(model_path):
        print('Loading model from {}'.format(model_path))
        net.load_state_dict(torch.load(model_path)['state_dict'])
        start_epoch = torch.load(model_path)['curr_iter']
        print('Restarting from epoch {}'.format(start_epoch))
    else:
        print('Model path not found!')

    net.to('cuda')
    net.eval()

    writer = SummaryWriter(f'{writer_name}/{exp_name}')

    eval_dataset = PointCloudDatasetBatched(eval_files, dataset_config)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn_batched)

    pruning = ME.MinkowskiPruning()

    parser = argparse.ArgumentParser(description="Training script parameters")
    pp = PipelineParams(parser)
    op = OptimizationParams(parser)
    args = parser.parse_args([])
    pipe = pp.extract(args)
    opt = op.extract(args)

    full_psnr = 0
    full_iou = 0
    full_ssim = 0
    full_iou_inc = 0

    num_samples= len(eval_dataloader)
    print('FINAL EVAl\n:')

    with torch.no_grad():
        for i, batch in enumerate(eval_dataloader):

            incomplete_tensor, ground_truth_tensor, scenes_info = batch

            cm = incomplete_tensor.coordinate_manager
            target_key, _ = cm.insert_and_map(
                ground_truth_tensor.C,
                string_id="target",
            )

            out_cls, targets, _ = net(incomplete_tensor, target_key)

            if eval_iou:

                for b in range(1):

                    keep = (out_cls[-1].C[:, :1] == b).squeeze()
                    out_cls_b = pruning(out_cls[-1], keep)

                    print(targets[-1].shape)
                    keep = targets[-1] #(out_cls_b.F[:, :1] > 0).squeeze()
                    out_cls_pruned = pruning(out_cls_b, keep)

                    keep = (ground_truth_tensor.C[:, :1] == b).squeeze()
                    ground_truth_cls_b = pruning(ground_truth_tensor, keep)

                    A = out_cls_pruned.C[:, 1:]
                    B = ground_truth_cls_b.C[:, 1:]

                    set_A = set([tuple(coord.tolist()) for coord in A])
                    set_B = set([tuple(coord.tolist()) for coord in B])

                    num_intersection = len(set_A.intersection(set_B))

                    num_union = len(set_A.union(set_B))

                    iou = num_intersection / num_union
                    full_iou += iou / num_samples
                    print('IOU', b, iou)

                    if visualize:
                        print('Visualize EVAL\n\n\n')
                        ll = []
                        for i in [2]:  # range(len(out_cls)): #Only last layer change for all layers
                            for b in range(1):
                                print(b)
                                keep = (out_cls[i].C[:, :1] == b).squeeze()
                                out_cls_b = pruning(out_cls[i], keep)

                                # Predicted anchors
                                keep = targets[-1] #(out_cls_b.F[:, :1] > 0).squeeze()

                                print(keep.shape)

                                out_cls_pruned = pruning(out_cls_b, keep)
                                points_matched_t = (out_cls_pruned.C[:, 1:] + torch.tensor(
                                    [0, b * 300, 0]).cuda()).cpu().detach().numpy()
                                pc1 = PointCloud(points_matched_t)

                                # Input tensor
                                keep = (incomplete_tensor.C[:, :1] == b).squeeze()
                                points_in = incomplete_tensor.C[keep][:, 1:]
                                pc3 = PointCloud(
                                    (points_in + torch.tensor([-300, b * 300, 0]).cuda()).cpu().detach().numpy())

                                # Ground truth tensor
                                keep = (ground_truth_tensor.C[:, :1] == b).squeeze()
                                points_in = ground_truth_tensor.C[keep][:, 1:]
                                pc5 = PointCloud(
                                    (points_in + torch.tensor([300, b * 300, 0]).cuda()).cpu().detach().numpy())

                                ll.append(pc1)
                                ll.append(pc3)
                                ll.append(pc5)

                        o3d.visualization.draw_geometries(ll)
                        ll.clear()

            if eval_img:
                train_cameras = scenes_info[0]['test_cameras']
                num_imgs =len(train_cameras)
                path = scenes_info[0]['ground_truth_path']

                keep = targets[-1] #(out_cls[-1].F[:, :1] > 0).squeeze()
                out_cls_pruned = pruning(out_cls[-1], keep)

                features = out_cls_pruned.F[:, 1:]
                anchors = out_cls_pruned.C[:, 1:].float() * dataset_config['voxel_size']

                gaussians = GaussianSmall(features, anchors.detach(), mlps)

                incfeatures = incomplete_tensor.F[:, 1:]
                incanchors = incomplete_tensor.C[:, 1:].float() * dataset_config['voxel_size']

                inc_gaussians = GaussianSmall(incfeatures, incanchors.detach(), mlps)

                if render_video:
                    video_path = os.path.join('videos', exp_name, path, 'predicted')
                    os.makedirs(video_path, exist_ok=True)

                    center_cam = train_cameras[0]
                    cameras = scenes_info[0]['train_cameras']#generate_rotated_cameras(center_cam)
                    interpolated_cameras = interpolate_cameras(cameras)

                    for view, viewpoint_cam in enumerate(interpolated_cameras): #range(100):
                        bg_color = [0, 0, 0]
                        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


                        voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background,
                                                             no_scaffold=True)

                        retain_grad = True  # (iteration < opt.update_until and iteration >= 0)
                        render_pkg = render(viewpoint_cam, gaussians, pipe, background, is_training=True,
                                            visible_mask=voxel_visible_mask,
                                            retain_grad=retain_grad, no_scaffold=True)

                        image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = \
                            render_pkg[
                                "render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], \
                                render_pkg[
                                    "selection_mask"], \
                                render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]

                        # Convert to HWC format
                        image_np = image.permute(1, 2, 0).detach().cpu().numpy()

                        # **Clip the values to [0, 1] range**
                        image_np = np.clip(image_np, 0, 1)

                        # Scale to [0, 255] and convert to uint8
                        image_np = (image_np * 255).astype('uint8')

                        # Create an image from the array
                        image = Image.fromarray(image_np)

                        # Save image to disk
                        full_path = os.path.join(video_path, f'room_{str(view).zfill(3)}.png')
                        image.save(full_path)
                        print(f"Saved image: {full_path}")

                    video_path = os.path.join('videos', exp_name, path, 'incomplete')
                    os.makedirs(video_path, exist_ok=True)

                    for view, viewpoint_cam in enumerate(interpolated_cameras): #range(100):
                        bg_color = [0, 0, 0]
                        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

                        voxel_visible_mask = prefilter_voxel(viewpoint_cam, inc_gaussians, pipe, background,
                                                             no_scaffold=True)

                        retain_grad = True  # (iteration < opt.update_until and iteration >= 0)
                        render_pkg = render(viewpoint_cam, inc_gaussians, pipe, background, is_training=True,
                                            visible_mask=voxel_visible_mask,
                                            retain_grad=retain_grad, no_scaffold=True)

                        image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = \
                            render_pkg[
                                "render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], \
                                render_pkg[
                                    "selection_mask"], \
                                render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]

                        # Convert to HWC format
                        image_np = image.permute(1, 2, 0).detach().cpu().numpy()
                        image_np = np.clip(image_np, 0, 1)
                        image_np = (image_np * 255).astype('uint8')
                        image = Image.fromarray(image_np)

                        # Save image to disk
                        full_path = os.path.join(video_path, f'room_{str(view).zfill(3)}.png')
                        image.save(full_path)
                        print(f"Saved image: {full_path}")



                viewpoint_stack = None
                if not viewpoint_stack:
                    viewpoint_stack = train_cameras.copy()


                gtfeatures = ground_truth_tensor.F[:, 1:]
                gtanchors = ground_truth_tensor.C[:, 1:].float() * dataset_config['voxel_size']

                gt_gaussians = GaussianSmall(gtfeatures, gtanchors.detach(), mlps)

                eval_psnr = 0
                eval_ssim = 0
                print(path)

                for view in range(num_imgs):
                    bg_color = [0, 0, 0]
                    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

                    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

                    voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background,
                                                         no_scaffold=True)

                    retain_grad = True  # (iteration < opt.update_until and iteration >= 0)
                    render_pkg = render(viewpoint_cam, gaussians, pipe, background, is_training=True,
                                        visible_mask=voxel_visible_mask,
                                        retain_grad=retain_grad, no_scaffold=True)

                    image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = \
                        render_pkg[
                            "render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], \
                            render_pkg[
                                "selection_mask"], \
                            render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]


                    gt_image = viewpoint_cam.original_image.cuda()

                    ssim_loss = (1.0 - ssim(image, gt_image))
                    psnr_v = psnr(image.detach().cuda(), gt_image.detach().cuda()).mean().double()

                    torch.cuda.empty_cache()

                    eval_psnr += psnr_v.detach().cpu().item() / num_imgs

                    eval_ssim += ssim_loss.detach().cpu().item() / num_imgs
                    print(f'{view}: PSNR: {psnr_v:.4f}, SSIM: {ssim_loss.detach().cpu().item():.4f}')

                    voxel_visible_mask = prefilter_voxel(viewpoint_cam, gt_gaussians, pipe, background,
                                                         no_scaffold=True)

                    retain_grad = True  # (iteration < opt.update_until and iteration >= 0)
                    render_pkg = render(viewpoint_cam, gt_gaussians, pipe, background, is_training=True,
                                        visible_mask=voxel_visible_mask,
                                        retain_grad=retain_grad, no_scaffold=True)

                    gt_rec_image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = \
                        render_pkg[
                            "render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], \
                            render_pkg[
                                "selection_mask"], \
                            render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]

                    voxel_visible_mask = prefilter_voxel(viewpoint_cam, inc_gaussians, pipe, background,
                                                         no_scaffold=True)

                    retain_grad = True  # (iteration < opt.update_until and iteration >= 0)
                    render_pkg = render(viewpoint_cam, inc_gaussians, pipe, background, is_training=True,
                                        visible_mask=voxel_visible_mask,
                                        retain_grad=retain_grad, no_scaffold=True)

                    inc_rec_image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = \
                        render_pkg[
                            "render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], \
                            render_pkg[
                                "selection_mask"], \
                            render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]


                    combined_image = torch.cat((gt_image, inc_rec_image, gt_rec_image, image), dim=2)  # Concatenate on width

                    writer.add_image(f'FINALEVAL_{split}/{path}/{view}', combined_image.detach().cpu(), 0)

                writer.add_scalar(f'FINALEVAL_{split}/{path}/PSNR', eval_psnr, 0)
                writer.add_scalar(f'FINALEVAL_{split}/{path}/SSIM', eval_ssim, 0)
                print(f'Results: {exp_name}, {path}, PSNR: {eval_psnr}, SSIM: {eval_ssim}')
                full_psnr += eval_psnr / num_samples
                full_ssim += eval_ssim / num_samples



        print(f'\nFull Dataset Results:')
        if eval_iou:
            print(f'IOU: {full_iou} Before: {full_iou_inc}')
        if eval_img:
            print(f'PSNR: {full_psnr:.4f}, SSIM: {full_ssim:.4f}')




def load_files(file_path):
    """Load file paths from a text file."""
    with open(file_path, 'r') as file:
        return [line.strip() for line in file]

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Train or evaluate a model based on configuration.")
    parser.add_argument("config", type=str, help="Path to the configuration JSON file.")
    parser.add_argument("--evaluate", action="store_true", help="Run the evaluation mode instead of training.")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as config_file:
        config = json.load(config_file)

    # Load train and test file names
    train_files = load_files(config['file_names']['train'])
    test_files = load_files(config['file_names']['val'])

    print(f"Train Files: {len(train_files)} loaded.")
    print(f"Test Files: {len(test_files)} loaded.")

    # Load dataset configuration
    dataset_config = config['dataset_config']

    # Load MLPs
    mlps_path = os.path.join('mlps', dataset_config['mlps_name'])
    mlps = {
        'mlp_opacity': torch.jit.load(os.path.join(mlps_path, 'opacity_mlp.pt')).cuda(),
        'mlp_cov': torch.jit.load(os.path.join(mlps_path, 'cov_mlp.pt')).cuda(),
        'mlp_color': torch.jit.load(os.path.join(mlps_path, 'color_mlp.pt')).cuda()
    }

    if args.evaluate:
        eval_config = config['evaluation']
        print("Starting Evaluation...")
        evaluate(
            eval_files=test_files,
            dataset_config=dataset_config,
            mlps=mlps,
            model_path=eval_config['model_path'],
            exp_name=eval_config['exp_name'],
            eval_iou=eval_config['eval_iou'],
            eval_img=eval_config['eval_img'],
            render_video=eval_config['render_video'],
            visualize=eval_config['visualize'],
            split=eval_config['split'],
            writer_name=config['logging']['writer_name']
        )
    else:
        train_config = config['training']
        print("Starting Training...")
        train(
            train_files=train_files,
            eval_files=test_files,
            dataset_config=dataset_config,
            mlps=mlps,
            loss_config=train_config['loss'],
            model_path=train_config['model_path'],
            save_path=train_config['save_path'],
            exp_name=train_config['exp_name'],
            num_epochs=train_config['num_epochs'],
            lr=train_config['learning_rate'],
            visualize=train_config['visualize'],
            writer_name=config['logging']['writer_name']
        )

if __name__ == "__main__":
    main()
