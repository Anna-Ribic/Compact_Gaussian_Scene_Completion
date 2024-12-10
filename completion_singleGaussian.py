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
from http.cookiejar import request_port
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

import torch.multiprocessing as mp


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


###############################################################################
# End of utility functions
###############################################################################

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


def get_cuda_tensors():
    return {id(obj): obj for obj in gc.get_objects() if isinstance(obj, torch.Tensor) and obj.is_cuda}


class PointCloudDatasetBatched(Dataset):
    def __init__(self, files, config):
        self.dirs = files  # List of (incomplete_path, ground_truth_path) tuples
        self.data_path = config['data_path']
        self.file_pairs = [(f'{config["source_path"]}/{p}_pent/{config["rec_name"]}/gaussians.pth',
                            f'{config["source_path"]}/{p}/{config["rec_name"]}/gaussians.pth') for p in files]
        self.voxel_size = config['voxel_size']  # Size of each voxel in the grid
        self.init_threshold = config['threshold']
        self.normalize = config['normalize']
        self.cache_dir = os.path.join(config['cache_dir'], config['rec_name'])
        self.z_threshold = None
        self.mean = None
        self.std = None
        self.border = None

        self.use_cache = config['use_cache']

        # Ensure cache directory exists
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        self.rotator = RotationZ90()
        self.flipper = RandomMirror()

        self.augmentation = config['augmentation']
        self.rotate = config['rotate']
        self.flip = config['flip']

        self.load_scene_info = config['load_scene_info']
        self.center = config['center']

        parser = argparse.ArgumentParser(description="Process resolution input")

        # Add the '--resolution' argument
        parser.add_argument('--resolution', default=-1, type=int, help='Set the resolution (e.g., 1080)')
        parser.add_argument('--data_device', default='cuda', type=str, help='Device')

        # Parse the command-line arguments
        self.cam_args = parser.parse_args([])

    def __len__(self):
        return len(self.file_pairs)

    def _get_cache_path(self, idx):
        return os.path.join(self.cache_dir, f'sample_{idx}.pkl')

    def read_ply_file(self, file_path, normalize):
        # Load the saved point cloud data
        data = torch.load(file_path)

        # Extract attributes from the loaded data
        xyz = data["xyz"].numpy()
        offset = data["offset"]
        color = data["color"].numpy()  # Color is always present
        opacity = data["opacity"].numpy()
        scaling = data["scaling"].numpy()
        rotation = data["rotation"].numpy()

        # Define threshold to remove ceiling, use gt threshold if defined
        if self.z_threshold is None:
            self.z_threshold = np.percentile(xyz[:, 2], self.init_threshold)
        # Filter out the points above the threshold
        mask = xyz[:, 2] <= self.z_threshold
        xyz = xyz[mask]
        color = color[mask]
        opacity = opacity[mask]
        scaling = scaling[mask]
        rotation = rotation[mask]
        offset = offset[mask]

        # Discretize point coordinates
        discrete_coords = np.floor(xyz / self.voxel_size).astype(np.int32)

        # Build features: exclude rotation for consistency with ScaffoldGS
        features = np.hstack([
            color,  # Include color
            opacity,
            scaling,
            rotation,
            offset
        ])

        # Normalize features
        if normalize:
            features = (features - self.global_mean) / (self.global_std + 1e-8)
            print(f'Normalize Features mean: {features.mean(axis=0)} '
                  f'features std: {features.std(axis=0)} '
                  f'features_min: {features.min(axis=0)} '
                  f'features_max: {features.max(axis=0)}')

        # Define all points as present initially
        ones_column = np.ones((features.shape[0], 1))
        features_with_ones = np.hstack((ones_column, features))

        # Convert to tensors
        features_tensor = features_with_ones
        coordinates_tensor = discrete_coords

        return features_tensor, coordinates_tensor

    def __getitem__(self, idx):
        cache_path = self._get_cache_path(idx)

        print(idx, cache_path)

        if self.use_cache and os.path.exists(cache_path):
            print(f'Loading from cache: {cache_path}')
            with open(cache_path, 'rb') as f:
                res = pickle.load(f)
            incomplete_features, incomplete_coords = res['incomplete']
            ground_truth_features, ground_truth_coords = res['ground_truth']
        else:
            print(f'Cache not found for index {idx}, processing data.')
            room_dir = self.dirs[idx]
            data_dir = f'{self.data_path}/{room_dir}'
            res = {}
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
                ground_truth_coords, ground_truth_features = self.rotator(ground_truth_coords, ground_truth_features)
                incomplete_coords, incomplete_features = self.rotator(incomplete_coords, incomplete_features)
                self.rotator.resample()
            if self.flip and random.random() < 0.2:
                ground_truth_coords, ground_truth_features = self.flipper(ground_truth_coords, ground_truth_features)
                incomplete_coords, incomplete_features = self.flipper(incomplete_coords, incomplete_features)
                self.flipper.resample()

        res.update({
            'incomplete': (incomplete_features, incomplete_coords),
            'ground_truth': (ground_truth_features, ground_truth_coords),
        })


        print(res['incomplete_path'])

        self.z_threshold = None
        self.mean = None
        self.std = None
        self.border = None

        return res




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
    pred_np = pred.detach().cpu().numpy()
    gt_np = gt.detach().cpu().numpy()

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
        average_squared_distances = compute_average_squared_distances(xyz)
        #self.scaling = torch.log(torch.sqrt(torch.clamp_min(average_squared_distances, 1e-7)))[..., None].repeat(1, 3).requires_grad_(True)
        self.scaling = torch.zeros((xyz.shape[0], 3)).float() + 0.05
        self.scaling = self.scaling.cuda().requires_grad_(True)
        print(self.scaling.shape, self.scaling, self.scaling.max(), self.scaling.min())
        self.rotation = torch.zeros((xyz.shape[0], 4), device="cuda")  # Initialize rotation as identity quaternion
        self.rotation[:, 0] = 1
        self.rotation = self.rotation.requires_grad_(True)

    def from_sparse(self, features, xyz):
        self.xyz = xyz
        self.color = features[:, :3]
        self.opacity = features[:, 3].unsqueeze(1)
        self.scaling = features[:, 4:7]
        self.rotation = features[:, 7:11]
        self.offset = features[:, 11:]


    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr

    def export_to_ply(self, filename):
        """
        Exports the GaussianSmall object as a PLY file with color information.
        Args:
            filename (str): Path to the output PLY file.
        """
        if self.xyz is None or self.color is None:
            raise ValueError("xyz and color attributes must be initialized before exporting to PLY.")

        # Convert tensors to numpy arrays
        vertices = (self.xyz + self.offset).cpu().detach().numpy()
        colors = self.color.cpu().detach().numpy()

        # Ensure colors are in the range [0, 255]
        colors = np.clip(colors * 255, 0, 255).astype(np.uint8)

        # Prepare PLY header
        num_vertices = vertices.shape[0]
        ply_header = f"""ply
format ascii 1.0
element vertex {num_vertices}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
        # Combine vertex positions and colors
        ply_data = np.hstack((vertices, colors))

        # Write PLY file
        with open(filename, 'w') as f:
            f.write(ply_header)
            np.savetxt(f, ply_data, fmt='%f %f %f %d %d %d')

        print(f"PLY file saved to {filename}")

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
from torch.optim.lr_scheduler import StepLR


def image_gradients(image):
    dx = image[:, :, :-1, 1:] - image[:, :, :-1, :-1]
    dy = image[:, :, 1:, :-1] - image[:, :, :-1, :-1]
    return dx, dy

def gradient_loss(pred, gt):
    pred_dx, pred_dy = image_gradients(pred)
    gt_dx, gt_dy = image_gradients(gt)
    return torch.mean(torch.abs(pred_dx - gt_dx) + torch.abs(pred_dy - gt_dy))


def train(train_files=None, eval_files=None, dataset_config=None, loss_config=None,  model_path='weights_005_sep.pth',
          save_path='weights_005_add.pth',
          exp_name="CompletionNet_0.05_sep", num_epochs=200, lr=0.1, visualize=False, writer_name=None):
    mp.set_start_method('spawn')

    print('Experiment:', exp_name)
    # Load Model
    net = CompletionNetSmaller()

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

            for iteration in range(epoch*n_it, (epoch+1)*n_it):
                loss = 0
                print(incomplete_tensor.device)
                print(scenes_info[0]['incomplete_path'])

                out_cls, targets, _ = net(incomplete_tensor, target_key)

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

                    del pres_loss, num_layers, curr_loss

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

                    keep = targets[-1] #(out_cls[-1].F[:, :1] > 0).squeeze()
                    out_cls_pruned = pruning(out_cls[-1], keep)

                    # Find matching anchors
                    A = out_cls_pruned.C[:, 1:]
                    B = ground_truth_tensor.C[:, 1:]

                    A_indices, B_indices = find_matching_indices(A, B)

                    if len(A_indices) > 0:
                        mapped_features = out_cls_pruned.F[A_indices, 1:]

                        ground_truth_features = ground_truth_tensor.F[B_indices, 1:]


                        #mask = (torch.sum(torch.abs(ground_truth_features), dim=0) != 0).float()
                        #mask = mask.unsqueeze(0).expand(ground_truth_features.shape)

                        mse_loss_per_element = (mapped_features - ground_truth_features) ** 2

                        masked_loss = mse_loss_per_element #* mask

                        feature_loss = masked_loss.mean() #/ mask.sum()

                        loss += feature_loss

                        print('Feauture_loss:', feature_loss)

                        epoch_loss_feat += feature_loss.cuda().detach().item()

                    del keep, out_cls_pruned, A, B,mse_loss_per_element, masked_loss, feature_loss, A_indices, B_indices
                    del mapped_features, ground_truth_features
                    torch.cuda.empty_cache()
                    gc.collect()


                if (loss_feat_flag and epoch % logging_frequency and iteration==epoch*n_it) or loss_render_flag:


                    #TODO change this for multi scenes
                    train_cameras = scenes_info[0]['train_cameras']
                    # Pick a random Camera
                    viewpoint_stack = None
                    if not viewpoint_stack:
                        viewpoint_stack = train_cameras.copy()

                    keep = targets[-1]#(out_cls[-1].F[:, :1] > 0).squeeze()
                    out_cls_pruned = pruning(out_cls[-1], keep)

                    features = out_cls_pruned.F[:, 1:]
                    anchors = out_cls_pruned.C[:, 1:].float() * dataset_config['voxel_size']

                    gaussians = GaussianSmall()
                    gaussians.from_sparse(features, anchors.detach())

                    print('feature', features.shape, anchors.shape)


                    for mini in range(mini_size):
                        bg_color = [0, 0, 0]
                        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

                        #train_cameras = batch['train_cameras']
                        # Pick a random Camera
                        #viewpoint_stack = None
                        #if not viewpoint_stack:
                        #    viewpoint_stack = train_cameras.copy()
                        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

                        voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background, no_scaffold=True)

                        retain_grad = True  # (iteration < opt.update_until and iteration >= 0)
                        render_pkg = render(viewpoint_cam, gaussians, pipe, background, is_training=True,
                                            visible_mask=voxel_visible_mask,
                                            retain_grad=retain_grad, no_scaffold=True)

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

                        print('images', image.shape, gt_image.shape)
                        Ll1 = l1_loss(image, gt_image)

                        ssim_loss = (1.0 - ssim(image, gt_image))
                        psnr_v = psnr(image.detach().cuda(), gt_image.detach().cuda()).mean().double()
                        print('PSNR LOSS:', psnr_v)

                        del image, gt_image, render_pkg, voxel_visible_mask, visibility_filter, offset_selection_mask, radii, opacity
                        torch.cuda.empty_cache()

                        scaling_reg = scaling.prod(dim=1).mean()
                        print(f'ssim:  {ssim_loss}, psnr: {psnr_v}, l1: {Ll1}, scaling_reg: {scaling_reg}')
                        loss_render = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss #+ 0.01 * scaling_reg
                        print('Render loss', loss_render.item())
                        epoch_loss_render += loss_render.detach().cpu().item()

                        #print('Render loss', loss_render.item())
                        if loss_render_flag:
                            print('Compute Render Loss')
                            loss += loss_render /mini_size


                        epoch_psnr += psnr_v.detach().cpu().item() /mini_size
                        epoch_ssim += ssim_loss.detach().cpu().item() /mini_size

                        epoch_loss += loss.detach().cpu().item() /mini_size

                        del loss_render, psnr_v, ssim_loss, scaling_reg
                        torch.cuda.empty_cache()

                    del features, keep, gaussians, anchors
                    torch.cuda.empty_cache()

                # Backward pass and optimize
                loss.backward()

                net_optimizer.step()
                net_optimizer.zero_grad(set_to_none=True)
                gc.collect()  # Forces Python's garbage collector to free memory


                for p in net.parameters():
                    if hasattr(p, 'grad') and p.grad is not None:
                        print(delete)
                        p.grad = None
                        del p.grad

                del loss, incomplete_tensor, ground_truth_tensor, out_cls, targets, cm, target_key
                torch.cuda.empty_cache()


        writer.add_scalar('Loss/loss', epoch_loss/ num_samples, epoch)
        writer.add_scalar('PSNR_LOSS/train_epoch', epoch_psnr / num_samples, epoch)
        writer.add_scalar('RenderLoss/train_epoch', epoch_loss_render /num_samples , epoch)

        writer.add_scalar('SSIM_LOSS/train_epoch', epoch_ssim /num_samples , epoch)

        writer.add_scalar('FeatLoss/train_epoch', epoch_loss_feat /num_samples , epoch)

        writer.add_scalar('PresLoss/train_epoch', epoch_loss_pres /num_samples , epoch)

        current_lr = net_optimizer.param_groups[0]['lr']
        writer.add_scalar('LR', current_lr , epoch)

        scheduler.step()

        epoch_loss = 0
        epoch_loss_feat = 0
        epoch_loss_pres = 0
        epoch_loss_render = 0
        pred_var = 0
        epoch_psnr = 0
        epoch_ssim = 0

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

                    incomplete_tensor, ground_truth_tensor, scenes_info = batch

                    cm = incomplete_tensor.coordinate_manager
                    target_key, _ = cm.insert_and_map(
                        ground_truth_tensor.C,
                        string_id="target",
                    )
                    print(scenes_info[0]['incomplete_path'])
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
                            eval_iou_loss += iou
                            print('len', num_intersection, num_union)
                            print('IOU', b, iou)

                    if feat_eval:
                        keep = targets[-1] #(out_cls[-1].F[:, :1] > 0).squeeze()
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

                        keep = targets[-1]#(out_cls[-1].F[:, :1] > 0).squeeze()
                        out_cls_pruned = pruning(out_cls[-1], keep)

                        features = out_cls_pruned.F[:, 1:]
                        anchors = out_cls_pruned.C[:, 1:].float() * dataset_config['voxel_size']

                        gaussians = GaussianSmall()
                        gaussians.from_sparse(features, anchors.detach())

                        num_imgs = len(train_cameras)
                        for mini in range(num_imgs):
                            bg_color = [0, 0, 0]
                            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

                            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

                            voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background, no_scaffold=True)

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

                            # writer.add_image('LOSS/pred', image, epoch)
                            # writer.add_image('LOSS/gt', gt_image, epoch)
                            if mini == 0:
                                writer.add_image(f'EVAL/pred_{i}/{mini}', image.detach().cpu(), epoch)
                                writer.add_image(f'EVAL/gt_{i}/{mini}', gt_image.detach().cpu(), epoch)

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
                            eval_render_loss += loss_render / num_imgs

                            eval_psnr_loss += psnr_v.detach().cpu().item() / num_imgs

                            eval_ssim_loss += ssim_loss.detach().cpu().item() / num_imgs

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

from io import BytesIO
from utils.graphics_utils import fov2focal
import json


def camera_to_JSON(id, camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.image_width,
        'height' : camera.image_height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FoVy, camera.image_height),
        'fx' : fov2focal(camera.FoVx, camera.image_width)
    }
    return camera_entry

def process_ply_to_splat(ply_file_path):
    plydata = PlyData.read(ply_file_path)
    vert = plydata["vertex"]
    sorted_indices = np.argsort(
        -np.exp(vert["scale_0"] + vert["scale_1"] + vert["scale_2"])
        / (1 + np.exp(-vert["opacity"]))
    )
    buffer = BytesIO()
    for idx in sorted_indices:
        v = plydata["vertex"][idx]
        position = np.array([v["x"], v["y"], v["z"]], dtype=np.float32)
        scales = np.exp(
            np.array(
                [v["scale_0"], v["scale_1"], v["scale_2"]],
                dtype=np.float32,
            )
        )
        rot = np.array(
            [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]],
            dtype=np.float32,
        )
        SH_C0 = 0.28209479177387814
        color = np.array(
            [
                0.5 + SH_C0 * v["f_dc_0"],
                0.5 + SH_C0 * v["f_dc_1"],
                0.5 + SH_C0 * v["f_dc_2"],
                1 / (1 + np.exp(-v["opacity"])),
            ]
        )


        buffer.write(position.tobytes())
        buffer.write(scales.tobytes())
        buffer.write((color * 255).clip(0, 255).astype(np.uint8).tobytes())
        buffer.write(
            ((rot / np.linalg.norm(rot)) * 128 + 128)
            .clip(0, 255)
            .astype(np.uint8)
            .tobytes()
        )

    return buffer.getvalue()

from scene.cameras import Camera
def generate_rotated_cameras(camera, num_cameras=100):
    """
    Generate `num_cameras` rotated instances of a given Camera object by rotating around its Z-axis.

    Args:
        camera (Camera): An instance of the Camera class to be rotated.
        num_cameras (int): Number of cameras to generate (default: 100).

    Returns:
        list[Camera]: A list of rotated Camera objects.
    """
    # Full rotation in radians
    full_rotation = 2 * np.pi

    # Angle step for each rotation
    angle_step = full_rotation / num_cameras

    # Extract the rotation matrix (R) and translation vector (T) from the input camera
    original_R = camera.R  # Convert to numpy for matrix operations
    original_T = camera.T  # Translation remains constant

    C = -original_R.T @ original_T


    # List to store new Camera objects
    rotated_cameras = []

    for i in range(num_cameras):
        # Compute the rotation angle
        angle = i * angle_step

        print('og', original_R, angle)

        R_z = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ], dtype=np.float32)

        # Apply the Z-axis rotation to the original rotation matrix (R)
        # Choose pre-multiplication or post-multiplication based on your coordinate system
        #new_R = R_z @ original_R  # Rotate around world Z-axis
        new_R = original_R @ R_z  # Rotate around camera's local Z-axis

        # Recompute T to keep the camera center C the same
        new_T = -new_R @ C

        print(new_R)

        # Create a new Camera object with the updated R and the same T
        new_camera = Camera(
            colmap_id=camera.colmap_id,
            R=new_R,
            T=new_T,
            FoVx=camera.FoVx,
            FoVy=camera.FoVy,
            image=camera.original_image,  # Reuse the same image
            gt_alpha_mask=None,  # Reuse existing alpha mask if needed
            image_name=camera.image_name,
            uid=f"{camera.uid}_rot_{i}",  # Give a unique ID to each rotated camera
            data_device=camera.data_device
        )

        rotated_cameras.append(new_camera)

    return rotated_cameras

from scipy.spatial.transform import Rotation as R, Slerp


def interpolate_cameras(cameras):
    """
    Given a list of Camera objects, interpolate between each pair of adjacent cameras
    and insert the interpolated camera between them.

    Args:
        cameras (list of Camera): The original list of Camera objects.

    Returns:
        list of Camera: The new list of Camera objects, including the original cameras
        and the interpolated cameras.
    """
    interpolated_cameras = []

    for i in range(len(cameras) - 1):
        cam1 = cameras[i]
        cam2 = cameras[i + 1]

        # Interpolate between cam1 and cam2
        cam_interp = interpolate_camera(cam1, cam2, alpha=0.5)

        # Append cam1 and cam_interp to the new list
        interpolated_cameras.append(cam1)
        interpolated_cameras.append(cam_interp)

    # Append the last original camera
    interpolated_cameras.append(cameras[-1])

    return interpolated_cameras

def interpolate_camera(cam1, cam2, alpha):
    """
    Interpolate between two Camera objects.

    Args:
        cam1 (Camera): The first camera.
        cam2 (Camera): The second camera.
        alpha (float): Interpolation parameter between 0 and 1.

    Returns:
        Camera: The interpolated camera.
    """
    # Interpolate translations
    T1 = cam1.T
    T2 = cam2.T
    T_interp = (1 - alpha) * T1 + alpha * T2

    # Convert rotation matrices to Rotation objects
    rot1 = R.from_matrix(cam1.R)
    rot2 = R.from_matrix(cam2.R)

    # Set up SLERP interpolation
    times = [0, 1]
    rots = R.from_matrix([cam1.R, cam2.R])

    slerp = Slerp(times, rots)

    # Interpolate at the desired time
    rot_interp = slerp([alpha])[0]

    # Convert back to rotation matrix
    R_interp = rot_interp.as_matrix()

    # Interpolate other parameters if needed
    FoVx_interp = (1 - alpha) * cam1.FoVx + alpha * cam2.FoVx
    FoVy_interp = (1 - alpha) * cam1.FoVy + alpha * cam2.FoVy

    # Generate a new UID for the interpolated camera
    new_uid = f"{cam1.uid}_interp_{cam2.uid}"

    # Create a new Camera object with interpolated parameters
    new_camera = Camera(
        colmap_id=None,
        R=R_interp,
        T=T_interp,
        FoVx=FoVx_interp,
        FoVy=FoVy_interp,
        image=cam1.original_image,             # Set to None or handle as needed
        gt_alpha_mask=None,     # Set to None or handle as needed
        image_name=cam1.image_name,        # Set to None or handle as needed
        uid=new_uid,
        data_device=cam1.data_device  # Assume same device as cam1
    )

    return new_camera


def evaluate(eval_files=None, dataset_config=None,  model_path='weights_005_sep.pth', exp_name='', eval_iou=True, eval_img=True, render_video=False, split='test', visualize=False, writer_name=None):
    print(exp_name)
    # Load Model
    net = CompletionNetSmaller()

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

                gaussians = GaussianSmall()
                gaussians.from_sparse(features, anchors.detach())

                incfeatures = incomplete_tensor.F[:, 1:]
                incanchors = incomplete_tensor.C[:, 1:].float() * dataset_config['voxel_size']

                inc_gaussians = GaussianSmall()
                inc_gaussians.from_sparse(incfeatures, incanchors.detach())

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

                gt_gaussians = GaussianSmall()
                gt_gaussians.from_sparse(gtfeatures, gtanchors.detach())


                eval_psnr = 0
                eval_ssim = 0
                print(path)
                json_cams = []

                for view in range(num_imgs):
                    bg_color = [0, 0, 0]
                    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

                    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

                    json_cams.append(camera_to_JSON(view, viewpoint_cam))
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



from plyfile import PlyData, PlyElement
import numpy as np
def convert_gaussian_to_ply(gaussian_small, output_ply_path):
    # Prepare the data for the PLY file
    vertex_data = []
    for i in range(len(gaussian_small.xyz)):
        position = gaussian_small.xyz[i] + gaussian_small.offset[i]
        scaling = gaussian_small.scaling[i]
        rotation = gaussian_small.rotation[i]
        color = gaussian_small.color[i]
        opacity = gaussian_small.opacity[i, 0]  # Assuming opacity is a 2D array with shape [N, 1]

        # Append the data to the list in the format expected by the PLY file
        vertex_data.append(
            (
                position[0], position[1], position[2],  # x, y, z
                scaling[0], scaling[1], scaling[2],  # scale_0, scale_1, scale_2
                rotation[0], rotation[1], rotation[2], rotation[3],  # rot_0, rot_1, rot_2, rot_3
                color[0], color[1], color[2],  # f_dc_0, f_dc_1, f_dc_2
                opacity,  # opacity
            )
        )

    # Define the data type for the PLY file
    ply_dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),  # Position
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),  # Scaling factors
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),  # Rotation (quaternion)
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),  # SH coefficients for color
        ("opacity", "f4"),  # Opacity
    ]

    # Convert the list of data to a NumPy structured array
    vertex_array = np.array(vertex_data, dtype=ply_dtype)

    # Create the PLY element
    ply_element = PlyElement.describe(vertex_array, "vertex")

    print(len(vertex_data))
    # Write the PLY file
    PlyData([ply_element]).write(output_ply_path)

    print(f"PLY file successfully written to {output_ply_path}")



def load_files(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file]


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Train or evaluate a model based on configuration.")
    parser.add_argument("config", type=str, help="Path to the configuration JSON file.")
    parser.add_argument("--evaluate", action="store_true", help="Run the evaluation mode instead of training.")
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as config_file:
        config = json.load(config_file)

    # Load file names
    train_files = load_files(config['file_names']['train'])
    test_files = load_files(config['file_names']['val'])

    print('Train Files:', len(train_files), train_files)
    print('Test Files:', test_files)

    # Dataset configuration
    dataset_config = config['dataset_config']

    # Check mode from argument or config
    if args.evaluate:
        eval_config = config['evaluation']
        print('Starting Evaluation...')
        evaluate(
            eval_files=test_files,
            dataset_config=dataset_config,
            model_path=eval_config['model_path'],
            exp_name=eval_config['exp_name'],
            eval_iou=eval_config['eval_iou'],
            eval_img=eval_config['eval_img'],
            render_video=eval_config['render_video'],
            visualize=eval_config['visualize'],
            split=eval_config['split'],
            writer_name = config['logging']['writer_name']
        )
    else:
        train_config = config['training']
        print('Starting Training...')
        train(
            train_files=train_files,
            eval_files=test_files,
            dataset_config=dataset_config,
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
