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
import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import *
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal

def readTrajectorySceneInfo(args):
    cameras_extrinsic_file = os.path.join(args.source_path, "ego.npy")
    extrinsics = np.load(cameras_extrinsic_file)
        
    focal = 1000.0
    H = 1080
    W = 1920
    
    cx, cy = W / 2, H / 2
    intrinsic = np.array([
        [focal, 0, cx],
        [0, focal, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    cam_infos_unsorted = []
    for idx in range(extrinsics.shape[0]):
        extr = extrinsics[idx]
        intr = intrinsic
        height = H
        width = W
        
        R = np.transpose(extr[:3, :3])
        T = np.array(extr[:3, -1])
        
        FovY = focal2fov(focal, height)
        FovX = focal2fov(focal, width)
        
        image_path = ''
        image_name = '{:03}.jpg'.format(idx)
        # image = Image.fromarray(np.zeros((height, width, 3)).astype('uint8'))
        image = None
        
        cam_info = CameraInfo(uid=0, global_id=idx, R=R, T=T, FovY=FovY, FovX=FovX, 
                              image_path=image_path, image_name=image_name, width=width, height=height, fx=focal, fy=focal)
        cam_infos_unsorted.append(cam_info)
    
    cam_infos_unsorted = cam_infos_unsorted[150: 600]
    # cam_infos_unsorted = cam_infos_unsorted[::10]
    # test_cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    test_cam_infos = cam_infos_unsorted
    nerf_normalization = getNerfppNorm(test_cam_infos)
    
    ply_path = os.path.join(args.source_path, "sparse/0/points3D.ply")
    bin_path = os.path.join(args.source_path, "sparse/0/points3D.bin")
    txt_path = os.path.join(args.source_path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(manifold=pcd,
                           train_cameras=test_cam_infos[:1],
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info
        

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        scene_info = readTrajectorySceneInfo(args)

        if not self.loaded_iter:
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

        args.preload_img = False
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            # self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            self.gaussians.create_from_manifold(scene_info.manifold, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]