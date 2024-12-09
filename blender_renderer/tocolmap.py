import os
import argparse
import collections
import math
import trimesh
import cv2
import numpy as np
import imageio
from tqdm import tqdm
from PIL import Image
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

class COLMAPImage(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def write_cameras_text(cameras, path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    HEADER = "# Camera list with one line of data per camera:\n" + \
             "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n" + \
             "# Number of cameras: {}\n".format(len(cameras))
    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, cam in cameras.items():
            to_write = [cam.id, cam.model, cam.width, cam.height, *cam.params]
            line = " ".join([str(elem) for elem in to_write])
            fid.write(line + "\n")

def write_images_text(images, path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    with open(path, "w") as fid:
        # fid.write(HEADER)
        for _, img in images.items():
            image_header = [img.id, *img.qvec, *img.tvec, img.camera_id, img.name]
            first_line = " ".join(map(str, image_header))
            fid.write(first_line + "\n")

            points_strings = []
            fid.write(" ".join(points_strings) + "\n")

def write_points3D_text(points3D, path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    with open(path, "w") as fid:
        pass

def write_model(cameras, images, points3D, path, ext=".txt"):
    if ext == ".txt":
        write_cameras_text(cameras, os.path.join(path, "cameras" + ext))
        write_images_text(images, os.path.join(path, "images" + ext))
        write_points3D_text(points3D, os.path.join(path, "points3D") + ext)
    else:
        raise NotImplementedError
    return cameras, images, points3D

def build_colmap_cameras(f, cx, cy, height, width):
    params = [f, f, cx, cy]
    camera_id = 1
    cameras = {}
    cameras[camera_id] = Camera(id=camera_id,
                                model="PINHOLE",
                                width=int(width),
                                height=int(height),
                                params=np.array(params))
    return cameras

def build_colmap_images(view_matrices):
    '''
    :param view_matrices: [N, 4, 4], dtype: np.ndarray
    '''
    images = {}
    camera_id = 1
    for vid in range(view_matrices.shape[0]):
        image_id = vid + 1
        qvec = rotmat2qvec(view_matrices[vid, :3, :3])
        tvec = view_matrices[vid, :3, 3]
        images[image_id] = COLMAPImage(
            id=image_id, qvec=qvec, tvec=tvec,
            camera_id=camera_id, name="{:03}.png".format(vid),
            xys=None, point3D_ids=None)
    return images

def create_argparser():
    parser = argparse.ArgumentParser()
    # I/O
    parser.add_argument("--out_path", type=str, default="/media/womoer/Wdata/aRobotics/Relit/blender_output/airbot_act10/right",
                        help='directory that saves the results')
    # intrinsics
    parser.add_argument("--resolution", type=int, default=512, help="resolution of the rendering")
    parser.add_argument("--lens", type=int, default=40, help="focal length in mm")
    parser.add_argument("--sensor_size", type=int, default=32, help="focal length in mm")
    return parser

if __name__ == "__main__":
    args = create_argparser().parse_args()
    res_pose_path = os.path.join(args.out_path, 'sparse')
    os.makedirs(res_pose_path, exist_ok=True)

    cam_path = os.path.join(args.out_path, "poses")
    depth_path = os.path.join(args.out_path, "depths")

    obj_path = os.path.join(args.out_path, "1.obj")
    mesh = trimesh.load_mesh(obj_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_mesh()
    print('Successfully loading mesh')

    verts = mesh.vertices
    faces = mesh.faces
    bbox_max = np.max(verts, axis=0)
    bbox_min = np.min(verts, axis=0)
    scale = (bbox_max - bbox_min).max()
    center = (bbox_max + bbox_min) / 2

    focal_mm = args.lens
    height = args.resolution
    width = args.resolution
    sensor_width = args.sensor_size
    sensor_height = args.sensor_size
    focal = focal_mm * width / sensor_width
    cx, cy = width / 2, height / 2
    zn = 0.001  # z_near for open-gl clip space
    zf = 1000.0  # z_far for open-gl clip space
    print("focal: ", focal)

    colmap_cameras = build_colmap_cameras(f=focal, cx=cx, cy=cy, height=height, width=width)

    cam_items = sorted(os.listdir(cam_path))
    colmap_view_mats = []
    mv_template = np.eye(4)

    for i, cam_name in tqdm(enumerate(cam_items)):
        bdpt = cv2.imread(os.path.join(depth_path, cam_name.split('.')[0] + '0001.exr'), cv2.IMREAD_UNCHANGED)
        extr = np.load(os.path.join(cam_path, cam_name))

        R_c2w = extr[:3, :3].transpose(1, 0)
        t_c2w = -R_c2w @ extr[:3, 3:]
        t_c2w = t_c2w * scale + center[:, None]
        t_w2c = -extr[:3, :3] @ t_c2w
        extr[:3, 3:] = t_w2c
        mv = mv_template.copy()
        mv[:3] = extr
        colmap_view_mats.append(mv)

        bdpt = bdpt * scale
        os.system('rm -rf ' + os.path.join(depth_path, cam_name.split('.')[0] + '0001.exr'))
        cv2.imwrite(os.path.join(depth_path, cam_name.split('.')[0] + '0001.exr'), bdpt)

    colmap_view_mats = np.stack(colmap_view_mats, axis=0)

    colmap_images = build_colmap_images(view_matrices=colmap_view_mats)
    write_model(colmap_cameras, colmap_images, points3D=None, path=res_pose_path)
