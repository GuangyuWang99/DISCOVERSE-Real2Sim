import os
import collections
import struct
import torch
import math
import cv2
import numpy as np
import nvdiffrast.torch as dr
import trimesh
from tqdm import tqdm
from PIL import Image

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

def build_colmap_cameras(f, cx, cy):
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
            camera_id=camera_id, name="{:05}.jpg".format(vid),
            xys=None, point3D_ids=None)
    return images

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def pcwrite(filename, xyzrgb):
    """Save a point cloud to a polygon .ply file.
    """
    xyz = xyzrgb[:, :3]
    rgb = xyzrgb[:, 3:].astype(np.uint8)

    # Write header
    ply_file = open(filename,'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n"%(xyz.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(xyz.shape[0]):
        ply_file.write("%f %f %f %d %d %d\n"%(
            xyz[i, 0], xyz[i, 1], xyz[i, 2],
            rgb[i, 0], rgb[i, 1], rgb[i, 2],
        ))

def render_pose_axis(poses, unit_length=0.1):
    """
    poses: c2w matrix in opencv manner
    unit_length (m:meter): unit axis length for visualization
    axis-x, axis-y, axis-z: red, green, blue
    """
    pose_coord_x = poses[:, :3, -1] + poses[:, :3, 0] * unit_length
    pose_coord_y = poses[:, :3, -1] + poses[:, :3, 1] * unit_length
    pose_coord_z = poses[:, :3, -1] + poses[:, :3, 2] * unit_length
    poses_vis = np.concatenate([poses[:, :3, -1], pose_coord_x, pose_coord_y, pose_coord_z], axis=0)
    poses_rgb = np.concatenate([np.ones([poses.shape[0], 3])*255,
                                np.ones([poses.shape[0], 3])*np.array([255, 0, 0]),
                                np.ones([poses.shape[0], 3])*np.array([0, 255, 0]),
                                np.ones([poses.shape[0], 3])*np.array([0, 0, 255]),
                                ])
    # pcwrite("camera_raw_axis.ply", np.concatenate([poses_vis, poses_rgb], axis=1))
    return poses_vis, poses_rgb

if __name__ == "__main__":
    """
    Hemispherical Uniform Sampling for Mesh-to-GS pipeline using `nvdiffrast`.
    """
    radius = 2.5    # radius of the hemisphere to locate cameras
    num_horiz_angle = 10    # num of angles spanning along the xy-plane
    num_verti_angle = 5     # num of angles spanning along the z-plane
    start_verti_angle = 30  # start angle w.r.t. z-axis, defining the range of hemisphere
    final_verti_angle = 90  # final angle w.r.t. z-axis, defining the range of hemisphere
    
    fov = 60 / 180 * np.pi  # camera FOV, can be fixed
    height = 989    # rendering resolution
    width = 1320    # rendering resolution
    cx, cy = width / 2, height / 2
    zn = 0.001      # z_near for open-gl clip space
    zf = 1000.0     # z_far for open-gl clip space
    focal = fov2focal(fov, width)
    print("focal: ", focal)
    
    root_folder = "/data/guangyu/aRobotics/data/objaverse"
    scene_name = "4"
    mesh_path = os.path.join(root_folder, scene_name, 'ImageToStl.com_2167421ddaa44d66bf50c5d4d91b2aa1.obj')
    texture_path = os.path.join(root_folder, scene_name, 'main_baseColor.jpg')
    if texture_path is not None and os.path.exists(texture_path):
        res_rgb_path = os.path.join(root_folder, scene_name, 'buffers/images')
        os.makedirs(res_rgb_path, exist_ok=True)
    vis_normal_path = os.path.join(root_folder, scene_name, 'buffers/vis_n')
    res_normal_path = os.path.join(root_folder, scene_name, 'buffers/normals')
    res_depth_path = os.path.join(root_folder, scene_name, 'buffers/depths')
    res_pose_path = os.path.join(root_folder, scene_name, 'buffers/sparse')
    res_alpha_path = os.path.join(root_folder, scene_name, 'buffers/masks')
    os.makedirs(vis_normal_path, exist_ok=True)
    os.makedirs(res_normal_path, exist_ok=True)
    os.makedirs(res_depth_path, exist_ok=True)
    os.makedirs(res_pose_path, exist_ok=True)
    os.makedirs(res_alpha_path, exist_ok=True)
    
    projection_matrix = torch.tensor(
        [[2*focal/width,   0, (width - 2*cx) / width, 0],
         [0, -2*focal/height, (height - 2*cy)/height, 0],
         [0,    0, -(zf+zn)/(zf-zn), -(2*zf*zn)/(zf-zn)], 
         [0,   0,  -1,  0]], dtype=torch.float32
    )
    
    horiz_angles = torch.linspace(0, 360 * (1 - 1. / num_horiz_angle), steps=num_horiz_angle) / 180 * np.pi
    verti_angles = torch.linspace(start_verti_angle, final_verti_angle, steps=num_verti_angle) / 180 * np.pi
    # print(horiz_angles, verti_angles)
    
    # rotation around z-axis
    quant_horiz = torch.stack(
        [torch.cos(horiz_angles / 2), 0. * torch.sin(horiz_angles / 2), 0. * torch.sin(horiz_angles / 2), 1. * torch.sin(horiz_angles / 2)],
        dim=-1
    )   # [Nh, 4]
    
    # rotation around y-axis 
    quant_verti = torch.stack(
        [torch.cos(verti_angles / 2), 0. * torch.sin(verti_angles / 2), 1. * torch.sin(verti_angles / 2), 0. * torch.sin(verti_angles / 2)],
        dim=-1
    )   # [Nv, 4]
    
    rot_horiz = quaternion_to_matrix(quant_horiz)
    rot_verti = quaternion_to_matrix(quant_verti)
    
    # hemispherical sampling of camera centers
    trans = torch.stack(
        [
            radius * torch.sin(verti_angles[None, ...].repeat(num_horiz_angle, 1)) * torch.cos(horiz_angles[:, None].repeat(1, num_verti_angle)),
            radius * torch.sin(verti_angles[None, ...].repeat(num_horiz_angle, 1)) * torch.sin(horiz_angles[:, None].repeat(1, num_verti_angle)),
            radius * torch.cos(verti_angles[None, ...].repeat(num_horiz_angle, 1))
        ],
        dim=-1
    )   # [Nh, Nv, 3]
    rots = torch.matmul(rot_horiz[:, None, ...], rot_verti[None, ...])  # [Nh, Nv, 3, 3]
    # rots = rot_horiz[:, None, ...].repeat(1, num_verti_angle, 1, 1)    # [Nh, Nv, 3, 3]
    # rots = rot_verti[None, ...].repeat(num_horiz_angle, 1, 1, 1)    # [Nh, Nv, 3, 3]

    # convert to open-gl convention
    rots = rots[..., [1, 0, 2]]
    rots[..., 1] *= -1
    
    colmap_rots = rots.clone()
    colmap_rots[..., 1:3] *= -1
    R_w2c_colmap = colmap_rots.permute(0, 1, 3, 2)
    t_w2c_colmap = torch.matmul(-R_w2c_colmap, trans.unsqueeze(-1))
    view_mats_colmap = torch.cat([R_w2c_colmap, t_w2c_colmap], dim=-1).flatten(0, 1)
    
    # cam w.r.t. world, the camera location & rotation expressed in world coordinate system
    # only for debug visualization
    poses = torch.cat([colmap_rots, trans.unsqueeze(-1)], dim=-1).flatten(0, 1)
    
    # convert from `cam w.r.t. world` to `world w.r.t. cam`
    R_w2c = rots.permute(0, 1, 3, 2)
    t_w2c = torch.matmul(-R_w2c, trans.unsqueeze(-1))
    # open-gl view matrix
    view_mats = torch.cat([R_w2c, t_w2c], dim=-1).flatten(0, 1)
    
    view_mats = torch.cat([view_mats, torch.tensor([[[0., 0., 0., 1.]]]).repeat(view_mats.shape[0], 1, 1)], dim=1)
    # open-gl projection matrix
    projection_mats = projection_matrix[None, ...].repeat(view_mats.shape[0], 1, 1)
    # open-gl full projection matrix, i.e., mvp in nvdiffrast
    view_proj_mats = torch.matmul(projection_mats, view_mats)
    
    mesh = trimesh.load_mesh(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_mesh()
    print('Successfully loading mesh')
    # normalize the mesh into a unit bounding sphere
    center, radius = trimesh.nsphere.minimum_nsphere(mesh.vertices)
    verts = (mesh.vertices - center) / radius
    # -------------------------------------------------------------------------------
    # hard-code for objaverse objects, align `top` directions of objects to z-axis
    flip = np.array([
        [1., 0., 0.],
        [0., 0., 1.],
        [0., 1., 0.]
    ], dtype=np.float32)
    verts = np.matmul(verts, flip)
    mesh.vertices = verts
    mesh.export(os.path.join(root_folder, scene_name, 'buffers/1.obj'))
    # -------------------------------------------------------------------------------

    # poses_vis, poses_rgb = render_pose_axis(poses.numpy(), unit_length=0.05)
    # colors = np.zeros_like(verts)
    # pcwrite("camera_raw_axis_gt.ply", np.concatenate([np.concatenate([poses_vis, verts], axis=0), np.concatenate([poses_rgb, colors], axis=0)], axis=1))
    # exit(0)
    colmap_cameras = build_colmap_cameras(f=focal, cx=cx, cy=cy)
    colmap_images = build_colmap_images(view_matrices=view_mats_colmap.clone().numpy())
    write_model(colmap_cameras, colmap_images, points3D=None, path=res_pose_path)
    
    verts = torch.from_numpy(verts).float().cuda()
    faces = torch.from_numpy(mesh.faces).int().cuda()
    normals = torch.from_numpy(mesh.vertex_normals.copy()).float().cuda()
    
    if texture_path is not None and os.path.exists(texture_path):
        uvs = torch.from_numpy(mesh.visual.uv).float().cuda()
        texture = np.array(Image.open(texture_path))[::-1, ...]
        texture = torch.from_numpy(texture.copy()).float().unsqueeze(0).cuda() / 255.0
        texture_mip = dr.texture_construct_mip(texture)
    
    glctx = dr.RasterizeGLContext()
    for i, (mvs, mvps) in tqdm(enumerate(zip(view_mats, view_proj_mats))):
        mvs = mvs[None].cuda()
        mvps = mvps[None].cuda()
        v_pos_cam = torch.matmul(torch.nn.functional.pad(verts, pad=(0,1), mode='constant', value=1.0), torch.transpose(mvs, 1, 2))
        v_pos_clip = torch.matmul(torch.nn.functional.pad(verts, pad=(0,1), mode='constant', value=1.0), torch.transpose(mvps, 1, 2))
        rast, rast_db = dr.rasterize(glctx, v_pos_clip, faces, (height, width))  # [N_v, H, W, 4]
        inlier_mask = rast[..., 3] > 0
        # alpha = (inlier_mask[0].cpu().numpy() * 255.).astype('uint8')
        frg_normal, _ = dr.interpolate(normals, rast, faces)        # [N_v, H, W, 3]
        frg_normal = torch.nn.functional.normalize(frg_normal, p=2, dim=-1, eps=1e-8).contiguous()
        
        world_view_transform = torch.tensor(getWorld2View2(view_mats_colmap[i][:3, :3], view_mats_colmap[i][:3, 3])).transpose(0, 1).cuda()
        frg_normal = frg_normal @ world_view_transform[:3, :3]
        frg_normal = -frg_normal

        frg_depth, _ = dr.interpolate(v_pos_cam, rast, faces)           # shape: (N_v, H, W, 4)
        sav_depth = -frg_depth[0, ..., 2:3].cpu()

        torch.save(sav_depth, os.path.join(res_depth_path, "{:05}.pt".format(i)))
        torch.save(frg_normal[0].cpu(), os.path.join(res_normal_path, "{:05}.pt".format(i)))
        
        # cv2.imwrite(os.path.join(res_alpha_path, "{:05}.png".format(i)), alpha)
        
        vis_n = (frg_normal[0].cpu().numpy() * 0.5 + 0.5) * 255.
        cv2.imwrite(os.path.join(vis_normal_path, "{:05}.jpg".format(i)), vis_n.astype('uint8'))
        
        if texture_path is not None and os.path.exists(texture_path):
            frg_uv, _ = dr.interpolate(uvs, rast, faces)
            frg_image = dr.texture(texture, frg_uv, mip=texture_mip, filter_mode='linear')
            frg_image[~inlier_mask] = 0.0
            cv2.imwrite(os.path.join(res_rgb_path, "{:05}.jpg".format(i)), (frg_image[0] * 255.).cpu().numpy()[..., ::-1].astype('uint8'))
    