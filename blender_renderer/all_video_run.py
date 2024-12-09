import os
import bpy
import datetime
import math
import random
import cv2
import numpy as np
import argparse
from typing import Tuple
from mathutils import Vector, Matrix, Quaternion


def get_3x4_RT_matrix_from_blender(cam: bpy.types.Object) -> Matrix:
    """Returns the 3x4 RT matrix from the given camera.

    Taken from Zero123, which in turn was taken from
    https://github.com/panmari/stanford-shapenet-renderer/blob/master/render_blender.py

    Args:
        cam (bpy.types.Object): The camera object.

    Returns:
        Matrix: The 3x4 RT matrix from the given camera.
    """
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]

    R_bcam2cv = Matrix(
        ((1, 0, 0),
         (0, -1, 0),
         (0, 0, -1))
    )

    R_world2bcam = rotation.to_matrix().transposed()

    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1 * R_world2bcam @ location

    R_world2bcam = R_bcam2cv @ R_world2bcam
    T_world2bcam = R_bcam2cv @ T_world2bcam

    # put into 3x4 matrix
    RT = Matrix(
        (
            R_world2bcam[0][:] + (T_world2bcam[0],),
            R_world2bcam[1][:] + (T_world2bcam[1],),
            R_world2bcam[2][:] + (T_world2bcam[2],),
        )
    )

    # rotate around x-axis by 90-degree
    rotx90 = Matrix(
        (
            (1, 0, 0, 0),
            (0, 0, -1, 0),
            (0, 1, 0, 0),
            (0, 0, 0, 1)
        )
    )
    RT = RT @ rotx90

    # # colmap to open-gl
    # matcv2gl = Matrix(
    #     (
    #         (1, 0, 0),
    #         (0, -1, 0),
    #         (0, 0, -1)
    #     )
    # )
    # RT = matcv2gl @ RT
    return RT


def sample_random_point_on_sphere(radius: float) -> Tuple[float, float, float]:
    theta = random.random() * 2 * math.pi
    phi = math.acos(2 * random.random() - 1)
    return (
        radius * math.sin(phi) * math.cos(theta),
        radius * math.sin(phi) * math.sin(theta),
        radius * math.cos(phi),
    )


def sample_ring_on_sphere(radius: float, height: float, num_points: int,
                          center: Tuple[float, float, float] = [0, 0, 0]) -> list:
    points = []

    # Calculate the radius of the circle at the given height
    ring_radius = math.sqrt(radius ** 2 - height ** 2)

    if ring_radius <= 0:
        raise ValueError("The height must be smaller than the radius of the sphere!")

    # Sample points on the ring in the horizontal plane
    for i in range(num_points):
        # Evenly distribute the points on the ring
        theta = (i / num_points) * 2 * math.pi  # Angle around the ring

        # Calculate the x, y coordinates of the point on the ring at the given height
        x = ring_radius * math.cos(theta)
        y = ring_radius * math.sin(theta)
        z = height  # Height is constant for all points

        # Adjust positions by adding the center offset
        points.append((
            center[0] + x,  # x position
            center[1] + y,  # y position
            center[2] + z  # z position
        ))

    return points


def sample_uniform_point_on_sphere(radius: float, num_horiz: int, num_verti: int):
    rets = []
    for j in range(num_verti):
        phi = ((j + 1) / num_verti) * math.pi
        for i in range(num_horiz):
            theta = ((i + 1) / num_horiz) * math.pi * 2
            rets.append((
                radius * math.sin(phi) * math.cos(theta),
                radius * math.sin(phi) * math.sin(theta),
                radius * math.cos(phi),
            ))
    return rets


def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


# load the glb model
def load_object(object_path: str) -> None:
    """Loads a glb model into the scene."""
    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    elif object_path.endswith(".obj"):
        bpy.ops.import_scene.obj(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")


def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def normalize_scene():
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")


def setup_camera(args):
    cam = scene.objects["Camera"]
    cam.location = (0, 1.2, 0)
    cam.data.lens = args.lens
    cam.data.sensor_width = args.sensor_size
    cam.data.sensor_height = args.sensor_size
    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"
    return cam, cam_constraint


def add_lighting_and_set_background_with_mix(lit_path, lit_strength=1.0) -> None:
    # 删除默认光源
    if "Light" in bpy.data.objects.keys():
        bpy.data.objects["Light"].select_set(True)
        bpy.ops.object.delete()

    # 获取并清空当前的 world 节点
    scene = bpy.context.scene
    world_node_tree = scene.world.node_tree
    world_node_tree.nodes.clear()  # 清除旧节点

    # 创建环境光照节点，加载 .exr 环境图像
    lit_obj = bpy.data.images.load(lit_path)
    environment_texture_node = world_node_tree.nodes.new(type="ShaderNodeTexEnvironment")
    environment_texture_node.image = lit_obj

    # 创建一个背景节点，设置为纯白色背景
    background_node = world_node_tree.nodes.new(type="ShaderNodeBackground")
    background_node.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)  # 设置为纯白色背景
    background_node.inputs["Strength"].default_value = 10.0  # 设置背景强度

    # 创建混合着色器
    mix_shader_node = world_node_tree.nodes.new(type="ShaderNodeMixShader")

    # 创建光程节点，用于控制射线
    light_path_node = world_node_tree.nodes.new(type="ShaderNodeLightPath")

    # 创建用于输出到场景的世界输出节点
    world_output_node = world_node_tree.nodes.new(type="ShaderNodeOutputWorld")

    # 创建环境光照背景节点，它会影响光照但不影响背景颜色
    environment_background_node = world_node_tree.nodes.new(type="ShaderNodeBackground")
    environment_background_node.inputs["Strength"].default_value = lit_strength  # 控制环境图光照强度

    # 将环境图连接到环境光照背景节点
    world_node_tree.links.new(environment_texture_node.outputs["Color"], environment_background_node.inputs["Color"])

    # 将环境光照节点与背景节点合并
    world_node_tree.links.new(environment_background_node.outputs["Background"],
                              mix_shader_node.inputs[1])  # 第二输入为环境图光照

    # 将纯白色背景连接到混合着色器
    world_node_tree.links.new(background_node.outputs["Background"], mix_shader_node.inputs[2])  # 第三输入为纯白色背景

    # 使用光程节点的 IsCameraRay 输出作为混合系数
    world_node_tree.links.new(light_path_node.outputs["Is Camera Ray"], mix_shader_node.inputs["Fac"])

    # 将混合着色器的输出连接到世界输出节点
    world_node_tree.links.new(mix_shader_node.outputs["Shader"], world_output_node.inputs["Surface"])

    # 确保渲染的背景不是透明的
    scene.render.film_transparent = False  # 禁用透明背景


def add_lighting(lit_path, lit_strength=1.0) -> None:
    # delete the default light
    if "Light" in bpy.data.objects.keys():
        bpy.data.objects["Light"].select_set(True)
        bpy.ops.object.delete()

    location_x = 0

    lit_obj = bpy.data.images.load(lit_path)

    environment_texture_node = world_node_tree.nodes.new(type="ShaderNodeTexEnvironment")
    environment_texture_node.image = lit_obj
    location_x += 300

    background_node = world_node_tree.nodes.new(type="ShaderNodeBackground")
    background_node.inputs["Strength"].default_value = lit_strength
    background_node.location.x = location_x
    location_x += 300

    world_output_node = world_node_tree.nodes.new(type="ShaderNodeOutputWorld")
    world_output_node.location.x = location_x

    from_node = environment_texture_node
    to_node = background_node
    world_node_tree.links.new(from_node.outputs["Color"], to_node.inputs["Color"])

    from_node = background_node
    to_node = world_output_node
    world_node_tree.links.new(from_node.outputs["Background"], to_node.inputs["Surface"])


def render_image(args) -> None:
    res_rgb_path = os.path.join(args.out_path, "images")
    # res_pos_path = os.path.join(args.out_path, "poses")
    res_depth_path = os.path.join(args.out_path, "depths")
    res_vedio_path = os.path.join(args.out_path, "vedio")
    os.makedirs(res_rgb_path, exist_ok=True)
    # os.makedirs(res_pos_path, exist_ok=True)
    os.makedirs(res_depth_path, exist_ok=True)
    os.makedirs(res_vedio_path, exist_ok=True)

    reset_scene()
    load_object(args.object_path)
    normalize_scene()

    # Create input render layer node
    render_layers = bpy.context.scene.node_tree.nodes.new('CompositorNodeRLayers')
    # Create depth output nodes
    depth_file_output = bpy.context.scene.node_tree.nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.label = 'Depth Output'
    depth_file_output.base_path = ''
    depth_file_output.file_slots[0].use_node_format = True
    depth_file_output.format.file_format = "OPEN_EXR"
    bpy.context.scene.node_tree.links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])

    add_lighting_and_set_background_with_mix(lit_path=args.lit_path, lit_strength=args.lit_strength)
    # add_lighting(lit_path=args.lit_path, lit_strength=args.lit_strength)
    cam, cam_constraint = setup_camera(args)
    # create an empty object to track
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty

    video_path = os.path.join(res_vedio_path, args.vedio_name)
    frame_width = args.resolution
    frame_height = args.resolution
    fps = args.vedio_fps

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, frameSize=(frame_width, frame_height))

    uniform_cam_points = sample_ring_on_sphere(radius=args.camera_dist, height=args.camera_height,
                                               num_points=args.num_points)
    for i in range(len(uniform_cam_points)):
        # set the camera position
        cam.location = uniform_cam_points[i]

        direction = -cam.location
        rot_quat = direction.to_track_quat("-Z", "Y")
        cam.rotation_euler = rot_quat.to_euler()

        # render the image
        render_path = os.path.join(res_rgb_path, f"{i:03d}")
        scene.render.filepath = render_path
        depth_file_output.file_slots[0].path = os.path.join(res_depth_path, f"{i:03d}")
        bpy.ops.render.render(write_still=True)

        # 读取渲染的图片并添加到视频
        image_path = render_path + '.png'
        print(image_path)
        img = cv2.imread(image_path)  # 读取渲染出的图片
        video_writer.write(img)  # 将图像写入视频

        # # save camera RT matrix
        # rt_matrix = get_3x4_RT_matrix_from_blender(cam)
        # rt_matrix_path = os.path.join(res_pos_path, f"{i:03d}.npy")
        # np.save(rt_matrix_path, rt_matrix)

    # 释放视频写入器
    video_writer.release()

    bpy.ops.export_scene.obj(filepath=os.path.join(args.out_path, "1.obj"))


def create_argparser():
    parser = argparse.ArgumentParser()
    # I/O
    parser.add_argument("--lit_path", type=str, default="/data/guangyu/yupei/BlenderGS/hdr_map/6.exr",
                        help='directory that contains the hdr environment map')
    parser.add_argument("--object_path", type=str,
                        # default="/media/womoer/Wdata/aRobotics/robot_model/mmk2/avg_link/avg_link.obj",
                        default="/home/appp/3D_Rec/BlenderGS/glbs/000-075/9669365a5dfd43bfaad90b71f302fedd.glb",
                        help='directory that contains the model, in formats of .obj or .glb or .fbx')
    parser.add_argument("--out_path", type=str, default="/home/appp/3D_Rec/BlenderGS/blender_output/1",
                        help='directory that saves the results')

    # environment HDR map
    parser.add_argument("--lit_strength", type=float, default=5.0, help="Strength of the environment lighting")
    # camera sampling
    parser.add_argument("--camera_dist", type=float, default=1.5, help="radius of the sphere to locate cameras")
    parser.add_argument("--num_points", type=int, default=30, help="The number of points to sample on the ring.")
    parser.add_argument("--camera_height", type=float, default=1.0,
                        help="The height offset from the center (the vertical distance to the circle)")
    parser.add_argument("--num_horiz", type=int, default=10,
                        help="num of cameras uniformly spanning along the xy-plane")
    parser.add_argument("--num_verti", type=int, default=10, help="num of cameras uniformly spanning along the z-plane")
    # intrinsics
    parser.add_argument("--resolution", type=int, default=256, help="resolution of the rendering")
    parser.add_argument("--lens", type=int, default=40, help="focal length in mm")
    parser.add_argument("--sensor_size", type=int, default=32, help="focal length in mm")
    parser.add_argument("--vedio_fps", type=int, default=20, help="vedio fps")
    parser.add_argument("--vedio_name", type=str, default='vedio_output.mp4', help="vedio name")

    return parser


if __name__ == "__main__":
    args = create_argparser().parse_args()

    context = bpy.context
    scene = context.scene
    render = scene.render

    render.engine = "CYCLES"
    render.image_settings.file_format = "PNG"
    render.image_settings.color_mode = "RGB"
    render.resolution_x = args.resolution
    render.resolution_y = args.resolution
    render.resolution_percentage = 100

    scene.use_nodes = True
    view_layer = scene.view_layers[0]
    view_layer.use_pass_z = True
    view_layer.use_pass_normal = True
    view_layer.use_pass_diffuse_color = True
    view_layer.use_pass_object_index = True
    scene.cycles.device = "GPU"
    scene.cycles.samples = 32
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 1
    scene.cycles.transparent_max_bounces = 3
    scene.cycles.transmission_bounces = 3
    scene.cycles.filter_width = 0.01
    scene.cycles.use_denoising = True
    scene.render.film_transparent = False

    world_node_tree = bpy.context.scene.world.node_tree
    world_node_tree.nodes.clear()
    render_image(args)
    # exit(0)