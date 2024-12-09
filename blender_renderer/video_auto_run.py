import os
import subprocess

# 主文件夹路径
main_folder_path = 'Arodin'
scripts_to_run = []


part_folders = ['part_1']
# part_folders = ['part_7', 'part_8', 'part_9', 'part_10']

# model_folders = []

glb_file_names = ["chair_1", "chair_2"]
in_root_path = "/media/womoer/Wdata/data/Arodin"
ot_root_path = "/media/womoer/Wdata/aRobotics/Relit/blender_output/aigc"
hdr_path = "/media/womoer/Wdata/aDLABSIM/code/BlenderGS/hdr_map/studio_small_08_1k.exr"



# 遍历 vedio_stuff 下的每个 part 文件夹
# for part_folder in sorted(os.listdir(main_folder_path)):
for glb_name in glb_file_names:
    input_file_path = os.path.join(in_root_path, f"{glb_name}.glb")

    # 构建输出路径，包含 part 文件夹和模型文件夹名称
    out_path = os.path.join(ot_root_path, glb_name)
    os.makedirs(out_path, exist_ok=True)

    vedio_name = f'vedio_{glb_name}.mp4'

    # 构建两个脚本的命令
    render_script = [
        'python', 'blender_renderer/vedio_render.py',
        '--object_path', input_file_path,
        '--out_path', out_path,
        '--vedio_name', vedio_name,
        '--vedio_fps', str(24),
        '--num_points', str(72),
        '--lit_path', hdr_path,

    ]
    # colmap_script = [
    #     'python', 'blender_renderer/tocolmap.py',
    #     '--out_path', out_path
    # ]

    # 将命令添加到 scripts_to_run 列表中
    scripts_to_run.append(render_script)
    # scripts_to_run.append(colmap_script)



# 打印检查生成的命令（可选）
for script in scripts_to_run:
    print("Command to run:", ' '.join(script))


# 依次执行每个脚本，等待完成后再继续执行下一个
for script in scripts_to_run:
    try:
        result = subprocess.run(script, check=True)
        print(f"Executed {script[1]} successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while executing {script[1]}: {e}")
        break  # 如果某个脚本出错，停止执行后续脚本