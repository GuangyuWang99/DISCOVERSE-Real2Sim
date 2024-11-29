import os
import subprocess

# 主文件夹路径
main_folder_path = 'vedio_show_model'
scripts_to_run = []


part_folders = ['part_1']
# part_folders = ['part_7', 'part_8', 'part_9', 'part_10']

# model_folders = []


# 遍历 vedio_stuff 下的每个 part 文件夹
# for part_folder in sorted(os.listdir(main_folder_path)):
for part_folder in part_folders:
    part_folder_path = os.path.join(main_folder_path, part_folder)
    
    if os.path.isdir(part_folder_path):
        
        for model_folder in sorted(os.listdir(part_folder_path)):
        # for model_folder in model_folders
            model_folder_path = os.path.join(part_folder_path, model_folder)

            if os.path.isdir(model_folder_path):
                
                # 构建 .glb 和 .obj 文件路径
                glb_file_path = os.path.join(model_folder_path, f"{model_folder}.glb")
                obj_file_path = os.path.join(model_folder_path, f"{model_folder}.obj")
                
                # 检查文件是否存在，并优先使用 .glb 文件
                if os.path.exists(glb_file_path):
                    input_file_path = glb_file_path
                elif os.path.exists(obj_file_path):
                    input_file_path = obj_file_path
                else:
                    # 如果两种文件都不存在，则跳过
                    print(f"Neither .glb nor .obj found for model: {model_folder} in {part_folder}")
                    continue
                
                # 构建输出路径，包含 part 文件夹和模型文件夹名称
                out_path = f"blender_output/vedio_show_model_hdr_studio/{main_folder_path}/{part_folder}/{model_folder}"
                hdr_path = "hdr_map/studio_small_08_1k.exr"
                vedio_name = f'vedio_{model_folder}.mp4'
                
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