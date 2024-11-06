import os

# scene='GreatWall/TW1_Ego'
# # scene='MipNeRF360/Stump'
# data_base_path='/data/guangyu/dataset/aLit/{}/colmap'.format(scene)
# out_base_path='/data/guangyu/aGaussian/record/{}/ManifoldPGSR/all_few_dm'.format(scene)

# ----------------------------------------------------------------------------------
## rodin
# scene=2
# data_base_path='/data/guangyu/aRobotics/data/rodin/{}/buffers'.format(scene)
# out_base_path='/data/guangyu/aGaussian/record/rodin/{}/Mesh2GS'.format(scene)
# gpu_id=2
# ----------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------
## objaverse
scene=1
data_base_path='/data/guangyu/aRobotics/data/objaverse/{}'.format(scene)
out_base_path='/data/guangyu/aGaussian/record/objaverse/{}/mesh2gs_nod'.format(scene)
gpu_id=2
# ----------------------------------------------------------------------------------

# common_args = f"-r 1 --data_device cuda --densify_abs_grad_threshold 0.0002 --eval"
# cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python initialize.py -s {data_base_path} -m {out_base_path} {common_args}'
# print(cmd)
# os.system(cmd)

common_args = f"--data_device cuda --densify_grad_threshold 0.0002"
cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py -s {data_base_path} -m {out_base_path} {common_args}'
print(cmd)
os.system(cmd)

# common_args = f"--data_device cuda --eval --skip_train"
# cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python render.py -s {data_base_path} -m {out_base_path} {common_args}'
# print(cmd)
# os.system(cmd)

# common_args = f"--data_device cuda --eval --skip_train"
# cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python test.py -s {data_base_path} -m {out_base_path} {common_args}'
# print(cmd)
# os.system(cmd)