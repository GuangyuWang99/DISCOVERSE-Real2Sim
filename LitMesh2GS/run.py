import os

# ----------------------------------------------------------------------------------
## objaverse
# scene=1
# data_base_path='/data/guangyu/aRobotics/data/objaverse/{}'.format(scene)
# out_base_path='/data/guangyu/aGaussian/record/objaverse/{}/litmesh2gs'.format(scene)
# gpu_id=0
# ----------------------------------------------------------------------------------

# # # ----------------------------------------------------------------------------------
# # robot_objects
# scene="symmetrical_garden_02_1k_pack"
# data_base_path='/data/guangyu/aRobotics/data/relit/{}'.format(scene)
# out_base_path='/data/guangyu/aGaussian/record/relit/{}/litmesh2gs'.format(scene)
# gpu_id=6
# # # ----------------------------------------------------------------------------------

# # ----------------------------------------------------------------------------------
# # robot_objects
scene="9_50"
data_base_path='/data/guangyu/aRobotics/data/rodin_part1/{}'.format(scene)
out_base_path='/data/guangyu/aGaussian/record/rodin_part1/{}/litmesh2gs'.format(scene)
gpu_id=2
# # ----------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------
# # ## robot
# # scene="airbot_sundial/link1"
# # scene="mmk2_4/avg_link"
# # scene="dji/wheel3_4"
# # scene="mmk2_shanghai_bund/avg_link"
# # scene="airbot_flowerhouse/right"
# # scene="mmk2_act512/avg_link"
# scene="mmk2_act10/yaw"
# # scene="jujube"
# data_base_path='/data/guangyu/aRobotics/data/robot/{}'.format(scene)
# out_base_path='/data/guangyu/aGaussian/record/robot/{}/litmesh2gs'.format(scene)
# gpu_id=5
# ----------------------------------------------------------------------------------

common_args = f"--data_device cuda --densify_grad_threshold 0.0002 -r 1"
cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py -s {data_base_path} -m {out_base_path} {common_args}'
print(cmd)
os.system(cmd)

# common_args = f"--data_device cuda --densify_grad_threshold 0.0002 -r 1"
# cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py -s {data_base_path} -m {out_base_path} {common_args}'
# print(cmd)
# os.system(cmd)

# common_args = f"--data_device cuda --skip_train -r 1"
# cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python test.py -s {data_base_path} -m {out_base_path} {common_args}'
# print(cmd)
# os.system(cmd)