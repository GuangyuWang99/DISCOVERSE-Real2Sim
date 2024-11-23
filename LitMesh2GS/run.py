import os

# ----------------------------------------------------------------------------------
## objaverse
# scene=1
# data_base_path='/data/guangyu/aRobotics/data/objaverse/{}'.format(scene)
# out_base_path='/data/guangyu/aGaussian/record/objaverse/{}/litmesh2gs'.format(scene)
# gpu_id=0
# ----------------------------------------------------------------------------------

# # ----------------------------------------------------------------------------------
# ## robot_objects
# scene="rm2_1122_new"
# data_base_path='/data/guangyu/yupei/models/obj_model_hdr_8/{}'.format(scene)
# out_base_path='/data/guangyu/aGaussian/record/robot_objects/{}/litmesh2gs_di5k'.format(scene)
# gpu_id=0
# # ----------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------
## robot
# scene="airbot_sundial/link1"
scene="rm2/right_front_steer_sundial"
data_base_path='/data/guangyu/aRobotics/data/robot/{}'.format(scene)
out_base_path='/data/guangyu/aGaussian/record/robot/{}/litmesh2gs_di1k_df'.format(scene)
gpu_id=1
# ----------------------------------------------------------------------------------

common_args = f"--data_device cuda --densify_grad_threshold 0.0002 -r 1"
cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py -s {data_base_path} -m {out_base_path} {common_args}'
print(cmd)
os.system(cmd)

# common_args = f"--data_device cuda --skip_train -r 1"
# cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python test.py -s {data_base_path} -m {out_base_path} {common_args}'
# print(cmd)
# os.system(cmd)