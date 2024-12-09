CUDA_VISIBLE_DEVICES=0 python inpaint.py --dataset /data/guangyu/aRobotics/data/scene --output_dir /data/guangyu/aRobotics/data/light
CUDA_VISIBLE_DEVICES=0 python ball2envmap.py --ball_dir /data/guangyu/aRobotics/data/light/square --envmap_dir /data/guangyu/aRobotics/data/light/envmap
CUDA_VISIBLE_DEVICES=0 python exposure2hdr.py --input_dir /data/guangyu/aRobotics/data/light/envmap --output_dir /data/guangyu/aRobotics/data/light/hdr
