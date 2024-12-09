import ezexr
import os
import numpy as np
import cv2
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

if __name__ == "__main__":
    root_dir = "/data/guangyu/aRobotics/data/light/hdr"
    in_name = "flwhs.exr"
    ot_name = "flwhs_1.exr"
    light = cv2.imread(os.path.join(root_dir, in_name), cv2.IMREAD_UNCHANGED).astype(np.float32).transpose([1, 0, 2])
    h, w, _ = light.shape
    ezexr.imwrite(os.path.join(root_dir, ot_name), light[..., ::-1])
    vis_light = light.copy()
    vis_light = (vis_light - vis_light.min()) / (vis_light.max() - vis_light.min())
    cv2.imwrite(os.path.join(root_dir, ot_name.split('.')[0]+'.jpg'), (vis_light*255).astype('uint8'))