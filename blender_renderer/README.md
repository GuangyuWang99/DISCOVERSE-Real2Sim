# BlenderRenderer

Uniformly Sample cameras on a sphere and perform physics-based rendering based on blender (bpy) with customized environment HDR map (distant lighting). 

### Render Native 3D Assets
Set the configs in `create_argparser()` and render uniformly using blender by:
```bash
python render.py
```
Note that the rendering is done by firstly normalizing the 3D model into a unit sphere, i.e., the output 3D model (`1.obj`) is different from the input model (`.glb` or `.obj` or `.fbx`) with automatic centering and scaling. 

Then, convert the intrinsics and extrinsics from blender to the colmap formats by:
```bash
python tocolmap.py
```
Make sure to set the intrinsics (i.e., `--resolution`, `--lens`, `--sensor_size` in `create_argparser()`) **strictly the same** when running `render.py` and `tocolmap.py`.

### Render Specified 3D Assets
In cases where the output 3D model (`1.obj`) should strictly align with the original input model (`.glb` or `.obj` or `.fbx`), run by:
```bash
python bot_render.py
python bot2colmap.py
```
These scripts make only minor modifications compared to the previous ones, which use the centering and scaling parameters to compensate for extrinsics when exporting colmap formats.

