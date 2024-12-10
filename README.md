# Mesh2GS Asset Generation for DISCOVERSE

## Installation
These tools are tested on Ubuntu 18.04.

To setup the Python environment for Mesh2GS & DiffusionLight, run:
```bash
conda create -n mesh2gs python=3.9
conda activate mesh2gs
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 #replace your cuda version
pip install -r requirements.txt
pip install LitMesh2GS/submodules/diff-plane-rasterization
pip install LitMesh2GS/submodules/simple-knn
```
Then, install [Blender](https://www.blender.org/) and [Blender Python API (bpy)](https://docs.blender.org/api/current/info_advanced_blender_as_bpy.html). We strongly recommend installing [bpy](https://pypi.org/project/bpy/) in the Python environment, but if you do have difficulty installing it, you can also run the related scripts in the `Scripting` panel of the Blender executable.

## DiffusionLight
*Estimate HDR environment map from a single RGB image.*

### Pretrained Weights for Huggingface Model
If you can not connect to [huggingface](https://huggingface.co/) due to VPN issues, please manually download the pretrained models from this [link](https://pan.baidu.com/s/1hVsfmpQav7DQQY3tdy9q-Q) (Code: 61i2). Then, manually modify the model paths (`SD_MODELS`, `VAE_MODELS`, `CONTROLNET_MODELS`, `DEPTH_ESTIMATOR`) in `DiffusionLight/relighting/argument.py` as the absolute path of your downloaded model folders.

### How to Run

**Firstly, prepare the input images.** Please resize the input image to **1024x1024**. To achieve this, we recommond cropping the images to contain as much background information as possible. As an alternative, we also recommend padding it with a black border. 

Organize all the processed images into a folder and specify the absolute path of the folder as `YourInputPath`. Specify `YourOutputPath` as a folder for saving your results. Then, run by:
```bash
python DiffusionLight/inpaint.py --dataset YourInputPath --output_dir YourOutputPath
python DiffusionLight/ball2envmap.py --ball_dir YourOutputPath/square --envmap_dir YourOutputPath/envmap
python DiffusionLight/exposure2hdr.py --input_dir YourOutputPath/envmap --output_dir YourOutputPath/hdr
```
And the final `.exr` results (saved in `YourOutputPath/hdr/`) will be used for the subsequent Blender PBR.

## Blender Renderer
*Render mesh into multi-view images for 3DGS optimization, by uniformly sampling cameras on a sphere and performing (Pre-) Physically-Based Rendering (PBR) based on Blender (bpy) with customized environment HDR map (distant lighting effects).*

Note that this is **NOT** the real PBR functionality, since it simply bakes the lighting to the SH appearance of the 3DGS to mimic the hue of the background scene. 

### Prepare `.exr` HDR Maps
Organize all the hdr maps for (Pre-)PBR into a single folder like:
```
YourHDRPath                          
├── hdr_name_0.exr
├── hdr_name_1.exr
├── hdr_name_2.exr
...
└── hdr_name_n.exr
```

### Render 3D Mesh Assets

#### For `.glb` Assets (e.g., Objaverse / Rodin Assets)
We strongly recommend using `.glb` 3D mesh assets similar to [objaverse](https://github.com/allenai/objaverse-xl). All of the `.glb` 3D assets to be converted should be put together into a single folder like:
```
YourInputPath                          
├── model_or_part_name_0.glb
├── model_or_part_name_1.glb
├── model_or_part_name_2.glb
...
└── model_or_part_name_n.glb
```

Then, render by:
```bash
python blender_renderer/glb_render.py --root_in_path YourInputPath --root_hdr_path YourHDRPath --root_out_path YourOutputPath
```
The results will be saved at `YourOutputPath`, in which each folder wil contain the RGB images, depth maps, camera parameters, `.obj` geometry for each 3D model. 

There are several other parameters to tune if the renderings are not satisfactory.
- `lit_strength`: strength of the environment lighting, a larger value leads to a brighter rendering.
- `lens`: focal length of the camera. If the object is too small in the rendering, i.e., too many pixels are wasting, try increasing this value. Otherwise, when only a fraction of the object is rendered, try decreasing it.
- `resolution`: rendering resolution, default value 512x512, a larger resolution leads to much slower rendering time.


#### For `.obj` Assets (e.g., Robot Models)
If you are dealing with `.obj` assets, e.g., robot models, each model will come with several texture and material maps, and the data should be organized into individual folders for each model, as the following:
```
YourInputPath                          
├── model_or_part_name_0
│   ├── obj_name_0.obj       
│   ├── mtl_name_0.mtl       
│   ├── tex_name_0.png       
│   └── ...                
├── model_or_part_name_1            
│   ├── obj_name_1.obj       
│   ├── mtl_name_1.mtl       
│   ├── tex_name_1.png       
│   └── ...                
├── model_or_part_name_2
...
└── model_or_part_name_n
```
The robot models developed by DISCOVER LAB, including MMK2, AirBot, DJI, RM2, etc., can be accessed through this [link](https://pan.baidu.com/s/1BW0GoDFmd0mPz9QItuJs7A) (Code: 94po)

Then, render by:
```bash
python blender_renderer/obj_render.py --root_in_path YourInputPath --root_hdr_path YourHDRPath --root_out_path YourOutputPath
```
The parameter arguments are the same as `blender_renderer/glb_render.py`.


### Convert Cameras to COLMAP Convention
Convert the camera parameters from Blender rendering to the colmap formats by:
```bash
python blender_renderer/models2colmap.py --root_path YourOutputPath
```
Make sure to set the intrinsics (i.e., `--resolution`, `--lens`, `--sensor_size`) **strictly the same** when running `obj_render.py` / `glb_render.py` and `models2colmap.py`.

## Mesh2GS
Run Mesh2GS for each 3D asset one-by-one:
```bash
python LitMesh2GS/train.py -s YourOutputPath/model_or_part_name_i -m YourOutputPath/model_or_part_name_i/mesh2gs --data_device cuda --densify_grad_threshold 0.0002 -r 1
```
The 3DGS results will be saved at a new folder `mesh2gs` in `YourOutputPath/model_or_part_name_i` for each 3D asset.