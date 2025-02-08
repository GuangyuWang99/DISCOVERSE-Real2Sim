<h1 align="center">Real2Sim Asset Generation Tools for DISCOVERSE</h1>
<p align="center"><a href="https://drive.google.com/file/d/1637XPqWMajfC_ZqKfCGxDxzRMrsJQA1g/view?usp=sharing"><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
<a href='https://air-discoverse.github.io/'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'></a>
</p>
<p align="center"><img src="assets/real2sim.jpg" width="100%"></p>

DISCOVERSE unifies real-world captures, 3D AIGC, and any existing 3D assets in formats of 3DGS (.ply), mesh
(.obj/.stl), and MJCF physical models (.xml), enabling their use as interactive scene nodes (objects and robots) or the background node (scene). We use [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) as a universal visual representation and integrate laser scanning, state-of-the-art generative
models, and physically-based relighting to boost the geometry and appearance fidelity of the reconstructed radiance fields.

## Installation
This repo is tested with Ubuntu 18.04+.

To setup the Python environment for Mesh2GS & DiffusionLight, run:
```bash
conda create -n mesh2gs python=3.9
conda activate mesh2gs
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # replace your cuda version
pip install LitMesh2GS/submodules/diff-gaussian-rasterization
pip install LitMesh2GS/submodules/simple-knn
```
Please manually install other dependencies described in `requirements.txt`. 

Also, install [Blender](https://www.blender.org/)(recommonded version: 3.4) and [Blender Python API (bpy)](https://docs.blender.org/api/current/info_advanced_blender_as_bpy.html). We strongly recommend installing [bpy](https://pypi.org/project/bpy/) in the Python environment. However, if you do have difficulty installing it, you can also run the related scripts in the `Scripting` panel of the Blender executable.

## Image-to-3D Generation with [TRELLIS](https://github.com/microsoft/TRELLIS)
*Generate object-level, high-quality textured mesh from a single RGB image.*

[TRELLIS](https://github.com/microsoft/TRELLIS) is the latest, open-source, state-of-the-art 3D generative model that generates high-quality textured meshes, 3DGSs, or radiance fields. We recommond to set up a new environment for TRELLIS and run image-to-3D generation following the [official guidelines](https://github.com/microsoft/TRELLIS). We recommond generating textured meshes as `.glb` files to be compatible with the subsequent lighting estimation, blender relighting, and Mesh2GS steps. **Note that, for a quick setup, you can also directly generate 3DGS (`.ply`) assets for DISCOVERSE skipping the following steps.**

For 3D generation with higher quality, we recommond using commercial software like [Deemos Rodin](https://hyper3d.ai/) ([CLAY](https://arxiv.org/abs/2406.13897)).

## 3D Scene Reconstruction
We recommond using [LixelKity K1 scanner](https://www.xgrids.cn/lixelk1) and [Lixel CyberColor](https://www.xgrids.cn/lcc) for generating high-quality 3DGS field to serve as the background node. Without access to the scanner, you can use the [vanilla 3DGS](https://github.com/graphdeco-inria/gaussian-splatting) for scene reconstruction.

## Lighting Estimation with [DiffusionLight](https://github.com/DiffusionLight/DiffusionLight)
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

## Physically-Based Relighting with Blender
*Render mesh into multi-view images for 3DGS optimization, by uniformly sampling cameras on a sphere and performing (Pre-) physically-based relighting with Blender (bpy) with customized environment HDR map (distant lighting effects).*

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
The results will be saved at `YourOutputPath`, in which each folder (namely {hdr_name_i}_{model_or_part_name_i}) will store the rendered RGB images, depth maps, camera parameters, `.obj` geometry for one of the 3D models under one of the lightings. 

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
The robot models developed by DISCOVER LAB, including MMK2, AirBot, DJI, RM2, etc., can be accessed through this [link](https://pan.baidu.com/s/1BW0GoDFmd0mPz9QItuJs7A) (Code: 94po).

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
*Convert textured meshes to 3DGSs.*

Run Mesh2GS for each 3D asset one-by-one:
```bash
python LitMesh2GS/train.py -s YourOutputPath/model_or_part_name_i -m YourOutputPath/model_or_part_name_i/mesh2gs --data_device cuda --densify_grad_threshold 0.0002 -r 1
```
The 3DGS results will be saved at a new folder `mesh2gs` in `YourOutputPath/model_or_part_name_i` for each 3D asset.

Since 3DGS is memory-efficient by nature, we recommond to specify `--densification_interval` to roughly control the amounts of the resulting 3DGS points. A larger value will lead to a sparser 3DGS field.