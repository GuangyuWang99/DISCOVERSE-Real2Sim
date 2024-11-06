# Mesh2GS: Robust, Photorealistic 3DGS Empowered by Manifold Priors

## Key Features
This implementation is based on [PGSR](https://github.com/zju3dv/PGSR), with the following modifications to better benefit from manifold priors:

### Manifold Initialization
The 3DGSs are strictly initialized as flattened, planar disks on mesh surface. The disks are located at the center of mesh facets, with their radius being the mean distance to the corresponding vertices and with their normals aligning with the face normals. See `create_from_manifold()` in `scene/gaussian_model.py` for details.

This actually enables ultra-fast convergence for simple scenes, e.g., only the first few thousand iterations taking 1~2 minutes.

### Depth and Normal Regularization
The planar gaussians are regularized with the multi-view normals and depths rendered from the mesh throughout the optimization, see `train.py` for details. The regularizations here significantly improve the geometry of 3DGS, making individual gaussians better align with the surface and represent richer details. Note that the multi-view terms in PGSR are deprecated since they offer little help here and inefficient.

### Mask Regularization
For Mesh2GS pipeline, we always want to reconstruct only the very clean foreground objects indicated by the mesh. To achieve this, multi-view masks are firstly generated as by-products during the rasterization of g-buffers (multi-view depths and normals), i.e., `mask = depth > 0`. 

The masks are applied in two ways to eliminate floaters and make the reconstructed scene with clear boundaries. 
- make all loss terms to only focus on the foreground regions;
- a binary-cross-entropy loss to explicitly enhance the boundaries;

See `train.py` for more details.

## How to Run
### Data Preprocess
Set the configs manually in the script and run by:
```bash
python sample_cams.py
```
This script uniformly samples cameras on a hemisphere around the mesh, and generate the multi-view g-buffers.

### Training
Set the configs in `scripts/run_mip360.py` and enables only the following lines:
```bash
common_args = f"--data_device cuda --densify_grad_threshold 0.0002"
cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py -s {data_base_path} -m {out_base_path} {common_args}'
print(cmd)
os.system(cmd)
```
Then run by:
```bash
python scripts/run_mip360.py
```

### Visualize Initializations
Set the configs in `scripts/run_mip360.py` and enables only the following lines:
```bash
common_args = f"-r 1 --data_device cuda --densify_abs_grad_threshold 0.0002 --eval"
cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python initialize.py -s {data_base_path} -m {out_base_path} {common_args}'
print(cmd)
os.system(cmd)
```
Then run by:
```bash
python scripts/run_mip360.py
```

## Updates
- [2024.11.18]: We ...