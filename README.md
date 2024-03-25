# Aerial-NeRF
Rendering for arbitrary aerial trajectories and arbitrary aerial heights.
![image](https://github.com/Xiaohan-Z/Aerial-NeRF/blob/main/images/render_all_6.png)
Our environment is based on Python 3.7. Please configure the environment using the following command:
```
cd Aerial-NeRF
pip install -r requirements
```
## Train and test on the SCUTer dataset  
We provide the SCUTer aerial dataset with uneven distribution of drones:   
Baidu Netdisk: https://pan.baidu.com/s/14VLd4QzTJL6k2bXOVMgIzA?pwd=4g7j 
Password：4g7j  
Training Aerial-NeRF use the following command:
```
cd Aerial-NeRF
python run_nerf.py --config configs/SCUT.txt 
```
And set the `--ckpt_list` in `run_nerf.py` to `[]`.  
We also provide the network parameters in `Nerial-NeRF/logs`.  
Test the rendering results on a region:
```
cd Aerial-NeRF
python run_nerf.py --config configs/SCUT.txt --render_test
```
Set `cluster_cur` in `Aerial-NeRF/configs` to the region you want to render. 
Set `--ckpt_list` to the corresponding region's parameter.  
Test the rendering results on the whole scene:
```
cd Aerial-NeRF
python run_nerf.py --config configs/SCUT.txt --render_test --test_all
```
Set `--ckpt_list` to all regions' parameters.  
## Train and test on your own dataset  
Through testing, it was found that the best rendering results can be achieved when the drone is at the same height and the tilt angle is 45 degrees.  
step 1: Convert the results of COLMAP to the ENU coordinate system by using `model_aligner` in COLMAP.  
step 2: Generate a JSON file for the camera pose. Replace the `--recon_die` and `--output_dir` in `ENU2nerf_SCUT.py` with your own paths, and modify `134 line` `120` to the altitude corresponding to the camera with a z-value of `0` in the ENU coordinate system of your dataset. Then execute:
```
cd Aerial-NeRF
python ENU2nerf_SCUT.py
```
step 3: Generate the pair.txt. "Replace the path in `colmap_input` with your own path, then execute:
```
cd Aerial-NeRF
python colmap_input.py
```
step 4: Modify `configs/SCUT.txt`. `camera_bin` represents the output file of COLMAP that has not been processed by `model_aligner`. Other parameters are the same as in nerf-pytorch: https://github.com/yenchenlin/nerf-pytorch
The training and testing are the same as train and test on the SCUTer dataset.  




