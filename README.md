# Aerial-NeRF
Rendering for arbitrary aerial trajectories and arbitrary aerial heights.
![image](https://github.com/Xiaohan-Z/Aerial-NeRF/blob/main/images/render_all_6.png)
Our environment is based on Python 3.7. Please configure the environment using the following command:
```
cd Aerial-NeRF
pip install -r requirements
```
We provide the SCUTer aerial dataset with uneven distribution of drones:   
Baidu Netdisk: https://pan.baidu.com/s/14VLd4QzTJL6k2bXOVMgIzA?pwd=4g7j 
Password：4g7j  
Training Aerial-NeRF use the following command:
```
cd Aerial-NeRF
python run_nerf.py --config configs/SCUT.txt 
```
 And set the `--ckpt_list` in `run_nerf.py` to `[]`.



