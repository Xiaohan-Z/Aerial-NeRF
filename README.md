# Aerial-NeRF
Rendering for arbitrary aerial trajectories and arbitrary aerial heights.
![image](https://github.com/Xiaohan-Z/Aerial-NeRF/blob/main/images/render_all_6.png)
Our environment is based on Python 3.7. Please configure the environment using the following command:
```
cd Aerial-NeRF
pip install -r requirements
```
We provide the SCUTer aerial dataset with uneven distribution of drones: 
We provide the network parameters in Aerial-NeRF/logs. Use the following code to reproduce the results in the paper:
```
cd Aerial-NeRF
python run_nerf.py --config configs/SCUT.txt
```

