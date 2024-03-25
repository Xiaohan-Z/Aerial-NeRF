import argparse
import json
from pathlib import Path
from tqdm import tqdm
import os
import cv2
import random
from datetime import datetime
from PIL import Image
from PIL.ExifTags import TAGS

import numpy as np
from colmap_parsing_utils import (
    parse_colmap_camera_params,
    qvec2rotmat,
    read_cameras_binary,
    read_images_binary,
)

def list_files(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            full_path = os.path.join(root, file)
            path_parts = full_path.split(os.sep)
            if len(path_parts) > 2:
                short_path = os.sep.join(path_parts[-2:])
                file_paths.append(short_path)
            else:
                file_paths.append(full_path)
    return file_paths

def find_difference(list1, list2):
    return [item for item in list1 if item not in list2]

def parse_args():
    parser = argparse.ArgumentParser(description="convert colmap to transforms.json")

    parser.add_argument("--recon_dir", type=str, default="/home/hanxiao/xiaohan/data/AN_data_own/own_data/own_enu")
    parser.add_argument("--output_dir", type=str, default="/home/hanxiao/xiaohan/data/AN_data_own")

    args = parser.parse_args()
    return args

# obtain data from EXIF
def get_image_capture_date(image_path):
    try:
        image = Image.open(image_path)

        # read exif data
        exif_data = image._getexif()

        # find camera data
        for tag, value in exif_data.items():
            tag_name = TAGS.get(tag, tag)
            if tag_name == 'DateTimeOriginal':
                capture_date = value
                return capture_date

    except (AttributeError, KeyError, IndexError):
        print("can't obtain data!")

    return None


def colmap_to_json(recon_dir, output_dir):
    """Converts COLMAP's cameras.bin and images.bin to a JSON file.

    Args:
        recon_dir: Path to the reconstruction directory, e.g. "sparse/0"
        output_dir: Path to the output directory.
        camera_model: Camera model used.
        camera_mask_path: Path to the camera mask.
        image_id_to_depth_path: When including sfm-based depth, embed these depth file paths in the exported json
        image_rename_map: Use these image names instead of the names embedded in the COLMAP db

    Returns:
        The number of registered images.
    """

    # TODO(1480) use pycolmap
    # recon = pycolmap.Reconstruction(recon_dir)
    # cam_id_to_camera = recon.cameras
    # im_id_to_image = recon.images
    cam_id_to_camera = read_cameras_binary(recon_dir / "cameras.bin")
    im_id_to_image = read_images_binary(recon_dir / "images.bin")

    # im_id_to_image[2582][1] 一个像素的四元数

    scene_scale= 0.1


    # 打乱键的顺序
    # keys = list(im_id_to_image.keys())
    # random.shuffle(keys)

    # 创建一个新的字典，键的顺序被打乱
    # shuffled_dict = {key: im_id_to_image[key] for key in keys}

    # print(shuffled_dict)

    frames = []
    names=[]

    for im_id, im_data in im_id_to_image.items():
        # NB: COLMAP uses Eigen / scalar-first quaternions
        # * https://colmap.github.io/format.html
        # * https://github.com/colmap/colmap/blob/bf3e19140f491c3042bfd85b7192ef7d249808ec/src/base/pose.cc#L75
        # the `rotation_matrix()` handles that format for us.

        # TODO(1480) BEGIN use pycolmap API
        # rotation = im_data.rotation_matrix()
        rotation = qvec2rotmat(im_data.qvec)
        translation = im_data.tvec.reshape(3, 1)

        # translation=translation*scene_scale

        # scale_matrix=np.array([[scene_scale,0,0],[0,scene_scale,0],[0,0,scene_scale]])
        # rotation=np.matmul(rotation, scale_matrix)
        # translation=np.matmul(scale_matrix,translation)

        w2c = np.concatenate([rotation, translation], 1)
        w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)

        trans_matrix=np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
        w2c_openGL=np.matmul(trans_matrix, w2c)

        # scale_matrix=np.array([[scene_scale,0,0,0],[0,scene_scale,0,0],[0,0,scene_scale,0],[0,0,0,1]])
        # w2c_openGL=np.matmul(scale_matrix, w2c_openGL)


        c2w = np.linalg.inv(w2c_openGL)
        
        c2w[2,3]=c2w[2,3]+120

        c2w[:3,3]=c2w[:3,3]*0.1

        # Convert from COLMAP's camera coordinate system (OpenCV) to ours (OpenGL)
        # 旋转矩阵 y 和 z轴调换朝向

        # c2w[0:3, 1:3] *= -1
        # R_c2w=c2w[0:3,0:3]
        # T_c2w= -np.matmul(R_c2w, translation)
        # c2w[0:3,3]=T_c2w.reshape(1, 3)

        # # x 和 y 轴调换
        # c2w = c2w[np.array([1, 0, 2, 3]), :]
        # # z 轴方向变换
        # c2w[2, :] *= -1

        name = im_data.name
        
        names.append(name)

        name = Path(f"./images/{name}")

        h_meter=c2w[2,3]

        imgfolder="/home/hanxiao/xiaohan/data/AN_data_own"
        image_path=os.path.join(imgfolder, name)

        # obtain camera data
        capture_date = get_image_capture_date(image_path)

        # if capture_date:
        #     print("camera data:", capture_date)
        #     # print(type(capture_date))
        # else:
        #     print("can't obtain data!")

        # trans data to date_object
        date_object = datetime.strptime(capture_date, "%Y:%m:%d %H:%M:%S")

        # trans data to unix
        timestamp = int(date_object.timestamp())

        # print("unix data:", timestamp)


        frame = {
            "file_path": name.as_posix(),
            "transform_matrix": c2w.tolist(),
            "colmap_im_id": im_id,
            "h_meter": h_meter,
            "time": timestamp
        }

        frames.append(frame)

    # 冒泡算法将高度从高到低排列

    n = len(frames)
    
    # # 使用方法
    # directory = '/home/hanxiao/xiaohan/data/colmap_down/images'
    # all_files = list_files(directory)
    
    # difference = find_difference(all_files, names)
    # print(difference)

    # 外层循环控制每一轮遍历
    for i in range(n - 1):
        # 内层循环控制每一轮中相邻元素的比较和交换
        for j in range(0, n - i - 1):
            # 如果相邻的元素逆序，则交换它们
            if frames[j]["h_meter"] > frames[j + 1]["h_meter"]:
                frames[j], frames[j + 1] = frames[j + 1], frames[j]

    # # 将图像按顺序输出
    # print("import sorted image")
    # img_idx=[i for i in range(n)]
    # for i in tqdm(img_idx):
    #     imgfolder="./data/residence"
    #     img_path=os.path.join(imgfolder, frames[i]["file_path"])
    #     image = cv2.imread(img_path)
    #     if not os.path.exists("img_sorted"):
    #         os.mkdir("img_sorted")
    #     cv2.imwrite(f'img_sorted/{i}.jpg', image)

    if set(cam_id_to_camera.keys()) != {1}:
        raise RuntimeError("Only single camera shared for all images is supported.")
    
    out = parse_colmap_camera_params(cam_id_to_camera[1])
    # out_test = parse_colmap_camera_params(cam_id_to_camera[1])


    # frames_train=[f for i, f in enumerate(frames)]
    # frames_test = [f for i, f in enumerate(frames) if i % holdout == 0]

    out["frames"] = frames
    # out_test["frames"] = frames_test

    with open(output_dir / "transforms_own_t_ENU.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4)

    # with open(output_dir / "transforms_test.json", "w", encoding="utf-8") as f:
    #     json.dump(out_test, f, indent=4)

    return len(frames)


if __name__ == "__main__":
    init_args = parse_args()
    Recondir = Path(init_args.recon_dir)
    Outputdir = Path(init_args.output_dir)
    colmap_to_json(Recondir, Outputdir)