import numpy as np
import os
import json
import cv2
import imageio
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import struct

def _load_google_data_train(basedir,factor):

    img_folder = 'images'

    with open(os.path.join(basedir, "transforms_own_t_ENU.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    # # pfx = ".jpeg"
    pfx = ".JPG"
    # # imgfolder=basedir
    imgfolder = os.path.join(basedir, img_folder)
    fnames = [
        frame["file_path"].split("/")[-2] + "/" + frame["file_path"].split("/")[-1].split(".")[0] + pfx
        for frame in meta["frames"]
    ]

    sh = np.array(cv2.imread(os.path.join(imgfolder, fnames[0])).shape)

    whole_idx = range(0, len(fnames))
    list_whole_idx = list(whole_idx) 

    fnames=[fnames[i] for i in list_whole_idx]

    poses_json = np.stack([np.array(frame["transform_matrix"]) for i, frame in enumerate(meta["frames"])])

    whole_idx = range(0, len(poses_json))
    list_whole_idx = list(whole_idx) 
    # find the indix is not divided by holdout 
    poses_json=[poses_json[i] for i in list_whole_idx]
    # transform to numpy
    poses_json=np.stack(poses_json,axis=0)

    poses=np.ones((len(poses_json),3, 5))

    # translaton and rotarion matrix
    poses[:,:,:3]=poses_json[:,:3,:3]
    poses[:,:,3]=poses_json[:,:3,-1]

    # size and focus of picture
    poses[:, :2, 4] = np.array(sh[:2]//factor).reshape([1, 2])
    poses[:, 2, 4] = meta["fl_x"] * 1./factor 

    return poses,fnames

def data_distribution(basedir,pair,camera_bin,cluster_num,factor):

    poses,fnames = _load_google_data_train(basedir,factor=factor)

    # print('Loaded train image data shape:', imgs_train.shape, ' hwf:', poses_train[0,:,-1])
    # poses_x and poses_y
    # [N_pic]

    x=poses[:,0,3]
    y=poses[:,1,3]

    # point map
    plt.scatter(x, y)

    plt.xlabel('X')
    plt.ylabel('Y')

    plt.savefig('distribution_plot.png')

    plt.close()

    # [N_pic,2]
    data = np.column_stack((x, y))

    # cerate K-Means model
    kmeans = KMeans(n_clusters=cluster_num, random_state=0)

    # fit data
    kmeans.fit(data)

    # obtain labels for each data
    labels = kmeans.labels_

    # the center of cluster
    cluster_centers = kmeans.cluster_centers_

    # point map，color according to cluster
    for i in range(cluster_num):
        plt.scatter(data[labels == i][:, 0], data[labels == i][:, 1], label=f'Cluster {i + 1}')

    # plot center
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='black', marker='x', s=100, label='Cluster Centers')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('K-Means')

    plt.legend()

    plt.savefig('k-means.png')

    plt.close()

    # initialize list to save cross points and labels 
    cross_points = []
    cross_labels=[]
    cross_index=[]
    cross_neibor_labels=[]


    # for each data
    for i in range(len(data)):
        point = data[i]
        label = labels[i]
        
        # for current point, if it has neiborhood from other cluster
        is_cross_point = False
        for j in range(cluster_num):
            if j != label:
                # check if any points in other clusters near this point
                nearby_points = data[labels == j]
                if np.any(np.linalg.norm(nearby_points - point, axis=1) < 1.5):  # threshold
                    is_cross_point = True
                    break
        
        if is_cross_point:
            # obtain corss poinnts
            cross_points.append(point)
            # obtain cross labels
            cross_labels.append(label)
            # obtain cross index
            cross_index.append(i)
            # obtain nerbor cluster index
            cross_neibor_labels.append(j)

    # [N_cross,2]
    cross_points = np.array(cross_points)
    # [N_cross]
    cross_labels = np.array(cross_labels)
    # [N_cross]
    cross_index= np.array(cross_index)
    # [N_cross]
    cross_neibor_labels= np.array(cross_neibor_labels)

    # calculate points in each lable
    # cross_lable0_data=cross_points[cross_labels==0]

    # point map，color according to cluster
    for i in range(cluster_num):
        plt.scatter(data[labels == i][:, 0], data[labels == i][:, 1], label=f'Cluster {i + 1}')

    # plot cross points
    plt.scatter(cross_points[:, 0], cross_points[:, 1], c='red', marker='x', s=100, label='Cross Points')

    # plot cluster center
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='black', marker='x', s=100, label='Cluster Centers')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('K-Means')

    plt.legend()

    plt.savefig('k_means_cross.png')

    plt.close()

    # obtain the pose matrix of each cross point
    cross_c2w=poses[cross_index]
    # obtain the rotation matrix [N_cross,3,3]
    cross_rot=cross_c2w[:,:,:3]
    # calculate the direction of cross points

    # z axis of camera
    reference_vector = np.array([0, 0, 1])

    # transfer z axis to world coordinate [N_cross,3]
    camera_direction = np.dot(cross_rot, reference_vector)

    # plot cross points direction
    x=cross_points[:,0]
    y=cross_points[:,1]

    u = camera_direction[:,0]
    v = camera_direction[:,1]

    # create a fig
    plt.figure()

    # point map，color according to cluster
    for i in range(cluster_num):
        plt.scatter(cross_points[cross_labels == i][:, 0], cross_points[cross_labels == i][:, 1], label=f'Cluster {i + 1}')

    # plot direction of cross data
    plt.quiver(x, y, u, v, color='r', angles='xy', scale_units='xy', scale=1, label='Directions')

    # plot cluster center
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='black', marker='x', s=100, label='Cluster Centers')

    plt.xlabel('X')
    plt.ylabel('Y')

    plt.title('direction')

    plt.legend()

    plt.savefig('cross_dir.png')

    plt.close()

    '''
    these two fuction obtain the pic name in pair.txt
    '''

    def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
        """Read and unpack the next bytes from a binary file.
        :param fid:
        :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
        :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
        :param endian_character: Any of {@, =, <, >, !}
        :return: Tuple of read and unpacked values.
        """
        data = fid.read(num_bytes)
        return struct.unpack(endian_character + format_char_sequence, data)

    def read_images_binary(path_to_model_file):
        """
        see: src/base/reconstruction.cc
            void Reconstruction::ReadImagesBinary(const std::string& path)
            void Reconstruction::WriteImagesBinary(const std::string& path)
        """
        images = {}
        with open(path_to_model_file, "rb") as fid:
            num_reg_images = read_next_bytes(fid, 8, "Q")[0]
            # print("img num")
            # print(num_reg_images)
            for image_index in range(num_reg_images):
                binary_image_properties = read_next_bytes(
                    fid, num_bytes=64, format_char_sequence="idddddddi")
                image_id = binary_image_properties[0]
                # print("image_id")
                # print(image_id)
                qvec = np.array(binary_image_properties[1:5])
                tvec = np.array(binary_image_properties[5:8])
                camera_id = binary_image_properties[8]
                # print("image_id_camera")
                # print(camera_id)
                image_name = ""
                current_char = read_next_bytes(fid, 1, "c")[0]
                while current_char != b"\x00":   # look for the ASCII 0 entry
                    image_name += current_char.decode("utf-8")
                    current_char = read_next_bytes(fid, 1, "c")[0]
                num_points2D = read_next_bytes(fid, num_bytes=8,
                                            format_char_sequence="Q")[0]
                x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                        format_char_sequence="ddq"*num_points2D)
                xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                    tuple(map(float, x_y_id_s[1::3]))])
                point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
                # print("point_3d")
                # print(point3D_ids)
                images[image_id-1]=image_name
        return images

    def read_pair_file(filename: str):
        """Read image pairs from text file and output a list of tuples each containing the reference image ID and a list of
        source image IDs

        Args:
            filename: pair text file path string

        Returns:
            List of tuples with reference ID and list of source IDs
        """
        data = []
        with open(filename) as f:
            num_viewpoint = int(f.readline())
            for _ in range(num_viewpoint):
                ref_view = int(f.readline().rstrip())
                src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                if len(src_views) != 0:
                    data.append((ref_view, src_views))
        return data

    # for each point find its 10 pair
    pair_data=read_pair_file(pair)

    # obtain corresponding name
    pair_name=read_images_binary(camera_bin)

    # test
    # pair_0=pair_data[0][0]
    # pair_0_name=pair_name[pair_0]
    # pairs=[]
    # for i in range(len(pair_data[0][1])):
    #     pairs.append(pair_name[pair_data[0][1][i]])
    # print(111)

    # class three cross_points
    cross_label={}
    # the index of each cluster points
    indices_ori={}
    # name of each cluster points
    cluster_ori={}
    # cross points name in each cluster
    cross={}
    # delete the cross_points in original clusters
    cluster_no_cross={}
    # add new cross_points to cluster
    cluster_new={}


    for i in range(cluster_num):
        cross_label[i]=[]

    # re-assign cross_point cluster
    for i in range(cross_points.shape[0]):
        # for each cross_point 
        cross_cur=cross_points[i]
        # cross_point label and neibor label
        cross_cur_label=cross_labels[i]
        cross_cur_neibor_label= cross_neibor_labels[i]
        # find two cluster center
        cross_cur_label_center=cluster_centers[cross_cur_label,:]
        cross_cur_label_neibor_center=cluster_centers[cross_cur_neibor_label,:]
        # calculate the angle of view direction and the line of camera and center
        cac= cross_cur_label_center-cross_cur
        cac_neighbor=cross_cur_label_neibor_center-cross_cur

        # calculate inner product
        # get xy-plane of camera_direction
        cam_xy_dir=camera_direction[i][:2]

        '''
        angle of cross_point and its label center
        '''
        dot_product = np.dot(cac,cam_xy_dir)

        # calculate length of vector
        magnitude1 = np.linalg.norm(cac)
        magnitude2 = np.linalg.norm(cam_xy_dir)

        # calculate cosin
        cosine_cur = dot_product / (magnitude1 * magnitude2)

        '''
        angle of cross_point and its neighbor label center 
        '''
        dot_product = np.dot(cac_neighbor,cam_xy_dir)

        # calculate length of vector
        magnitude1 = np.linalg.norm(cac_neighbor)
        magnitude2 = np.linalg.norm(cam_xy_dir)

        # calculate cosin
        cosine_neibor = dot_product / (magnitude1 * magnitude2)

        # assign cluster
        # find the name of this cross point
        cross_name=fnames[cross_index[i]]
        # find the 10 pair num
        matching_keys = [key for key, value in pair_name.items() if value == cross_name]
        pairs=[]
        pairs.append(cross_name)
        for i in range(len(pair_data[matching_keys[0]][1])//2):
            pairs.append(pair_name[pair_data[matching_keys[0]][1][i]])

        # view_ray towards cluster cennter
        if cosine_cur>0:
            # assign these names to corresponding cluster
            cross_label[cross_cur_label]=cross_label[cross_cur_label]+pairs
            if cosine_neibor>0:
                # this point is also in neighbor cluster, add pairs to  neighbor cluster
                cross_label[cross_cur_neibor_label]=cross_label[cross_cur_neibor_label]+pairs
        else:
            # view_ray towards neighbor cluster center
                cross_label[cross_cur_label]=cross_label[cross_cur_label]+pairs
                cross_label[cross_cur_neibor_label]=cross_label[cross_cur_neibor_label]+pairs


    fnames_numpy=np.array(fnames)
    cross_names=fnames_numpy[cross_index]

    # visulization
    # the index of cluster_new in the fnames
    indices_new={}
    # correspongding data
    data_new={}

    for i in range(cluster_num):
        # delete repeat elements
        cross_label[i]=list(set(cross_label[i]))
        # obtain original points index in original clusters
        indices_ori[i]=[key for key, x in enumerate(labels) if x == i]
        # obtian original points name in each cluster
        cluster_ori[i]= fnames_numpy[indices_ori[i]].tolist()
        # obtian cross_points name in each cluster
        cross[i]=cross_names[cross_labels == i].tolist()
        # delete the cross_points in original clusters
        cluster_no_cross[i] = [x for x in cluster_ori[i] if x not in cross[i]]
        # add new cross_points to cluster
        cluster_new[i]=cluster_no_cross[i]+cross_label[i]
        # delete repeat elements
        cluster_new[i]=list(set(cluster_new[i]))

        # visulization for each cluster(distribution and direction)
        # obtain cluster_i data

        # obtain the points of each cluster in fnames
        indices_new[i] = [key for key, x in enumerate(fnames) if x in cluster_new[i]]
        data_new[i]=data[indices_new[i]]

        x=data_new[i][:,0]
        y=data_new[i][:,1]

        # point map
        plt.scatter(x, y)

        plt.xlabel('X')
        plt.ylabel('Y')

        plt.savefig(f'distribution_{i}_new.png')

        plt.close()

        cross_c2w=poses[indices_new[i]]
        cross_rot=cross_c2w[:,:,:3]

        reference_vector = np.array([0, 0, 1])

        camera_direction = np.dot(cross_rot, reference_vector)


        u = camera_direction[:,0]
        v = camera_direction[:,1]

        plt.figure()

        # plot point map
        # plt.scatter(x, y, c='b', marker='o', label='Points')


        # plot direction of cross points
        plt.quiver(x, y, u, v, color='r', angles='xy', scale_units='xy', scale=1, label='Directions')


        plt.xlabel('X')
        plt.ylabel('Y')

        plt.title('direction')

        plt.legend()

        plt.savefig(f'cross_dir_{i}_new.png')

        plt.close()


    return indices_new,labels


def read_img(fnames,imgfolder,factor,sh):

    N = len(fnames)
    idxs = [i for i in range(N)]
    imgs=[]

    for i in tqdm(idxs):

        im = cv2.imread(os.path.join(imgfolder, fnames[i]), cv2.IMREAD_UNCHANGED)
        if im.shape[-1] == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        else:
            im = cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA)

        im = cv2.resize(im, (sh[1]//factor, sh[0]//factor), interpolation=cv2.INTER_AREA)
        im = im.astype(np.float32) / 255
        imgs.append(im)

    imgs = np.stack(imgs, -1) 
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    return imgs 


def load_data(basedir,pair,camera_bin,factor,holdout=16,cluster_num=3,cluster_cur=0,render_test=False,test_all=False,app=True):

    '''
    basedir: images and sparse folder
    pair: ten cameras similar to each camera
    camera_bin: sparse reconstruction result
    factor: downsample size
    holdout: test datasets whitch are not devided by holdout
    cluster_cur: the region to reconstruct
    rennder_test: test 
    test_all: if true, use all test datesets, else use currennt cluster datasets  
    '''

    img_folder = 'images'

    with open(os.path.join(basedir, "transforms_own_t_ENU.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    # # pfx = ".jpeg"
    pfx = ".JPG"
    # # imgfolder=basedir
    imgfolder = os.path.join(basedir, img_folder)
    fnames = [
        frame["file_path"].split("/")[-2] + "/" + frame["file_path"].split("/")[-1].split(".")[0] + pfx
        for frame in meta["frames"]
    ]
    poses_json = np.stack([np.array(frame["transform_matrix"]) for i, frame in enumerate(meta["frames"])])

    # [all_pic]
    time = np.stack([np.array(frame["time"]) for i, frame in enumerate(meta["frames"])],axis=0)

    sh = np.array(cv2.imread(os.path.join(imgfolder, fnames[0])).shape)

    # divide the test dataset, which can be divided by holdout
    whole_idx = range(0, len(fnames))
    list_whole_idx = list(whole_idx) 
    # find the index is not divided by holdout 
    test_idx = [x for x in list_whole_idx if x % holdout == 0]
    # load the test fnames and poses
    fnames_test=[fnames[i] for i in test_idx]
    poses_json_test=[poses_json[i] for i in test_idx]
    time_test=[time[i] for i in test_idx] # list

    cluster_index,labels=data_distribution(basedir,pair,camera_bin,cluster_num,factor)

    # load the cluster fnames and jsons
    if render_test:
        if test_all:
        # read train data
            test_data=read_img(fnames_test,imgfolder,factor,sh)
            test_json=np.stack(poses_json_test,axis=0)
            poses=np.ones((len(test_json),3, 5))
            # translaton and rotarion matrix
            poses[:,:,:3]=test_json[:,:3,:3]
            poses[:,:,3]=test_json[:,:3,-1]
            # size and focus of picture
            poses[:, :2, 4] = np.array(sh[:2]//factor).reshape([1, 2])
            poses[:, 2, 4] = meta["fl_x"] * 1./factor

            pic_time=np.array(time_test)
            return test_data, poses, pic_time,fnames_test
        else:
            # obtain test_idx label
            labels_cluster_test=[labels[i] for i in test_idx]
            labels_cluster_test=np.stack(labels_cluster_test,axis=0)
            labels_cluster_test=labels_cluster_test.astype(int)
            # obtain current cluster fnames 
            fnames_test=np.array(fnames_test)
            fnames_clster_test=fnames_test[labels_cluster_test==cluster_cur].tolist()
            # obtain current cluster poses_json
            poses_json_test=np.stack(poses_json_test,axis=0)
            poses_json_cluster_test=poses_json_test[labels_cluster_test==cluster_cur]
            test_data=read_img(fnames_clster_test,imgfolder,factor,sh)
            test_json=poses_json_cluster_test
            poses=np.ones((len(test_json),3, 5))
            # translaton and rotarion matrix
            poses[:,:,:3]=test_json[:,:3,:3]
            poses[:,:,3]=test_json[:,:3,-1]
            # size and focus of picture
            poses[:, :2, 4] = np.array(sh[:2]//factor).reshape([1, 2])
            poses[:, 2, 4] = meta["fl_x"] * 1./factor
            # picture time
            pic_time=np.array(time_test)
            pic_time = pic_time[labels_cluster_test==cluster_cur]
            return test_data, poses,pic_time,fnames_clster_test
    else:
        fnames_np=np.array(fnames)
        # obtain each cluster's names and poses_json
        cluster_cur_name=fnames_np[cluster_index[cluster_cur]].tolist()
        # delect elements in test dataset
        if app==False:
            cluster_cur_name = [x for x in cluster_cur_name if x not in fnames_test]
        else:
            cluster_cur_name = [x for x in cluster_cur_name if x in fnames_test]
        cluster_cur_fnames_index = [i for i, x in enumerate(fnames) if x in cluster_cur_name]
        cluster_cur_json=poses_json[cluster_cur_fnames_index]
        # read train data
        if not test_all:
            train_data=read_img(cluster_cur_name,imgfolder,factor,sh)
        else:
            # when render_test don't load imgs
            train_data=None 
        train_json=np.stack(cluster_cur_json,axis=0)
        poses=np.ones((len(train_json),3, 5))
        # translaton and rotarion matrix
        poses[:,:,:3]=train_json[:,:3,:3]
        poses[:,:,3]=train_json[:,:3,-1]
        # size and focus of picture
        poses[:, :2, 4] = np.array(sh[:2]//factor).reshape([1, 2])
        poses[:, 2, 4] = meta["fl_x"] * 1./factor 
        # picture time
        # pic_time=np.array(time)
        pic_time = time[cluster_cur_fnames_index]
        return train_data, poses, pic_time,cluster_cur_name
    
# test
# basedir='/home/Dataset/residence'
# pair='/home/lab-wang.yuxiao/xiaohan/multi-uncer-nerf_cluster0/pair.txt'
# camera_bin='/home/Dataset/residence/sparse/bin/images.bin'
# factor=16
# a=load_data(basedir,pair,camera_bin,factor,holdout=16,cluster_num=3,cluster_cur=0,render_test=True,test_all=False)