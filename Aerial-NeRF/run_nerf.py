import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from run_nerf_helpers import *
# from data_distribution_func_test import load_data
from data_distribution_func import load_data
from kornia.losses import ssim as dssim
import lpips
import matplotlib.pyplot as plt
from model_components import S3IM
from sklearn.neighbors import BallTree

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)

def ssim(image_pred, image_gt, reduction='mean'):
    """
    image_pred and image_gt: (1, 3, H, W)
    """
    dssim_ = dssim(image_pred, image_gt, 3, reduction) # dissimilarity in [0, 1]
    return 1-2*dssim_ # in [-1, 1]

def cascade(fn,input_chunk,app_chunk):
    out=fn(input_chunk,app_chunk)
    return out

def batchify(fn, chunk):
    if chunk is None:
        return fn
    def ret(inputs,a_embedded):
        if a_embedded==None:
            return torch.cat([cascade(fn,inputs[i:i+chunk],a_embedded) for i in range(0, inputs.shape[0], chunk)], 0)
        else:
            return torch.cat([cascade(fn,inputs[i:i+chunk],a_embedded[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs,a_embedded,fn, embed_fn, embeddirs_fn, netchunk=1024*64):

    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)
    # rewright this function

    outputs_flat = batchify(fn, netchunk)(embedded,a_embedded)   

    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(args,rays_flat, ts,score=None,chunk=1024*32,model_num=None, **kwargs):

    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        if not args.render_test:
            score=None
            model_num=None
            ret = render_rays(args,rays_flat[i:i+chunk],ts[i:i+chunk],score,model_num, **kwargs)
        else:
            ret = render_rays(args,rays_flat[i:i+chunk],ts[:,i:i+chunk],score,model_num ,**kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])
    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(args,H, W, focal,ts,score=None,chunk=1024*32, rays=None,c2w=None,model_num=None, **kwargs):

    if c2w is not None:
        rays_o, rays_d = get_rays(H, W, focal, c2w)
    else:
        rays_o, rays_d = rays
            
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    sh = rays_d.shape

    rays_o = torch.reshape(rays_o, [-1,3]).float() # [n_rays,3]
    rays_d = torch.reshape(rays_d, [-1,3]).float() # [n_rays,3]

    rays = torch.cat([rays_o, rays_d], -1) # [n_rays,6]

    all_ret = batchify_rays(args,rays,ts,score,chunk,model_num, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map', 'depth_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def render_path_cluster(args,index,score,render_poses, hwf, chunk, render_kwargs, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor!=0:
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs_pic=[]
    betas_pic=[]

    t = time.time()

    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()

        # rays_t [n_neibor,H,W]
        rays_t= np.stack([x*np.ones((H,W)) for key,x in enumerate(index[i])],axis=0) 
        # rays_t [n_neighbor,H*W]
        rays_t=np.reshape(rays_t,[rays_t.shape[0],-1])
        rays_t=torch.tensor(rays_t).to(device)
        # [n_neighbor]
        score_index=score[i]
        score_index=torch.tensor(score_index).to(device)

        rgb,_, _, _, _ = render(args,H, W, focal, rays_t,score_index,chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)

        rgb=rgb.cpu().numpy()

        rgbs_pic.append(rgb)
        
        if i==0:
            print(rgb.shape)

        if savedir is not None:
            rgb8 = to8b(rgbs_pic[-1])
            imageio.imwrite(os.path.join(savedir, '{:03d}.png'.format(i)), rgb8)
    
    # [N_pic,H,W,C]
    rgbs = np.stack(rgbs_pic, 0)

    return rgbs

def render_path(args,index_all,score_all,score_cluster_all,render_poses, hwf, chunk, render_kwargs, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor!=0:
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs_pic=[]

    t = time.time()


    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()

        rgbs = []


        # sort index of cluster according score_cluster form min to max 
        sorted_indices = np.argsort(score_cluster_all[:,i])
        print(score_cluster_all[:,i])
        # select index < 40 [N_select]
        # stage_index=[x for key,x in enumerate(sorted_indices) if score_cluster_all[:,i][x]<40]
        # if len(stage_index)==0:
        #     stage_index = [x for key,x in enumerate(sorted_indices) if np.abs(score_cluster_all[:,i][x]-score_cluster_all[:,i][sorted_indices[0]])<1]
        
        stage_index=[sorted_indices[0]]

        print(stage_index)

        # calculate every stage rgb and beta
        for cluster in stage_index:
            print(f"picture{i} go through {cluster} cluster")

            # rays_t [n_neibor,H,W]
            rays_t= np.stack([x*np.ones((H,W)) for key,x in enumerate(index_all[cluster][i])],axis=0) 
            # rays_t [n_neighbor,H*W]
            rays_t=np.reshape(rays_t,[rays_t.shape[0],-1])
            rays_t=torch.tensor(rays_t).to(device)
            # [n_neighbor]
            score_index=score_all[cluster][i]
            score_index=torch.tensor(score_index).to(device)

            rgb,_, _, _, _ = render(args,H, W, focal, rays_t,score_index,chunk=chunk, c2w=c2w[:3,:4],model_num=cluster, **render_kwargs)

            # [H,W,3]
            rgb=rgb.cpu().numpy()

            rgbs.append(rgb)


        # fuse pictures from different clusters
        # [N_select,H,W,3]
        rgbs_final=np.stack(rgbs,axis=0)

        # mean
        # [H,W,C]
        rgbs_final=np.mean(rgbs_final,axis=0)

        rgbs_pic.append(rgbs_final)
        
        if i==0:
            print(rgb.shape)

        if savedir is not None:
            rgb8 = to8b(rgbs_pic[-1])
            imageio.imwrite(os.path.join(savedir, '{:03d}.png'.format(i)), rgb8)

    # [N_pic,H,W,C]
    rgbs = np.stack(rgbs_pic, 0)

    return rgbs


def create_nerf(args,imgs_num):
    """
    Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)

    basedir = args.basedir

    start = 0
    total_iter = 0
    optimizer=None
    render_kwargs_train=None
    render_kwargs_test=None
    FPNeRF=[]

    if args.render_test:
        if args.test_all:
            # # a pic select a nerf according to beta
            # # load n cluster ckpt

            munerf=[]
            munerf_fine=[]
            muemb=[]

            for i in range(args.cluster_num):

            #     model = NeRF(D=8, W=256, input_ch=input_ch, input_ch_views=input_ch_views,output_ch=5,skips=[4], use_viewdirs=True).to(device)
            #     model = nn.DataParallel(model)
            #     # ft_path stores the former ckpt path list. ckpt_list is the expname list
            #     ckpts = [os.path.join(basedir, args.ckpt_list[i], f) for f in sorted(os.listdir(os.path.join(basedir, args.ckpt_list[i]))) if 'tar' in f]
            #     ckpt_path = ckpts[-1]
            #     print('Reloading from', ckpt_path)
            #     ckpt = torch.load(ckpt_path)
            #     model.load_state_dict(ckpt['network_fn_state_dict'], strict=False)
            #     # freeze all parameters for front parameters
            #     model.eval()
            #     # for param in model.parameters():
            #     #     param.requires_grad = False
            #     FPNeRF.append(model)

            # load current cluster ckpt

                model = NeRF(D=8, W=256, input_ch=input_ch, input_ch_views=input_ch_views,output_ch=5,skips=[4], use_viewdirs=True,app_enc=False).to(device)
                # model = nn.DataParallel(model)
                model_fine = NeRF(D=8, W=256, input_ch=input_ch, input_ch_views=input_ch_views,input_app=args.N_appearance,output_ch=5,skips=[4], use_viewdirs=True,app_enc=True).to(device)
                # model_fine = nn.DataParallel(model_fine)
                # appearance enbeding
                embedding_appearance = torch.nn.Embedding(imgs_num[i], args.N_appearance)
            
                # ft_path stores the former ckpt path list. ckpt_list is the expname list
                ckpts = [os.path.join(basedir, args.ckpt_list[i], f) for f in sorted(os.listdir(os.path.join(basedir, args.ckpt_list[i]))) if 'tar' in f]
                ckpt_path = ckpts[-1]
                print('Reloading from', ckpt_path)
                ckpt = torch.load(ckpt_path)
                model.load_state_dict(ckpt['network_fn_state_dict'])
                # freeze all parameters for front parameters
                model.eval()
                model_fine.load_state_dict(ckpt['network_fine_state_dict'])
                model_fine.eval()

                embedding_appearance.load_state_dict(ckpt['app_enc_state_dict'])
                embedding_appearance.eval()

                munerf.append(model)
                munerf_fine.append(model_fine)
                muemb.append(embedding_appearance)

            network_query_fn = lambda inputs, viewdirs,a_embedded, network_fn : run_network(inputs, viewdirs, a_embedded,network_fn,
                                                                                    embed_fn=embed_fn,
                                                                                    embeddirs_fn=embeddirs_fn,
                                                                                    netchunk=args.netchunk
                                                                                    )
            render_kwargs_test = {
                'network_query_fn' : network_query_fn,
                'perturb' : args.perturb,
                'N_importance' : args.N_importance,
                'N_samples' : args.N_samples,
                'network_fine' : munerf_fine,
                'network_fn' : munerf,
                'white_bkgd' : args.white_bkgd,
                'raw_noise_std' : args.raw_noise_std,
                'app_enc':muemb
            }

            render_kwargs_test['perturb'] = False
            render_kwargs_test['raw_noise_std'] = 0.

        else:

            model = NeRF(D=8, W=256, input_ch=input_ch, input_ch_views=input_ch_views,output_ch=5,skips=[4], use_viewdirs=True,app_enc=False).to(device)
            # model = nn.DataParallel(model)
            model_fine = NeRF(D=8, W=256, input_ch=input_ch, input_ch_views=input_ch_views,input_app=args.N_appearance,output_ch=5,skips=[4], use_viewdirs=True,app_enc=True).to(device)
            # model_fine = nn.DataParallel(model_fine)
            # appearance enbeding
            embedding_appearance = torch.nn.Embedding(imgs_num, args.N_appearance)
        
            # ft_path stores the former ckpt path list. ckpt_list is the expname list
            ckpts = [os.path.join(basedir, args.ckpt_list[0], f) for f in sorted(os.listdir(os.path.join(basedir, args.ckpt_list[0]))) if 'tar' in f]
            ckpt_path = ckpts[-1]
            print('Reloading from', ckpt_path)
            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt['network_fn_state_dict'])
            # freeze all parameters for front parameters
            model.eval()
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])
            model_fine.eval()

            embedding_appearance.load_state_dict(ckpt['app_enc_state_dict'])
            embedding_appearance.eval()

            # for param in model.parameters():
            #     param.requires_grad = False
            FPNeRF=model

            network_query_fn = lambda inputs, viewdirs,a_embedded, network_fn : run_network(inputs, viewdirs, a_embedded,network_fn,
                                                                                    embed_fn=embed_fn,
                                                                                    embeddirs_fn=embeddirs_fn,
                                                                                    netchunk=args.netchunk
                                                                                    )
            render_kwargs_test = {
                'network_query_fn' : network_query_fn,
                'perturb' : args.perturb,
                'N_importance' : args.N_importance,
                'N_samples' : args.N_samples,
                'network_fine' : model_fine,
                'network_fn' : FPNeRF,
                'white_bkgd' : args.white_bkgd,
                'raw_noise_std' : args.raw_noise_std,
                'app_enc':embedding_appearance
            }

            render_kwargs_test['perturb'] = False
            render_kwargs_test['raw_noise_std'] = 0.


    else:
        # load cur_stage model parameters
        model = NeRF(D=8, W=256, input_ch=input_ch, input_ch_views=input_ch_views,output_ch=5,skips=[4], use_viewdirs=True,app_enc=False).to(device)
        print(model)
        # model = nn.DataParallel(model)
        grad_vars = list(model.parameters())

        model_fine = NeRF(D=8, W=256, input_ch=input_ch, input_ch_views=input_ch_views,input_app=args.N_appearance,output_ch=5,skips=[4], use_viewdirs=True,app_enc=True).to(device)
        print(model_fine)
        # model_fine = nn.DataParallel(model_fine)
        grad_vars += list(model_fine.parameters())

        # appearance enbeding
        embedding_appearance = torch.nn.Embedding(imgs_num, args.N_appearance)
        # add the parameters to optimizer
        grad_vars += list(embedding_appearance.parameters())


        network_query_fn = lambda inputs, viewdirs,a_embedded, network_fn : run_network(inputs, viewdirs, a_embedded,network_fn,
                                                                                embed_fn=embed_fn,
                                                                                embeddirs_fn=embeddirs_fn,
                                                                                netchunk=args.netchunk
                                                                                )
        
        optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))



        # load the stage_(N-1) model parameters

        if len(args.ckpt_list)>0:

            ckpt_cur_folder = [os.path.join(basedir, args.ckpt_list[-1], f) for f in sorted(os.listdir(os.path.join(basedir, args.ckpt_list[-1]))) if 'tar' in f]

            if len(ckpt_cur_folder)>0:
                ckpt_cur_path = ckpt_cur_folder[-1]
                print('Reloading from', ckpt_cur_path)
                ckpt_cur = torch.load(ckpt_cur_path)

                start = ckpt_cur['global_step']
                total_iter = ckpt_cur['total_iter']
                model.load_state_dict(ckpt_cur['network_fn_state_dict'], strict=False)

                try:
                    optimizer.load_state_dict(ckpt_cur['optimizer_state_dict'])
                except:
                    print('Start a new training stage, reset optimizer.')
                    start = 0

    # check patameters grad is true
    # for param in model.parameters():
    #     print(param.requires_grad)

        FPNeRF=model

    # check patameters grad is true
    # for param in FPNeRF[3].parameters():
    #     print(param.requires_grad)
    
        render_kwargs_train = {
            'network_query_fn' : network_query_fn,
            'perturb' : args.perturb,
            'N_importance' : args.N_importance,
            'network_fine' : model_fine,
            'N_samples' : args.N_samples,
            'network_fn' : FPNeRF,
            'white_bkgd' : args.white_bkgd,
            'raw_noise_std' : args.raw_noise_std,
            'app_enc':embedding_appearance
        }

    return render_kwargs_train, render_kwargs_test, start, total_iter, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False):
    # raw--[2048,64,4/5] z_vals--[2048,65] rays_d--[2048,3] raw_noise_std--1 white_bkgd--False
    raw2alpha = lambda raw, dists, act_fn=F.softplus: 1.-torch.exp(-act_fn(raw-1)*dists) 
    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)
    # [2048,64]
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)
    # [2048,64,3]
    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std
    # [2048,64]
    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # [2048,64]
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    # [2048,3]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)
    # [2048]
    depth_map = torch.sum(weights * z_vals, -1)
    # [2048]
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / (torch.sum(weights, -1)+1e-8))
    # [2048]
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map

def cast(origin, direction, radius, t): 
    t0, t1 = t[..., :-1], t[..., 1:]
    c, d = (t0 + t1)/2, (t1 - t0)/2
    t_mean = c + (2*c*d**2) / (3*c**2 + d**2)
    t_var = (d**2)/3 - (4/15) * ((d**4 * (12*c**2 - d**2)) / (3*c**2 + d**2)**2)
    r_var = radius**2 * ((c**2)/4 + (5/12) * d**2 - (4/15) * (d**4) / (3*c**2 + d**2))
    mean = origin[...,None,:] + direction[..., None, :] * t_mean[..., None]
    null_outer_diag = 1 - (direction**2) / torch.sum(direction**2, -1, keepdims=True)
    cov_diag = (t_var[..., None] * (direction**2)[..., None, :] + r_var[..., None] * null_outer_diag[..., None, :])
    
    return mean, cov_diag

def render_rays(args,
                ray_batch,
                ts,
                score,
                model_num,
                network_fn,
                network_query_fn,
                N_samples,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                ray_nearfar=None,
                app_enc=None):

    '''
    for test:
    ts: [N_neibor,H*W]
    score: [N_neibor]
    for train:
    ts: [n_rays]
    score: None
    '''

    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,:3], ray_batch[:,-3:]

    viewdirs = rays_d
    viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
    viewdirs = torch.reshape(viewdirs, [-1,3]).float()
    
    t_vals = torch.linspace(0., 1., steps=N_samples)

    if ray_nearfar == 'sphere': ## treats earth as a sphere and computes the intersection of a ray and a sphere

        scene_origin=[0.0, 0.0, -6371011.0]
        scene_scaling_factor=0.1

        globe_center = torch.tensor(np.array(scene_origin) * scene_scaling_factor).float()
       
        # 6371011 is earth radius, 250 is the assumed height limitation of buildings in the scene
        earth_radius = 6371011 * scene_scaling_factor
        earth_radius_plus_bldg = (6371011+80) * scene_scaling_factor
        
        ## intersect with building upper limit sphere
        delta = (2*torch.sum((rays_o-globe_center) * rays_d, dim=-1))**2 - 4*torch.norm(rays_d, dim=-1)**2 * (torch.norm((rays_o-globe_center), dim=-1)**2 - (earth_radius_plus_bldg)**2)
        d_near = (-2*torch.sum((rays_o-globe_center) * rays_d, dim=-1) - delta**0.5) / (2*torch.norm(rays_d, dim=-1)**2)
        d_near = d_near.clamp(min=1e-6)
        rays_start = rays_o + (d_near[...,None]*rays_d)
        
        ## intersect with earth
        delta = (2*torch.sum((rays_o-globe_center) * rays_d, dim=-1))**2 - 4*torch.norm(rays_d, dim=-1)**2 * (torch.norm((rays_o-globe_center), dim=-1)**2 - (earth_radius)**2)
        d_far = (-2*torch.sum((rays_o-globe_center) * rays_d, dim=-1) - delta**0.5) / (2*torch.norm(rays_d, dim=-1)**2)
        rays_end = rays_o + (d_far[...,None]*rays_d)

        '''
        when delta<0 and d_far<=0, use unbounded sampling method
        delta<=0 means the ray go though unbounded space
        d_far<0 means those rays Looking up at the sky
        '''

        # d_near is 1e-6
        condition = (delta <= 0) | (d_far <= 0)
        d_near[condition] = 1e-6
        rays_start= rays_o + (d_near[...,None]*rays_d)
        d_far[condition]=200 # 2000m
        rays_end = rays_o + (d_far[...,None]*rays_d)

        ## compute near and far for each ray
        new_near = torch.norm(rays_o - rays_start, dim=-1, keepdim=True)
        near = new_near * 0.9
        
        new_far = torch.norm(rays_o - rays_end, dim=-1, keepdim=True)
        far = new_far * 1.1


        # near=1e-1*torch.ones(N_rays,1).to(ray_batch.device)
        # far=200*torch.ones(N_rays,1).to(ray_batch.device)
        
        # disparity sampling for the first half and linear sampling for the rest
        t_vals_lindisp = torch.linspace(0., 1., steps=N_samples) 
        z_vals_lindisp = 1./(1./near * (1.-t_vals_lindisp) + 1./far * (t_vals_lindisp))
        z_vals_lindisp_half = z_vals_lindisp[:,:int(N_samples*2/3)]

        linear_start = z_vals_lindisp_half[:,-1:]
        t_vals_linear = torch.linspace(0., 1., steps=N_samples-int(N_samples*2/3)+1)
        z_vals_linear_half = linear_start * (1-t_vals_linear) + far * t_vals_linear
        
        z_vals = torch.cat((z_vals_lindisp_half, z_vals_linear_half[:,1:]), -1)
        z_vals, _ = torch.sort(z_vals, -1)
        
        z_vals = z_vals.expand([N_rays, N_samples])
        
        def nob_sample(N_samples,near,far): # near--[N_delta,1]
            # uniform sampling in range [near,far]
            t_vals_linear = torch.linspace(0., 1., steps=int(N_samples*2/3)) 
            z_vals_linear_half_1 = near * (1-t_vals_linear) + far * t_vals_linear # [N_delta,N_samples]

            # 在[far,far+1/far]上均匀采样
            linear_start = z_vals_linear_half_1[:,-1:]
            t_vals_linear = torch.linspace(0., 1., steps=N_samples-int(N_samples*2/3)+2)
            t_vals_linear=t_vals_linear[1:-1]
            z_vals_linear_half_2 = linear_start * (1-t_vals_linear) + (far+1/far) * t_vals_linear
            # 将[far,far+1/far]映射到无穷远区域
            z_vals_linear_half_2=1/((far+1/far)-z_vals_linear_half_2)
            z_vals = torch.cat((z_vals_linear_half_1, z_vals_linear_half_2), -1)
            z_vals, _ = torch.sort(z_vals, -1)

            return z_vals
            # z_vals = z_vals.expand([N_rays, N_samples])
        
        z_vals[condition]=nob_sample(N_samples,near[condition],far[condition])

    elif ray_nearfar == 'flat': ## treats earth as a flat surface and computes the intersection of a ray and a plane
        normal = torch.tensor([0, 0, 1]).to(rays_o) * scene_scaling_factor
        p0_far = torch.tensor([0, 0, 0]).to(rays_o) * scene_scaling_factor
        p0_near = torch.tensor([0, 0, 250]).to(rays_o) * scene_scaling_factor

        near = (p0_near - rays_o * normal).sum(-1) / (rays_d * normal).sum(-1)
        far = (p0_far - rays_o * normal).sum(-1) / (rays_d * normal).sum(-1)
        near = near.clamp(min=1e-6)
        near, far = near.unsqueeze(-1), far.unsqueeze(-1)

        # disparity sampling for the first half and linear sampling for the rest
        t_vals_lindisp = torch.linspace(0., 1., steps=N_samples) 
        z_vals_lindisp = 1./(1./near * (1.-t_vals_lindisp) + 1./far * (t_vals_lindisp))
        z_vals_lindisp_half = z_vals_lindisp[:,:int(N_samples*2/3)]

        linear_start = z_vals_lindisp_half[:,-1:]
        t_vals_linear = torch.linspace(0., 1., steps=N_samples-int(N_samples*2/3)+1)
        z_vals_linear_half = linear_start * (1-t_vals_linear) + far * t_vals_linear
        
        z_vals = torch.cat((z_vals_lindisp_half, z_vals_linear_half[:,1:]), -1)
        z_vals, _ = torch.sort(z_vals, -1)
        z_vals = z_vals.expand([N_rays, N_samples])

    else:
        pass
   

    if perturb > 0.:
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        t_rand = torch.rand(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]

    # don't calculate sigma and color in front stages
    # if stage < cur_stage:
    #     gate = False
    # else:
    #     gate = True
    # [2048,64,4]
    a_embedded=None
    if args.test_all:
        raw = network_query_fn(pts, viewdirs,a_embedded,network_fn[model_num])
    else:
        raw = network_query_fn(pts, viewdirs,a_embedded,network_fn)
    # render formulation

    rgb_map,disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd)

    if N_importance > 0:
        rgb_map_0,disp_map_0, acc_map_0, depth_map_0 = rgb_map,disp_map, acc_map, depth_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=False)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        if args.render_test:
            if args.test_all:
                # [N_nei,n_rays]
                ts=ts.to(torch.int)
                # obtian every index app_enc [N_nei,n_rays,48]
                # a_embedded=torch.cat([app_enc(x) for key,x in enumerate(ts)],axis=0)
                app_enc_num=app_enc[model_num]
                a_embedded=[app_enc_num(x) for key,x in enumerate(ts)]
                a_embedded=torch.stack(a_embedded,dim=0)
                # weighted add by score [N_nei,1]
                score=score.unsqueeze(1)
                # [N_nei,1,1]
                score=score.unsqueeze(2)
                a_embedded=score*a_embedded
                # [n_rays,48]
                a_embedded=torch.sum(a_embedded,0)
                a_embedded=a_embedded.float()
                run_fn = network_fine[model_num] 
            else:
                # [N_nei,n_rays]
                ts=ts.to(torch.int)
                # obtian every index app_enc [N_nei,n_rays,48]
                # a_embedded=torch.cat([app_enc(x) for key,x in enumerate(ts)],axis=0)
                a_embedded=[app_enc(x) for key,x in enumerate(ts)]
                a_embedded=torch.stack(a_embedded,dim=0)
                # weighted add by score [N_nei,1]
                score=score.unsqueeze(1)
                # [N_nei,1,1]
                score=score.unsqueeze(2)
                a_embedded=score*a_embedded
                # [n_rays,48]
                a_embedded=torch.sum(a_embedded,0)
                a_embedded=a_embedded.float()
                run_fn = network_fine 

        else:
            ts=ts.to(torch.int)
            a_embedded = app_enc(ts) # [n_rays,48]
            run_fn = network_fine 
        # [n_rays*n_samples,48]
        a_embedded=a_embedded.unsqueeze(1) # [n_rays,1,48]
        a_embedded=a_embedded.repeat(1,N_samples+N_importance,1) # [n_rays,n_samples,48]
        a_embedded=a_embedded.view(a_embedded.shape[0]*a_embedded.shape[1],a_embedded.shape[2]) # [n_rays*n_samples,48]

        # [2048,64,5]
        raw = network_query_fn(pts, viewdirs, a_embedded,run_fn)
        rgb_map,disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'depth_map' : depth_map}
    ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['depth0'] = depth_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()):
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    # trian stage_2/test stage_1
    # parser.add_argument("--ckpt_list", type=str, default=["FP_resi_uncer_alterdata_0","FP_resi_uncer_alterdata_eval_para_1"],
    #                     help='front nerf expname name')
    # test stage_3
    # parser.add_argument("--ckpt_list", type=str, default=["FP_resi_uncer_alterdata_0","FP_resi_uncer_alterdata_eval_para_1","FP_resi_uncer_alterdata_eval_para_2","FP_resi_uncer_alterdata_eval_para_3"],
    #                     help='front nerf expname name')
    # test stage_1
    # parser.add_argument("--ckpt_list", type=str, default=["FP_resi_uncer_alterdata_0","FP_resi_uncer_alterdata_eval_para_1","FP_resi_uncer_alterdata_eval_para_2","FP_resi_uncer_alterdata_eval_para_3"],
    #                     help='front nerf expname name')
    # test stage_0/train stage_1
    # parser.add_argument("--ckpt_list", type=str, default=["FP_resi_uncer_alterdata_0"],
    #                     help='front nerf expname name')
    # train stage_0
    # parser.add_argument("--ckpt_list", type=str, default=[],
    #                     help='front nerf expname name')

    # test stage_2
    # parser.add_argument("--ckpt_list", type=str, default=["FP_resi_uncer_alterdata_0","FP_resi_uncer_alterdata_eval_para_1","FP_resi_uncer_alterdata_eval_para_3stage_2"],
    #                     help='front nerf expname name')

    # test model
    parser.add_argument("--ckpt_list", type=str, default=['AN_SCUT_4_0','AN_SCUT_4_1','AN_SCUT_4_2','AN_SCUT_4_3'],
                        help='front nerf expname name')
    # train model
    # parser.add_argument("--ckpt_list", type=str, default=[],
    #                     help='front nerf expname name')

    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    
    parser.add_argument("--datadir", type=str, 
                        help='input data directory')
    
    parser.add_argument("--pair", type=str, 
                        help='input data directory')
    
    parser.add_argument("--camera_bin", type=str, 
                        help='input data directory')
    
    parser.add_argument("--cluster_cur", type=int, default=0, 
                    help='select nerf to render')
    
    parser.add_argument("--cluster_num", type=int, default=3, 
                    help='how many nerf to render the region')
    
    parser.add_argument("--N_appearance", type=int, default=48, 
                    help='number of N_appearance')
    
    parser.add_argument("--use_viewdirs", type=bool, default=True, 
                help='use full 5D input instead of 3D')
    
    parser.add_argument("--n_neibor_test", type=int, default=3, 
                help='number neibor pictures for test')
    
    parser.add_argument("--n_neibor_cluster", type=int, default=5, 
                help='number neibor pictures for calculate the score of each cluster')
    
    # training options
    parser.add_argument("--N_iters", type=int, default=200000, 
                        help='number of iters to run at current stage')
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=500, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--use_batching", action='store_true',
                        help='recommand set to False at later training stage for speed up')
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 
    parser.add_argument("--ray_nearfar", type=str, default='sphere', help='options: sphere/flat')


    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--min_multires", type=int, default=0, 
                        help='log2 of min freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--test_all", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    
    parser.add_argument("--s3im_weight", type=float, default=1.0)    
    parser.add_argument("--s3im_kernel", type=int, default=4)
    parser.add_argument("--s3im_stride", type=int, default=4)
    parser.add_argument("--s3im_repeat_time", type=int, default=10)
    parser.add_argument("--s3im_patch_height", type=int, default=64)
    parser.add_argument("--s3im_patch_width", type=int, default=64)

    # dataset options
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for blender)')
    parser.add_argument("--factor", type=int, default=None, 
                        help='downsample factor for images')
    parser.add_argument("--holdout", type=int, default=8, 
                        help='will take every 1/N images as test set')


    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
 
    return parser

def cos_calculate(vec_a,vec_b):
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        cosine_similarity = dot_product / (norm_a * norm_b)
        return cosine_similarity

def find_neighbor(poses,poses_train,n_neibor,n_neibor_clu,time,time_train):
    # for each camera,find N neibor cameras
    rot=poses[:,:,:4] # [N_pic,3,4]
    trans=poses[:,:,3] # [N_pic,3]

    rot_train=poses_train[:,:,:4] # [pic_clus,3,4]
    trans_train=poses_train[:,:,3] # [pic_clus,3]

    # find neighbor in train dataset
    balltree = BallTree(trans_train)
    cur_nei=[]
    cur_nei_score=[]
    cur_nei_score_mean=[]

    # print("find neiborhood cameras of each camera")

    idxs = [i for i in range(poses.shape[0])]
    for i in tqdm(idxs):
        # select a neibor range of a camera
        target_point = trans[i]
        # find neibor cameras in 50m of current camera
        radius=5
        ball_neighbors_indices = balltree.query_radius(np.array([target_point]), r=radius)[0]

        # ensure at least have 10 neibor cameras 
        while len(ball_neighbors_indices)<=10:
            radius=radius+1
            ball_neighbors_indices = balltree.query_radius(np.array([target_point]), r=radius)[0]

        # calculate score for each neighbor camera in train dataset
        # [N_nei_r,3,4]
        rot_neighbor=rot_train[ball_neighbors_indices]
        # [3,4]
        rot_cur=rot[i]
        # [1,3,4]
        rot_cur = np.expand_dims(rot_cur, axis=0)
        # [N_nei_r,3,4]
        rot_cur=np.repeat(rot_cur, len(rot_neighbor), axis=0)
        # [N_nei_r,3*4]
        rot_cur=np.reshape(rot_cur,(len(rot_cur),3*4))
        # [N_nei_r,3*4]
        rot_neighbor=np.reshape(rot_neighbor,(len(rot_neighbor),3*4))
        '''
        add time
        '''
        # []
        time_cur=time[i]
        # [1]
        time_cur=np.expand_dims(time_cur, axis=0)
        # [N_nei_r]
        time_cur=np.repeat(time_cur, len(rot_neighbor), axis=0)
        # [N_nei_r,1]
        time_cur=np.expand_dims(time_cur, axis=1)

        # [N_nei_r]
        time_neighbor=time_train[ball_neighbors_indices]
        # [N_nei_r,1]
        time_neighbor=np.expand_dims(time_neighbor, axis=1)

        # [N_nei_r,3*4+1]
        rot_cur=np.concatenate((rot_cur, time_cur), axis=-1)
        rot_neighbor=np.concatenate((rot_neighbor, time_neighbor), axis=-1)

        # calculate score
        # [N_nei_r,3*4]
        score=rot_neighbor-rot_cur+1e-5
        # [N_nei_r]
        score= np.linalg.norm(score,axis=1)
        # the idx of up
        sorted_indices_up = np.argsort(score)
        # min to max
        sorted_arr_up = score[sorted_indices_up]
        # select N neighbor  
        # [N_nei]
        nei_view=sorted_indices_up[:n_neibor]
        nei_view=ball_neighbors_indices[nei_view]
        # [N_nei]
        nei_score=sorted_arr_up[:n_neibor]

        # select N_nei_clu cam to calculate score to classify unkonwn cams [N_nei_clu]
        nei_score_clu=sorted_arr_up[:n_neibor_clu]
        # compute mean of nei_score for test---selecting cluster
        nei_score_mean=np.mean(nei_score_clu)

        # softmax neighbor score
        nei_score_inverse=[1/x for x in nei_score]
        nei_score_inverse=np.stack(nei_score_inverse,axis=0)
        nei_score_inverse=softmax(nei_score_inverse)

        # save N view and score
        # cur_nei.append((i,nei_view))
        cur_nei.append(nei_view)
        cur_nei_score.append(nei_score_inverse)
        cur_nei_score_mean.append(nei_score_mean)

    cur_nei=np.stack(cur_nei,axis=0) # [N_pic,N_nei]
    cur_nei_score=np.stack(cur_nei_score,axis=0) # [N_pic,N_nei]
    cur_nei_score_mean=np.stack(cur_nei_score_mean,axis=0) # [N_pic]
    return cur_nei,cur_nei_score,cur_nei_score_mean


def train():

    parser = config_parser()
    args = parser.parse_args()

    s3im_func = S3IM(kernel_size=args.s3im_kernel, stride=args.s3im_stride, repeat_time=args.s3im_repeat_time, patch_height=args.s3im_patch_height, patch_width=args.s3im_patch_width).cuda()

    # Load data
    # imgs [N_num_pic,H,W,3]
    imgs,poses,time,name = load_data(args.datadir,args.pair,args.camera_bin,args.factor,args.holdout,args.cluster_num,args.cluster_cur,args.render_test,args.test_all,app=False)

    hwf = poses[0,:3,-1]

    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if not args.render_test:
        print("calculate appearance encoder data in each cluster")
        imgs_test,poses_test,time_test,name_test = load_data(args.datadir,args.pair,args.camera_bin,args.factor,args.holdout,args.cluster_num,args.cluster_cur,args.render_test,args.test_all,app=True)
        poses_test=poses_test[:,:3,:4]
        # divide the test_data to two part, using left half to finetun appearance
        # [N_num_test,H,W/2,3]
        imgs_test=imgs_test[:,:,:W//2,:] 

    # when test load train dataset to find N neibors of each camera in test dataset
    else:
        if args.test_all:

            index_all=[]
            score_all=[]
            imgs_num_emb=[]
            # save each cluster's score to class an unknown view
            score_cluster_all=[]

            # the unknown pic go through every cluster
            for i in range(args.cluster_num):
                print(f"load {i} cluster appearance encoder data")
                # load i cluster pictures and poses,only load poses
                imgs_train,poses_train,time_train,name_train = load_data(args.datadir,args.pair,args.camera_bin,args.factor,args.holdout,args.cluster_num,cluster_cur=i,render_test=False,test_all=args.test_all,app=False)
                
                imgs_train_app,poses_train_app,time_train_app,name_train_app = load_data(args.datadir,args.pair,args.camera_bin,args.factor,args.holdout,args.cluster_num,cluster_cur=i,render_test=False,test_all=args.test_all,app=True)
                # [N_pic,H,W,5]
                poses_train = np.concatenate([poses_train, poses_train_app], 0)
                time_train=np.concatenate([time_train, time_train_app], 0)
                name_train+=name_train_app

                # find N neibor of each camera in test dataset. index,score--[N,n_neibor] for each pic, find each cluster's neibor nearest to this pic. score_cluster--[N]
                index,score,score_cluster=find_neighbor(poses,poses_train,args.n_neibor_test,args.n_neibor_cluster,time,time_train)
                index_all.append(index)
                score_all.append(score)
                # Number of pic in each cluster for calculating appearence embeding
                imgs_num=poses_train.shape[0]
                imgs_num_emb.append(imgs_num)
                score_cluster_all.append(score_cluster)

            # save all cluster's neighbor index and score
            index_all=np.stack(index_all,axis=0) # [N_cluster,N_test_pic,n_neighbor]
            score_all=np.stack(score_all,axis=0) # [N_cluster,N_test_pic,n_neighbor]
            score_cluster_all=np.stack(score_cluster_all,axis=0) # [N_cluster,N_test_pic]

        else:
            print("load appearance encoder data")
            imgs_train,poses_train,time_train,name_train = load_data(args.datadir,args.pair,args.camera_bin,args.factor,args.holdout,args.cluster_num,args.cluster_cur,render_test=False,test_all=True,app=False)
            imgs_train_app,poses_train_app,time_train_app,name_train_app = load_data(args.datadir,args.pair,args.camera_bin,args.factor,args.holdout,args.cluster_num,args.cluster_cur,render_test=False,test_all=True,app=True)
            # [N_pic,H,W,5]
            poses_train = np.concatenate([poses_train, poses_train_app], 0)
            time_train=np.concatenate([time_train, time_train_app], 0)
            name_train+=name_train_app
            # find N neibor of each camera in test dataset. index,score--[N,n_neibor] score_cluster--[N]
            index,score,score_cluster=find_neighbor(poses,poses_train,args.n_neibor_test,args.n_neibor_cluster,time,time_train)

    poses = poses[:,:3,:4]

    if args.white_bkgd:
        imgs = imgs[...,:3]*imgs[...,-1:] + (1.-imgs[...,-1:])
    else:
        imgs = imgs[...,:3]

    # if args.holdout > 0:
    #     print('Auto holdout,', args.holdout)
    #     i_test = np.arange(images.shape[0])[::args.holdout]

    # i_train = np.array([i for i in np.arange(int(images.shape[0])) if
    #                 (i not in i_test)])

    if args.render_test:
        render_poses = poses
        render_images = imgs
        
    basedir = args.basedir
    expname = args.expname

    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    if not args.render_test:
        render_kwargs_train, render_kwargs_test, start_iter, total_iter, optimizer = create_nerf(args,imgs.shape[0]+imgs_test.shape[0])
    elif args.test_all:
        render_kwargs_train, render_kwargs_test, start_iter, total_iter, optimizer = create_nerf(args,imgs_num_emb)
    else:
        render_kwargs_train, render_kwargs_test, start_iter, total_iter, optimizer = create_nerf(args,poses_train.shape[0])

    scene_stat = {
        'ray_nearfar' : args.ray_nearfar
    }
    if not args.render_test:
        render_kwargs_train.update(scene_stat)
    else:
        render_kwargs_test.update(scene_stat)

    global_step = start_iter



    if args.render_test:
        render_poses = torch.Tensor(render_poses).to(device)
        print('RENDER TEST')
        with torch.no_grad():
            testsavedir = os.path.join(basedir, expname, 'render_{:06d}'.format(start_iter))
            os.makedirs(testsavedir, exist_ok=True)
            # By default it uses the deepest output head to render result (i.e. cur_stage). 
            # Sepecify 'stage' to shallower output head for lower level of detail rendering.
            if args.test_all:
                rgbs = render_path(args,index_all,score_all,score_cluster_all,render_poses, hwf, args.chunk, render_kwargs_test, savedir=testsavedir, render_factor=args.render_factor)
            else:
                rgbs = render_path_cluster(args,index,score,render_poses, hwf, args.chunk, render_kwargs_test, savedir=testsavedir, render_factor=args.render_factor)

            # add the PSNR for test
            test_psnr_all=[]
            test_ssim_all=[]
            test_lpips_all=[]

            for i in range(rgbs.shape[0]):

                # # use Matplotlib plot uncertenty map
                # plt.imshow(betas[i], cmap='viridis')  # choose color map
                # plt.colorbar()  # add color bar
                # plt.title('beta Map')
                # plt.xlabel('X')
                # plt.ylabel('Y')

                # # save error map
                # plt.savefig(f'{testsavedir}/beta_map_{i}.png')

                # plt.clf()

                test_loss_pixel = np.sum((rgbs[i] - render_images[i]) ** 2,2)

                # use Matplotlib plot mse map
                plt.imshow(test_loss_pixel, cmap='viridis') 
                plt.colorbar()  
                plt.title('MSE Map')
                plt.xlabel('X')
                plt.ylabel('Y')

                plt.savefig(f'{testsavedir}/mse_map_{i}.png')

                plt.clf()

                # save beta[i] and test_loss in json
                # beta=betas[i].tolist()
                # with open(f"{testsavedir}/beta_{i}.json", 'w') as f:
                #     json.dump(beta, f)
                
                # test_loss_pixel = np.sum((rgbs[i] - render_images[i]) ** 2,2)
                # test_loss_json=test_loss_pixel.tolist()

                # with open(f"{testsavedir}/mse_{i}.json", 'w') as f:
                #     json.dump(test_loss_json, f)

                test_loss = np.mean((rgbs[i] - render_images[i]) ** 2)
                # psnr
                test_psnr=-10.0 * np.log(test_loss) / np.log(10.0)
                test_psnr_all.append(test_psnr)
                np.savetxt(f"{testsavedir}/{i}mean_psnr.txt", np.asarray([test_psnr]))
                # ssim
                rgbs_tensor=torch.from_numpy(rgbs[i])
                render_images_tensor=torch.from_numpy(render_images[i])
                test_ssim=ssim(rgbs_tensor.unsqueeze(0),render_images_tensor.unsqueeze(0))
                test_ssim_all.append(test_ssim)
                np.savetxt(f"{testsavedir}/{i}mean_ssim.txt", np.asarray([test_ssim]))
                # lpips
                loss_fn_alex = lpips.LPIPS(net='alex')

                rgbs_tensor_lp=rgbs_tensor.permute(2,0,1)
                rgbs_tensor_lp=rgbs_tensor_lp.unsqueeze(0)
                rgbs_tensor_lp=rgbs_tensor_lp.to("cuda:0")
                render_images_tensor_lp=render_images_tensor.permute(2,0,1)
                render_images_tensor_lp=render_images_tensor_lp.unsqueeze(0)
                render_images_tensor_lp=render_images_tensor_lp.to("cuda:0")

                test_lpips = loss_fn_alex(rgbs_tensor_lp, render_images_tensor_lp)
                test_lpips=test_lpips.cpu()
                test_lpips_all.append(test_lpips)
                np.savetxt(f"{testsavedir}/{i}mean_lpips.txt", np.asarray([test_lpips]))


            # psnr_all
            test_psnr_all=np.stack(test_psnr_all,0)
            test_psnr_all_mean=np.mean(test_psnr_all)

            # ssim_all
            test_ssim_all=np.stack(test_ssim_all,0)
            test_ssim_all_mean=np.mean(test_ssim_all)

            # lpips_all
            test_lpips_all=np.stack(test_lpips_all,0)
            test_lpips_all_mean=np.mean(test_lpips_all)

            np.savetxt(f"{testsavedir}/mean_psnr.txt", np.asarray([test_psnr_all_mean]))
            np.savetxt(f"{testsavedir}/mean_ssim.txt", np.asarray([test_ssim_all_mean]))
            np.savetxt(f"{testsavedir}/mean_lpips.txt", np.asarray([test_lpips_all_mean]))

            print('Done rendering, saved in ', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)
            return
        
    # construct rays_t for appearance encoding [N_num_pic,H,W]
    rays_t=[key*np.ones((imgs.shape[1],imgs.shape[2])) for key, x in enumerate(imgs)]
    rays_t=np.stack(rays_t,axis=0)

    # [N_num_pic,H,W/2]
    rays_t_test=[(key+imgs.shape[0])*np.ones((imgs_test.shape[1],imgs_test.shape[2])) for key, x in enumerate(imgs_test)]
    rays_t_test=np.stack(rays_t_test,axis=0)

    if args.use_batching:
        # [N,2,H,W,3]
        rays = np.stack([get_rays_np(H, W, focal, p) for p in poses], 0)
        # [N,3,H,W,3]
        rays_rgb = np.concatenate([rays, imgs[:,None]], 1)
        # [N,H,W,3,3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) 
        rays_rgb = np.stack(rays_rgb, 0) 
        # [N*H*W,3,3]
        rays_rgb = np.reshape(rays_rgb, [-1,3,3])


        # [N,2,H,W,3]
        rays_app=np.stack([get_rays_np(H, W, focal, p) for p in poses_test], 0)
        # [N,2,H,W/2,3]
        rays_app=rays_app[:,:,:,:W//2,:]
        # [N,3,H,W/2,3]
        rays_rgb_app = np.concatenate([rays_app, imgs_test[:,None]], 1) 
        # [N,H,W/2,3,3]
        rays_rgb_app = np.transpose(rays_rgb_app, [0,2,3,1,4]) 
        rays_rgb_app = np.stack(rays_rgb_app, 0) 
        # [N*H*W/2,3,3]
        rays_rgb_app = np.reshape(rays_rgb_app, [-1,3,3])

        # [N_train*H*W+N_app*H*W/2,3,3]-->[N_pixels,3,3]
        rays_rgb=np.concatenate([rays_rgb,rays_rgb_app], 0)

        # rays_t [N,H,W]-->[N*H*W]
        rays_t=np.reshape(rays_t,[-1])
        # [N,H,W/2]-->[N*H*W/2]
        rays_t_test=np.reshape(rays_t_test,[-1])
        # [N_train*H*W+N_app*H*W/2]-->[N_pixels]
        rays_t=np.concatenate([rays_t,rays_t_test], 0)

        
        print('shuffle rays')
        rand_idx = torch.randperm(rays_rgb.shape[0])
        rays_rgb = rays_rgb[rand_idx.cpu().data.numpy()]
        # shuffle rays_t
        rays_t = rays_t[rand_idx.cpu().data.numpy()]
        print('done')
        i_batch = 0


    print('Begin')
    if not args.render_test:
        print('TRAIN views are', len(imgs))
    else:
        print('TEST views are', len(imgs))

    writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))

    # if args.cur_stage<=1:
    #     end_iter=args.N_iters+1
    # else:
    #     end_iter=start_iter+1+args.N_iters+1
    
    for i in trange(start_iter+1, args.N_iters+1):

        # [batch,3,3]
        batch = torch.tensor(rays_rgb[i_batch : i_batch+args.N_rand]).to(device)
        # [3,batch,3]
        batch = torch.transpose(batch, 0, 1)
        batch_rays, target_s = batch[:2], batch[2]
        # [batch]
        batch_rays_t=torch.tensor(rays_t[i_batch : i_batch+args.N_rand]).to(device)

        i_batch += args.N_rand
        if i_batch >= rays_rgb.shape[0]:
            print("Shuffle data after an epoch!")
            rand_idx = torch.randperm(rays_rgb.shape[0])
            rays_rgb = rays_rgb[rand_idx.cpu().data.numpy()]
            rays_t = rays_t[rand_idx.cpu().data.numpy()]
            i_batch = 0
            continue

        optimizer.zero_grad()

        # return the last stage rgb
        # rgb--[2048,3]
        rgb,_, _, _, extras = render(args,H, W, focal,batch_rays_t,score=None,chunk=args.chunk, rays=batch_rays, **render_kwargs_train)

        img_loss = img2mse(rgb, target_s)
        psnr = mse2psnr(img_loss)

        loss = img_loss

        if args.s3im_weight > 0:
            s3im_pp = args.s3im_weight * s3im_func(rgb, target_s)
            loss += s3im_pp

        if 'rgb0' in extras:

            loss += img2mse(extras['rgb0'], target_s)

            s3im_pp_0 = args.s3im_weight * s3im_func(extras['rgb0'], target_s)
            loss += s3im_pp_0

        loss.backward()
    
        optimizer.step()
        
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
       
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'total_iter': total_iter,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'app_enc_state_dict': render_kwargs_train['app_enc'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
            writer.add_scalar('Train/loss', loss, total_iter)
            writer.add_scalar('Train/psnr', psnr, total_iter)

        global_step += 1
        total_iter += 1


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
