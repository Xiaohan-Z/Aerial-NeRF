import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import math

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

# class Bungee_NeRF_baseblock_feature(nn.Module):
#     def __init__(self, net_width=256, input_ch=3):
#         super(Bungee_NeRF_baseblock_feature, self).__init__()
#         self.pts_linears = nn.ModuleList([nn.Linear(input_ch, net_width)] + [nn.Linear(net_width, net_width) for _ in range(3)])

#     def forward(self, input_pts):
#         # generate nerf feature
#         h = input_pts
#         for i, l in enumerate(self.pts_linears):
#             h = self.pts_linears[i](h)
#             h = F.relu(h)
#         return h
    
# class Bungee_NeRF_baseblock_concat(nn.Module):
#     def __init__(self, net_width=256, input_feature=256):
#         super(Bungee_NeRF_baseblock_concat, self).__init__()
#         self.concat_linear = nn.Linear(input_feature + net_width, net_width)

#     def forward(self, input_feature, current_feature):
#         # concat nerf feature
#         new_feature= torch.cat([current_feature, input_feature], -1)
#         new_feature= self.concat_linear(new_feature)
#         new_feature=F.relu(new_feature)
#         return new_feature
    
# class Bungee_NeRF_baseblock_sigmargbbeta(nn.Module):
#     def __init__(self, net_width=256, input_ch_views=3):
#         super(Bungee_NeRF_baseblock_sigmargbbeta, self).__init__()
#         self.views_linear = nn.Linear(input_ch_views + net_width, net_width//2)
#         self.feature_linear = nn.Linear(net_width, net_width)
#         self.alpha_linear = nn.Linear(net_width, 1)
#         self.rgb_linear = nn.Linear(net_width//2, 3)
#         self.beta_linear = nn.Sequential(nn.Linear(net_width//2, 1),nn.Softplus())

#     def forward(self, new_feature, input_views):
#         # calculate color and sigma
#         alpha = self.alpha_linear(new_feature)
#         feature0 = self.feature_linear(new_feature)
#         h0 = torch.cat([feature0, input_views], -1)
#         h0 = self.views_linear(h0)
#         h0 = F.relu(h0)
#         rgb = self.rgb_linear(h0)
#         beta=self.beta_linear(h0)

#         return rgb, alpha,beta

# class Bungee_NeRF_block(nn.Module):
#     def __init__(self, stage,net_width=256, input_ch=3, input_ch_views=3,input_feature=256):
#         super(Bungee_NeRF_block, self).__init__()
#         self.stage=stage
#         self.input_ch = input_ch
#         self.input_ch_views = input_ch_views
#         self.num_input_feature=input_feature

#         self.baseblock_feature = Bungee_NeRF_baseblock_feature(net_width=net_width, input_ch=input_ch)
#         if self.stage>0:
#             self.baseblock_concat = Bungee_NeRF_baseblock_concat(net_width,input_feature)
#         self.baseblock_sigmargbbeta = Bungee_NeRF_baseblock_sigmargbbeta(net_width,input_ch_views)

#     def forward(self, x, feature_up=None, gate=False):
#         output=None
#         # x--embedding feature_up--last leval feature gate--only calculate color and sigma in the final stage
#         input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)

#         # stage_0 is a normal nerf
#         if self.stage==0:
#             h=self.baseblock_feature(input_pts)
#             if gate:
#                 rgbs,alphas,beta=self.baseblock_sigmargbbeta(h,input_views)
#                 output = torch.cat([rgbs,alphas,beta],-1)
#         # stage_N concat stage_(N-1) feature 
#         else:
#             h=self.baseblock_feature(input_pts)
#             h=self.baseblock_concat(feature_up,h)
#             if gate:
#                 rgbs,alphas,beta=self.baseblock_sigmargbbeta(h,input_views)
#                 output = torch.cat([rgbs,alphas,beta],-1)
#         return output,h

# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3,input_app=None, output_ch=4, skips=[4], use_viewdirs=True,app_enc=True):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.app_enc=app_enc
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        if not app_enc:
            self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])
        else: 
            self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W + input_app, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)

        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x, app=None):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            
            if app==None:
                h = torch.cat([feature, input_views], -1)
            else:
                h = torch.cat([feature, input_views,app], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs

def get_rays(H, W, focal, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1)
    dirs = dirs/torch.norm(dirs, dim=-1)[...,None]
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d

def get_rays_np(H, W, focal, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    dirs = dirs/np.linalg.norm(dirs, axis=-1)[..., None]
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1) 
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d

def get_radii_for_test(H, W, focal, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[np.newaxis, ..., np.newaxis, :] * c2w[:, np.newaxis, np.newaxis, :3,:3], -1) 
    dx = torch.sqrt(
        torch.sum((rays_d[:, :-1, :, :] - rays_d[:, 1:, :, :])**2, -1))
    dx = torch.cat([dx, dx[:, -2:-1, :]], 1)
    radii = dx[..., None] * 2 / np.sqrt(12)
    return radii

def sorted_piecewise_constant_pdf(bins, weights, num_samples, randomized):
    eps = 1e-5
    weight_sum = torch.sum(weights, axis=-1, keepdims=True)
    padding = torch.maximum(torch.tensor(0), eps - weight_sum)
    weights += padding / weights.shape[-1]
    weight_sum += padding

    pdf = weights / weight_sum
    cdf = torch.minimum(torch.tensor(1), torch.cumsum(pdf[..., :-1], axis=-1))

    cdf = torch.cat([
            torch.zeros(list(cdf.shape[:-1]) + [1]), cdf,
            torch.ones(list(cdf.shape[:-1]) + [1])
    ], axis=-1)

    if randomized:
        s = 1 / num_samples
        u = np.arange(num_samples) * s
        u = np.broadcast_to(u, list(cdf.shape[:-1]) + [num_samples])
        jitter = np.random.uniform(high=s - np.finfo('float32').eps, size=list(cdf.shape[:-1]) + [num_samples])
        u = u + jitter
        u = np.minimum(u, 1. - np.finfo('float32').eps)
    else:
        u = np.linspace(0., 1. - np.finfo('float32').eps, num_samples)
        u = np.broadcast_to(u, list(cdf.shape[:-1]) + [num_samples])

    u = torch.from_numpy(u).to(cdf)
    mask = u[..., None, :] >= cdf[..., :, None]

    def find_interval(x):
        x0 = torch.max(torch.where(mask, x[..., None], x[..., :1, None]), dim=-2)[0]
        x1 = torch.min(torch.where(~mask, x[..., None], x[..., -1:, None]), dim=-2)[0]
        return x0, x1


    bins_g0, bins_g1 = find_interval(bins)
    cdf_g0, cdf_g1 = find_interval(cdf)

    t = (u - cdf_g0) / (cdf_g1 - cdf_g0)
    t[t != t] = 0
    t = torch.clamp(t, 0, 1)
    samples = bins_g0 + t * (bins_g1 - bins_g0)
    return samples

def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples



