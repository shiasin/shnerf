import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
def get_activation( activation :str):
    activation_functions = {
        "relu": F.relu,
        "sigmoid": F.sigmoid,
        "tanh": F.tanh,
        "leaky_relu": F.leaky_relu,
        "softmax": F.softmax
    }
    
    return activation_functions.get(activation,F.relu)

def image_normalization(img):
    img = torch.Tensor(img)
    within_range = torch.all(img >= 0) and torch.all(img <= 1)
    if not within_range:
        img = img / img.max()
    return img

import torch
#uniform cdf coarse
def interpolation(x, xp, fp):
    x = x.squeeze()
    xp = xp.squeeze()
    fp = fp.squeeze()
    
    inds = torch.searchsorted(xp, x, right=True).clamp(1, len(xp[0])-1)
    print(fp.shape, xp.shape)
    x0 = xp.gather(1, inds - 1)
    # print(x0)
    x1 = xp.gather(1, inds)
    y0 = fp.gather(1, inds - 1)
    y1 = fp.gather(1, inds)
    
    result = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
    # print(result)
    return result

def get_rays(width, height, intrinsic, extrinsic, n=2,normalized=False):
    print('---------in get_rays ----------')
    
    if normalized:
        i, j = torch.meshgrid(torch.linspace(0, 1, width), torch.linspace(0, 1, height), indexing='ij')
    else:
        i, j = torch.meshgrid(torch.linspace(0, width-1, width), torch.linspace(0, height-1, height), indexing='ij')
    
    directions = torch.stack([
        (i.reshape(-1) - intrinsic[0, 2]) / intrinsic[0, 0],
       (j.reshape(-1) - intrinsic[1, 2]) / -intrinsic[1, 1],
        -1*torch.ones_like(i.reshape(-1))
    ], dim=-1).unsqueeze(0).repeat(n, 1, 1)
    
    # np.savetxt('points{}.csv'.format(0), directions.squeeze().detach().cpu().numpy(), delimiter=',')
    
    rotation = extrinsic[...,None,:3, :3].permute(0,1,3,2).to(dtype= torch.float64)
    translation = extrinsic[..., :3, -1]
    # print( rotation.shape,  directions[..., :, None].shape)
    ray_d = ( directions[..., None, :].to(dtype= torch.float64) @ rotation).squeeze()

    ray_d = ray_d / torch.linalg.norm(ray_d, dim=-1, keepdim=True)

    ray_d = ray_d[None, ...].expand(n, *ray_d.shape) if ray_d.shape[0] != n else ray_d
    ray_o = translation[..., None, :].expand((n, ray_d.shape[1], translation.shape[1] ))
    return ray_d, ray_o

def complete_rays(ray_d, ray_o, near: float, far: float, view_directions= True):
    rays_shape = ray_o.shape
    near_tens = near * torch.ones((rays_shape[0], rays_shape[1], 1))
    far_tens = far * torch.ones((rays_shape[0], rays_shape[1],1 ))
    # ray_d = ray_d[None, ...] if ray_d.shape[0] != ray_o.shape[0] else ray_d

    if view_directions: 
        view_dirs = ray_d
    # print(ray_o.shape,    ray_d[None, ...].shape)
    return torch.cat([ray_o, ray_d, near_tens, far_tens, view_dirs if view_directions else torch.Tensor([]).to(ray_o)], -1)

def batch_maker(rays, img, batch_size, rand = True): 
    if rand:
        shuffled_indices = torch.randperm(rays.shape[0]).to(device=img.device) # Create a random permutation of indices
    else:
        shuffled_indices = torch.arange(rays.shape[0]).to(device=img.device)
    shuffled_rays = rays[shuffled_indices]
    shuffled_pixels = img[shuffled_indices]
    print('------------------------------------------------------------------')
    return torch.split(shuffled_rays, batch_size) , torch.split(shuffled_pixels, batch_size)

def sampling_rays(rays, coarse_samples): 
    # prepare rays  o+td
    rays_o = rays[..., :3]
    rays_d = rays[..., 3:6]
    near_bound, far_bound = rays[..., 6, None], rays[None,..., 7, None]
    samples_lspace = torch.linspace(0, 1, coarse_samples+1)
    bounded_lspace = (near_bound* (1- samples_lspace)+ far_bound* samples_lspace).squeeze()
    ts = torch.rand([rays.shape[0], coarse_samples])
    s = torch.sum(torch.stack([-1*bounded_lspace[...,None, :-1], bounded_lspace[...,None,1 :]],-2), -2).squeeze(1) 
    samples = ts * s + bounded_lspace[..., :-1]
    
    return torch.cat([rays_o[...,None, :] + samples[..., None]@ rays_d[..., None, :], rays[..., None,-3:].expand(-1,coarse_samples,-1) if rays.shape[-1]>8 else torch.tensor([]).to(rays_o)], -1), samples# batch, coarse_sample, 3
    
def importance_sampling(pts_opacity, coarse_samples, num_fine_samples, rays,near):
    rays_o = rays[..., :3]
    rays_d = rays[..., 3:6]
    near_bound = near
    
    delta = torch.cat([torch.sum(torch.stack([-1*coarse_samples[...,None, :-1], coarse_samples[...,None,1 :]],-2), -2), torch.Tensor(np.array([1e10])).expand((coarse_samples.shape[0], 1,-1))],-1)
    mul1 = 1 - torch.exp(delta.permute(0,2,1) * -pts_opacity)
    mul2 = torch.cumprod(1- mul1, 1)
    mul2_initial = torch.cat([torch.ones((mul2.shape[0], 1, 1)), mul2[..., :-1,:]], axis = -2)
    trans = mul2_initial * mul1

    rays_shape = trans.shape[:-2] 
    weights = trans + 1e-6
    pdf = weights/ torch.sum(weights, dim = -2, keepdim=True)
    cdf = torch.cumsum(pdf, dim = -2)
    # print('y')
    # print(cdf.shape)
    cdf = torch.cat([torch.zeros_like(cdf[:,:1,:]), cdf], dim = -2)
    samples = torch.cat([torch.full_like(coarse_samples[:,:1], 0), coarse_samples], dim = -1).unsqueeze(-1)
    #inverse sampling 
    uniform = torch.rand(list(rays_shape)+[num_fine_samples]).unsqueeze(-1)
    
    samples,_  = torch.sort(torch.cat([interpolation(uniform, cdf, samples), coarse_samples], dim=-1), dim=-1)
    # print(samples)
    return torch.cat([rays_o[...,None, :] + samples[...,None]@rays_d[..., None, :], rays[..., None,-3:].expand(-1,num_fine_samples+coarse_samples.shape[1],-1) if rays.shape[-1]>8 else torch.tensor([]).to(rays_o)], -1), samples# batch, coarse_sample, 3
    
      


def render(pts_rgb, pts_opacity, samples):
    # print((pts_rgb > 1).any(), (pts_opacity>1).any(),torch.isnan(pts_rgb).any())
    delta = torch.cat([torch.sum(torch.stack([-1*samples[...,None, :-1], samples[...,None,1 :]],-2), -2), torch.Tensor(np.array([1e10])).expand((samples.shape[0], 1,-1))],-1)
    mul1 = 1 - torch.exp(delta.permute(0,2,1) * -pts_opacity)
    mul2 = torch.cumprod(1- mul1, 1)
    mul2_initial = torch.cat([torch.ones((mul2.shape[0], 1, 1)), mul2[..., :-1,:]], axis = -2)
    f_color = torch.sum(mul2_initial * mul1 * pts_rgb, -2)
    # print(f_color)
    # exit()
    # if (f_color > 1).any() or torch.isnan(f_color).any():
    #     print('pekh')
    #     exit()
    return f_color