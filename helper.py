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
import matplotlib.pyplot as plt

import torch
glob = 0
#uniform cdf coarse
def interpolation(x, xp, fp, pdf, draw):
    x = x.squeeze().contiguous()
    xp = xp.squeeze()
    fp = fp.squeeze()
    xpp = xp
    global glob
    ran = torch.randint(0,xpp.shape[0] , (1,)).item()
    if draw:
        # print(xp)
        if xpp.shape[0] > ran:
            for i in range(ran,ran+2):
                plt.figure()
                plt.scatter(xpp[i,:].detach().cpu().numpy(), range(0, xpp.shape[1]),label='Fine Samples')
                plt.xlabel('Depth')
                plt.ylabel('Frequency')
                plt.legend()
                plt.title('Distribution of Coarse and Fine Samples')
                plt.savefig('yyyyyyyyyyyyyyyyyy{}{}.png'.format(glob,i))
                plt.close()
                
                # plt.figure()
                # plt.scatter(pdf[i,:].detach().cpu().numpy(), range(0, pdf.shape[1]),label='Fine Samples')
                # plt.xlabel('Depth')
                # plt.ylabel('Frequency')
                # plt.legend()
                # plt.title('Distribution of Coarse and Fine Samples')
                # plt.savefig('yyyyypppppyyyyyyy{}{}.png'.format(glob,i))
                # plt.close()
    glob = glob+1
    # inds = torch.searchsorted(xp, x, right=True).clamp(1, len(xp[0])-1)
    
    inds = torch.searchsorted(xp, x, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((xp.shape[-1]-1) * torch.ones_like(inds), inds)
    
    x0 = xp.gather(1, below)
    xx = xp.gather(1, above)
    x1 = xp.gather(1, above)
    y0 = fp.gather(1, below)
    y1 = fp.gather(1, above)
    
    denom = x1-x0
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    result = y0 + (x - x0) * (y1 - y0) / denom
    # if draw:
    #     # print(inds)
    #     unique_itemsi, countsi = torch.unique(inds[ran], return_counts=True)
    #     unique_items, counts = torch.unique(xx[ran], return_counts=True)
    #     # unique_items, counts  = unique_items[unique_items.shape[0]//2], counts[unique_items[unique_items.shape[0]//2]] 
    #     with open('unique_values_counts.txt', 'a') as f:
    #         f.write("Unique values and counts:\n")
    #         for i in range(len(unique_items)):
    #             f.write(f"Value: {unique_items[i].item()}, Count: {counts[i].item()}, val:{unique_itemsi[i].item()},  Counti: {countsi[i].item()} \t")

    #         f.write(f"\n")
    #         f.write(str(xp[ran]))
    #         f.write(str(x[ran]))
    #         f.write(str(inds[ran]))
    #         f.write(str(fp[ran]))
    #         f.write(str(y0[ran]))
    #         f.write(str(result[ran]))
    #         f.write(f"\n")
    #         # exit()

        # print(result[ran])
    return result

def get_rays(width, height, intrinsic, extrinsic, n=2,normalized=False):
    print('---------in get_rays ----------')
                    # if i < args.precrop_iters:
                    # dH = int(H//2 * args.precrop_frac)
                    # dW = int(W//2 * args.precrop_frac)
                    # coords = torch.stack(
                    #     torch.meshgrid(
                    #         torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                    #         torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                    #     ), -1)
                    # if i == start:
                    #     print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}") 
    # if precop:
        
    #     if normalized: 
    #         dH = int(1//2 * 0.5)
    #         dW = int(1//2 * 0.5)
    #         i, j = torch.meshgrid(torch.linspace(1//2-dW, 1//2+dW, 2*dW),torch.linspace(1//2-dH, 1//2+dH, 2*dH), indexing='ij')
            
    #     else:
    #         dH = int(height//2 * 0.5)
    #         dW = int(width//2 * 0.5)
    #         i, j = torch.meshgrid(torch.linspace(width//2-dW, width//2+dW-1, 2*dW),torch.linspace(height//2-dH, height//2+dH-1, 2*dH), indexing='ij')
    # else:
    if normalized:
        i, j = torch.meshgrid(torch.linspace(0, 1, width), torch.linspace(0, 1, height), indexing='ij')
    else:
        i, j = torch.meshgrid(torch.linspace(0, width-1, width), torch.linspace(0, height-1, height), indexing='ij')

    i= i.t()
    j= j.t()
    directions = torch.stack([
        (i - intrinsic[0, 2]) / intrinsic[0, 0],
       (j - intrinsic[1, 2]) / -intrinsic[1, 1],
        -1*torch.ones_like(i)
    ], dim=-1)
    
    rotation = extrinsic[...,None,:3, :3].permute(0,1,3,2).float()
    translation = extrinsic[..., :3, -1]
    # print( rotation.shape,  directions[..., :, None].shape)

    ray_d = ( directions.float() @ rotation).squeeze()
    # ray_d = torch.sum(directions[...,None,: ]* extrinsic[..., :3, :3], axis = -1)
    # ray_d = ray_d / torch.norm(ray_d, dim=-1, keepdim=True)
    # np.savetxt('dir{}.csv'.format(11), directions.squeeze().detach().cpu().numpy(), delimiter=',')
    ray_d = ray_d.reshape(-1, ray_d.shape[-1])
    ray_d = ray_d[None, ...].expand(n, *ray_d.shape) if ray_d.shape[0] != n else ray_d
    print(ray_d.shape)

    ray_o = translation[..., None, :].expand((n, ray_d.shape[1], translation.shape[1] ))
    
    return ray_d.float(), ray_o.float()

def complete_rays(ray_d, ray_o, near: float, far: float, view_directions= True):
    rays_shape = ray_o.shape
    near_tens = near * torch.ones((rays_shape[0], rays_shape[1], 1))
    far_tens = far * torch.ones((rays_shape[0], rays_shape[1],1 ))

    if view_directions: 
        view_dirs = ray_d / torch.norm(ray_d, dim=-1, keepdim=True)
    return torch.cat([ray_o, ray_d, near_tens, far_tens, view_dirs if view_directions else torch.Tensor([]).to(ray_o)], -1)

def batch_maker(rays, img, batch_size, height, width,rand = True, precrop = False,crop_factor = 0.5):
    if rand:
        if precrop:
            center_h = int(height * crop_factor) 
            center_w = int(width * crop_factor)
            
            center_x_start = (width // 2) - (center_w // 2)
            center_y_start = (height // 2) - (center_h // 2)

            indices = []
            for i in range(center_x_start, center_x_start + center_w):
                for j in range(center_y_start, center_y_start + center_h):
                    indices.append(i * width + j)

            indices = torch.tensor(indices)
            shuffled_indices = indices[torch.randperm(indices.shape[0]).to(device=img.device)] # random permutation
        else:
            
            shuffled_indices = torch.randperm(rays.shape[0]).to(device=img.device)

    else:
        shuffled_indices = torch.arange(rays.shape[0]).to(device=img.device)
    shuffled_rays = rays[shuffled_indices]

    shuffled_pixels = img[shuffled_indices]
    print('------------------------------------------------------------------')
    return torch.split(shuffled_rays, batch_size) , torch.split(shuffled_pixels, batch_size)

def sampling_rays(rays, coarse_samples, perturb = False): 
    # prepare rays  o+td
    rays_o = rays[..., :3]
    rays_d = rays[..., 3:6]
    near_bound, far_bound = rays[..., 6, None], rays[None,..., 7, None]
    samples_lspace = torch.linspace(0, 1, steps = coarse_samples)
    bounded_lspace = (near_bound* (1- samples_lspace)+ far_bound* samples_lspace).squeeze()
 
    if perturb:
        mids = .5 * (bounded_lspace[...,1:] + bounded_lspace[...,:-1])
        upper = torch.cat([mids, bounded_lspace[...,-1:]], -1)
        lower = torch.cat([bounded_lspace[...,:1], mids], -1)
        # stratified samples in those intervals
        ts = torch.rand([rays.shape[0], coarse_samples])
        # s = torch.sum(torch.stack([-1*bounded_lspace[...,None, :-1], bounded_lspace[...,None,1 :]],-2), -2).squeeze(1) 
        s = upper-lower
        # samples = ts * s + bounded_lspace[..., :-1]
        samples = s* ts + lower
    else:
        samples = bounded_lspace
    # print(rays_d[0:10])
    # print(rays_d[-10:])

    # print(samples.shape)
    # print(( rays_d[...,None,:] * samples[...,:,None])[-10:, 0:2])
    # exit()
    return torch.cat([rays_o[...,None, :] + samples[..., :, None]* rays_d[..., None, :], rays[..., None,-3:].expand(-1,coarse_samples,-1) if rays.shape[-1]>8 else torch.tensor([]).to(rays_o)], -1), samples# batch, coarse_sample, 3
    
def importance_sampling(pts_opacity, coarse_samples, num_fine_samples, rays,near, raw_noise_std= 0, draw=False):
    rays_o = rays[..., :3]
    rays_d = rays[..., 3:6]
    near_bound = near
    
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(pts_opacity.shape) * raw_noise_std
        
    pts_opacity =pts_opacity + noise
    delta = torch.cat([torch.sum(torch.stack([-1*coarse_samples[...,None, :-1], coarse_samples[...,None,1 :]],-2), -2), torch.Tensor(np.array([1e10], dtype=np.float32)).expand((coarse_samples.shape[0], 1,-1))],-1)
    delta = delta * torch.norm(rays_d[...,None,:], dim=-1).unsqueeze(-1)
    mul1 = 1 - torch.exp(delta.permute(0,2,1) * -pts_opacity)
    mul2_initial = torch.cat([torch.ones((mul1.shape[0], 1, 1)), 1-mul1+1e-10], axis = -2)
    mul2 = torch.cumprod(mul2_initial, 1)[...,:-1,:]
    trans =  mul1 *mul2

    rays_shape = trans.shape[:-2] 
    weights = trans[...,1:-1,:] + 1e-10
    # print(weights.shape)
    pdfx = weights
    pdf = weights/ torch.sum(weights, dim = -2, keepdim=True)

    # print(torch.max(pdf, dim = -2), torch.min(pdf, dim = -2))
    cdf = torch.cumsum(pdf, dim = -2)
    cdf = torch.cat([torch.zeros_like(cdf[:,:1,:]), cdf], dim = -2)
    # cdf = torch.cat([ cdf, torch.full_like(cdf[:,:1,:],1.)], dim = -2)
    # print(cdf)
    # samples = torch.cat([torch.full_like(coarse_samples[:,:1], 0), torch.cat([(0.5*(coarse_samples[...,:-1]+coarse_samples[...,1:])),  coarse_samples[...,-1].unsqueeze(-1)], axis=-1)], dim = -1).unsqueeze(-1)
    # samples = torch.cat([torch.full_like(coarse_samples[:,:1], near_bound),coarse_samples], dim = -1)
    # samples = torch.cat([samples,torch.full_like(coarse_samples[:,:1], 6.)], dim = -1).unsqueeze(-1)
    samples = 0.5 * (coarse_samples[..., 1:] + coarse_samples[..., :-1])
    
    # samples = 0.5*(coarse_samples[...,:-1]+coarse_samples[...,1:])
    uniform = torch.rand(list(rays_shape)+[num_fine_samples]).unsqueeze(-1)
    samples = interpolation(uniform, cdf, samples,pdfx, draw).detach()
    samples,_  = torch.sort(torch.cat([samples, coarse_samples], dim=-1), dim=-1)
    
    return torch.cat([rays_o[...,None, :] + samples[...,None]@rays_d[..., None, :], rays[..., None,-3:].expand(-1,num_fine_samples+coarse_samples.shape[1],-1) if rays.shape[-1]>8 else torch.tensor([]).to(rays_o)], -1), samples# batch, coarse_sample, 3
    
      
def render(pts_rgb, pts_opacity, samples, rays, raw_noise_std= 0):
    rays_o = rays[..., :3]
    rays_d = rays[..., 3:6]
   

    # print((pts_rgb > 1).any(), (pts_opacity>1).any(),torch.isnan(pts_rgb).any())
    delta = torch.cat([torch.sum(torch.stack([-1*samples[...,None, :-1], samples[...,None,1 :]],-2), -2), torch.Tensor(np.array([1e10], dtype=np.float32)).expand((samples.shape[0], 1,-1))],-1)
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(pts_opacity.shape) * raw_noise_std
    pts_opacity =pts_opacity +noise

    # print( torch.norm(rays_d[...,None,:], dim=-1).unsqueeze(-1))
    delta = delta * torch.norm(rays_d[...,None,:], dim=-1).unsqueeze(-1)
    # print(pts_opacity.shape)
    mul1 = 1 - torch.exp(delta.permute(0,2,1) * -pts_opacity)
    # print(mul1.shape)
    # exit()
    # mul2_initial = torch.cat([torch.ones((mul2.shape[0], 1, 1)), mul2[..., :-1,:]], axis = -2)
    mul2_initial = torch.cat([torch.ones((mul1.shape[0], 1, 1)), 1-mul1+1e-10], axis = -2)
    mul2 = torch.cumprod(mul2_initial, -2)[...,:-1,:]
    weights = mul1 * mul2
    # weights = mul2_initial * mul1
    # print(torch.all(0<weights<1))
    # print(torch.max(weights, dim = -2), torch.min(weights, dim = -2))
    f_color = torch.sum( weights* pts_rgb, -2)
    depth = torch.sum(weights * (samples).unsqueeze(-1), axis = -2)
    
    # print(f_color)
    return f_color, depth