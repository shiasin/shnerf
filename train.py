import torch
import torch.nn as nn
import torch.nn.functional as F
import helper
import nn as neun
import load_blender
import numpy as np
import imageio 
import matplotlib.pyplot as plt
# torch.set_default_dtype(torch.float64)
# Set default tensor type to use GPU if available
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')
 
def visualize_loss(loss, epoch):
    plt.plot(loss)
    plt.title("Simple Line Plot of a List")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.savefig('loss_{}.png'.format(epoch))
    
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
               
def train (data_dir, dataset_type= 'blender', rays_batch = 4096*2, coarse_samples = 60, fine_samples = 0, lrate= 5e-4, cont= False, start=0): 
    #load the dataset
    if dataset_type == 'blender':
        images, extrinsic_params, render_poses, hwf, i_split = load_blender.load_blender_data(data_dir)
        i_train, i_val, i_test = i_split
        images = images[...,:3]
        # images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        # images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])

        near = 2.
        far = 6.
        height, width, focal = hwf
        # print(hwf)

        intrinsic_params = np.array([[focal, 0, width*0.5],
                            [0, focal, height*0.5],
                            [0, 0, 1]])
    #rays direction and center in world coordination
    torch.set_default_dtype(torch.float64)
    # images = torch.tensor(images)
    intrinsic_params = torch.Tensor(intrinsic_params)
    render_poses = torch.Tensor(render_poses)
    
    # model 
    basic_nerf = neun.Nerf(view_direction=True,encode_coord_max=10, encode_coord_min=0, encode_dir_max=4, encode_dir_min=0)
    # print(basic_nerf)
    basic_nerf.apply(init_weights)
    
    grad_vars = list(basic_nerf.parameters())

    fine_nerf = None
    if fine_samples> 0:
        fine_nerf = neun.Nerf(view_direction=True,encode_coord_max=10, encode_coord_min=0, encode_dir_max=4, encode_dir_min=0)
        fine_nerf.apply(init_weights)
        grad_vars +=list(fine_nerf.parameters())

    optimizer = torch.optim.Adam(grad_vars, lr = lrate, betas=(0.9, 0.999), weight_decay=1e-6)
    # los = lambda x, y : (x - y)
    img2mse = lambda x, y : torch.mean((x-y)**2)
    epochs = 300000
    
    if cont:
        checkpoint = torch.load('model_checkpointv4 99999.pth')
        basic_nerf.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch
        loss = checkpoint['loss']
        start = start
    
    tr, val, test = i_split
    
    loss_s = []
    new_lrate = lrate
    log = 100
    for epoch in range(start, epochs):
        
        if epoch % log ==0 and epoch!=0:
            ind = np.random.choice(val, size=rays_batch//2048)
            print(ind)
        else:
            ind = np.random.choice(tr, size=rays_batch//2048)
        target = images[ind]
        target = helper.image_normalization(target)
        pose = torch.Tensor(extrinsic_params[ind, ...]).to(dtype= torch.float64)
        rays_d, rays_o = helper.get_rays(width, height, intrinsic_params, pose,n= len(ind))
        rays = helper.complete_rays(rays_d, rays_o, near, far, view_directions=True)
        if epoch == 0: np.savetxt('rays_{}.csv'.format(epoch), (rays.view(-1, rays.shape[-1]).detach().cpu().numpy())) 
        else: pass
        # print(rays.shape, rays.view(-1, rays.shape[-1]).shape)
        if epoch % log ==0 and epoch!=0:
            batched_rays, batched_pixels = helper.batch_maker(rays.view(-1, rays.shape[-1]),target.view(-1,3) ,rays_batch, rand=False)
        else:
            batched_rays, batched_pixels = helper.batch_maker(rays.view(-1, rays.shape[-1]),target.view(-1,3) ,rays_batch, rand=True)

        # batched_pixels = target.view(-1, 3).split(rays_batch) #hopefully n,x*y//batched_rays, batched_rays + %, 3
        if epoch % log ==0 and epoch !=0 :
            batched = [helper.sampling_rays(batch, coarse_samples) for batch in batched_rays] # hopefully n, x*y//batched_rays, batched_rays +batched_rays % , 8
        else:
            batched = [helper.sampling_rays(batched_rays[0], coarse_samples)]

        final_image = torch.Tensor(np.array([]))
        final_opacity = torch.Tensor(np.array([]))
        print(epoch)
        for i , b in enumerate(batched):
            b0, b1 =b
            if epoch % log == 0 and epoch != 0:
                if i == 0:
                    final_image = torch.Tensor(np.array([]))
                    final_opacity = torch.Tensor(np.array([]))

                with torch.no_grad():
                    # b0, b1 = helper.sampling_rays(b, coarse_samples)
                    rgb, opacity = fine_nerf(b0) if (fine_nerf is not None) else basic_nerf(b0)
                    # print(torch.isnan(rgb).any(), torch.isinf(rgb).any())

                    final_color = helper.render(rgb, opacity, b1)
                    final_image = torch.cat([final_image, final_color], axis = 0)
                    final_opacity = torch.cat([final_opacity, torch.max(opacity, axis = -2)[0]])
                    
            if epoch % log != 0 or epoch == 0:
                # print('here================================')
                # b0, b1 = helper.sampling_rays(b, coarse_samples)
                # print(b0[0][0:3])
                rgb, opacity = basic_nerf(b0)
                # lets add fine sampling
                # np.savetxt('pointsx.csv',b0[0].detach().cpu().numpy(), delimiter=',')
                
                if fine_nerf is not None:
                    importance_samples, b1 = helper.importance_sampling(opacity, b1,fine_samples, batched_rays[i], near)
                    print('yes')
                    # print(importance_samples)
                    # print(importance_samples[0][0:3])
                    rgb, opacity = fine_nerf(importance_samples)
                    # np.savetxt('points.csv', importance_samples[0].detach().cpu().numpy(), delimiter=',')
                    # exit()
                    final_color_f = helper.render(rgb, opacity, b1)
                    print(final_color_f)
                    loss_f = img2mse(final_color_f.unsqueeze(0),batched_pixels[i].unsqueeze(0))
                else:
                    final_color_c = helper.render(rgb, opacity, b1)
                    loss_c = img2mse(final_color_c.unsqueeze(0),batched_pixels[i].unsqueeze(0))
                    
                final_image = torch.cat([final_image,(final_color_f if (fine_nerf is not None) else final_color_c)], axis = 0)
                # print(batched_pixels[i].unsqueeze(0))
                loss_img = (loss_f if fine_nerf is not None else loss_c)
                regularizer = torch.mean(torch.abs(opacity))
                loss = loss_img + 1e-3*regularizer
                optimizer.zero_grad()
                print('epoch: {}    loss: {}   lrate: {}'.format(epoch,loss.item(), optimizer.param_groups[0]['lr']))
                loss.backward()
                optimizer.step()
                # loss_s.append(loss)
                # if epoch % 1001 == 0:
                #     print(rgb.shape, b[0].shape)
                #     np.savetxt('rescolor{}.csv'.format(epoch), rgb.reshape(-1,3).detach().cpu().numpy(), delimiter=',')
                #     np.savetxt('respoints{}.csv'.format(epoch), b[0].reshape(-1,6)[..., 0:3].squeeze().detach().cpu().numpy(), delimiter=',')
                break
            
            
        
        decay_rate = 0.1
        decay_steps = 500* 1000
        new_lrate = lrate * (decay_rate ** ((epoch+start)/ decay_steps))
        # if epoch in [2,4,8]:
        #     new_lrate = lrate * 0.5
        #     lrate= new_lrate if epoch == 8 else lrate

        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
            
        if epoch % log == 0 and epoch !=0: loss_s.append(img2mse(final_image.view(-1,3), target.view(-1,3)).item())
        if epoch % log ==0 and epoch !=0 :
            visualize_loss(loss_s, epoch)
            # loss_s = []
            print('vis')
            # np.savetxt('rescolor{}.csv'.format(epoch), rgb.squeeze().detach().cpu().numpy(), delimiter=',')
            # np.savetxt('restar.csv', target.reshape(-1,3).detach().cpu().numpy(), delimiter=',')
            normalized_image = final_image / final_image.max()
            print(normalized_image.shape)
            save_image = normalized_image.reshape((len(ind),height, width, 3)).detach().cpu().numpy()
            save_opac = final_opacity.reshape((len(ind),height, width, 1) ).detach().cpu().numpy()
            for x in range(save_image.shape[0]):
                imageio.imwrite('{} {}.png'.format(epoch, x), (save_image[x, ...]* 255).astype(np.uint8))
                imageio.imwrite('opac{} {}.png'.format(epoch, x), (save_opac[x, ...].squeeze(-1)* 255).astype(np.uint8))

        # Save the checkpoint
        if (epoch+1) % 20000 ==0 and epoch !=0 :
            checkpoint_path = 'model_checkpointv4 {}.pth'.format(epoch)

            torch.save({
                'epoch': epoch,  # Example epoch number
                'model_state_dict': basic_nerf.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,  # Example loss
            }, checkpoint_path)
            # exit()

train('./nerf_synthetic/lego/', start = 0, cont=False, rays_batch=2048, coarse_samples=60, fine_samples = 128, lrate=5e-4)
