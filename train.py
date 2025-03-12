import torch
import torch.nn as nn
import torch.nn.functional as F
import helper
import nn as neun
import load_blender
from load_llf import *
import numpy as np
import imageio 
import matplotlib.pyplot as plt
# torch.set_default_dtype(torch.float64)
# Set default tensor type to use GPU if available
# torch.set_default_tensor_type('torch.cuda.FloatTensor')
np.random.seed(0)
torch.set_printoptions(precision=10)  # Adjust the number of decimal places

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')
 
def visualize_loss(loss, epoch, ind):
    print('--------------------vis--------------------------')
    plt.figure()
    plt.plot(loss)
    plt.title("Simple Line Plot of a List")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.savefig('loss_{}_{}.png'.format(ind, epoch))
    
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.0)

            # torch.nn.init.zeros_(m.bias)
               
def train (data_dir, dataset_type= 'blender', rays_batch = 4096*2, coarse_samples = 60, fine_samples = 0, lrate= 5e-4, cont= False, start=0, precrop_num = 500): 
    #load the dataset
    if dataset_type == 'llff':
        images, extrinsic_params, bds, render_poses, test = load_llff_data(data_dir, 9,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=False)
        hwf =  extrinsic_params[0,:3,-1]
        height, width, focal = hwf
        extrinsic_params = extrinsic_params[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, data_dir)
        if not isinstance(test, list):
            test = [test]
        llffhold = 8
        if llffhold > 0:
            print('Auto LLFF holdout,', llffhold)
            i_test = np.arange(images.shape[0])[::llffhold]

        val = test
        
        tr = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in test and i not in val)])

        print('DEFINING BOUNDS')
        near = np.ndarray.min(bds) * .9
        far = np.ndarray.max(bds) * 1.
        # if args.no_ndc:
        #     near = np.ndarray.min(bds) * .9
        #     far = np.ndarray.max(bds) * 1.
            
        # else:
        #     near = 0.
        #     far = 1.
        print('NEAR FAR', near, far)
    
    
    
    
    if dataset_type == 'blender':
        images, extrinsic_params, render_poses, hwf, i_split = load_blender.load_blender_data(data_dir)
        i_train, i_val, i_test = i_split
        images = images[...,:3]


        near = 2.
        far = 6.
        height, width, focal = hwf
        # print(hwf)
        tr, val, test = i_split


    intrinsic_params = np.array([[focal, 0, width*0.5],
                        [0, focal, height*0.5],
                        [0, 0, 1]], dtype= np.float32)

    #rays direction and center in world coordination
    # images = torch.tensor(images)
    intrinsic_params = torch.Tensor(intrinsic_params)
    render_poses = torch.Tensor(render_poses)
    
    # model 
    embed_coord = neun.SinusoidalEncoder(max_deg=10, min_deg=0)
    embed_dir =  neun.SinusoidalEncoder(max_deg=4, min_deg=0)
    
    basic_nerf = neun.NeRF(view_direction=True, dir_input_size = embed_dir.latent_dim, coord_input_size = embed_coord.latent_dim)
    basic_nerf.apply(init_weights)
    grad_vars = list(basic_nerf.parameters())
    fine_nerf = None
    if fine_samples> 0:
        fine_nerf = neun.NeRF(view_direction=True, dir_input_size = embed_dir.latent_dim, coord_input_size = embed_coord.latent_dim)
        fine_nerf.apply(init_weights)
        grad_vars +=list(fine_nerf.parameters())
    
    
    # print(fine_nerf)
    optimizer = torch.optim.RAdam(grad_vars, lr = lrate, betas=(0.9, 0.999), weight_decay= 1e-5)
    # los = lambda x, y : (x - y)
    img2mse = lambda x, y : torch.mean((x-y)**2)
    mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))

    epochs = 300000
    
    if cont:
        checkpoint = torch.load('model_checkpointv5 99999.pth')
        
        fine_nerf.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch
        loss = checkpoint['loss']
        start = start
    
    
    loss_s = []
    new_lrate = lrate
    log = 1000
    loss_dic = {}
    counter = 0
    for epoch in range(start, epochs):
        if epoch % log ==0 and epoch!=0:
            if epoch % 3 == 0:
                ind = np.random.choice(tr, size=rays_batch//1024)
            else:
                ind = np.random.choice(val, size=rays_batch//1024)
    
        else:
            ind = np.random.choice(tr, size=rays_batch//1024)
        # ind = [0]
        target = images[ind]
        for i in ind:
            if i not in loss_dic.keys():
                loss_dic[i] = []
        target = helper.image_normalization(target)
        pose = torch.Tensor(extrinsic_params[ind, ...])
        rays_d, rays_o = helper.get_rays(width, height, intrinsic_params, pose,n= len(ind))
        
        
        rays = helper.complete_rays(rays_d, rays_o, near, far, view_directions=True)
        # if epoch == 0: np.savetxt('rays_{}.csv'.format(epoch), (rays.view(-1, rays.shape[-1]).detach().cpu().numpy())) 
        # else: pass
        # print(rays.shape, rays.view(-1, rays.shape[-1]).shape)
        if epoch % log ==0 and epoch!=0:
            batched_rays, batched_pixels = helper.batch_maker(rays.view(-1, rays.shape[-1]),target.view(-1,3) ,rays_batch, height = height, width = width,rand=False)

        else:
            batched_rays, batched_pixels = helper.batch_maker(rays.view(-1, rays.shape[-1]),target.view(-1,3) ,rays_batch, rand=True,height = height, width = width, precrop =True if epoch < precrop_num else False)

        # batched_pixels = target.view(-1, 3).split(rays_batch) #hopefully n,x*y//batched_rays, batched_rays + %, 3
        if epoch % log ==0 and epoch !=0 :
            batched = [helper.sampling_rays(batch, coarse_samples) for batch in batched_rays] # hopefully n, x*y//batched_rays, batched_rays +batched_rays % , 8
        else:
            batched = [helper.sampling_rays(batched_rays[0], coarse_samples)]
        
        final_image = torch.Tensor(np.array([],dtype= np.float32))
        final_opacity = torch.Tensor(np.array([],dtype= np.float32))
        final_depth = torch.Tensor(np.array([],dtype= np.float32))
        # print(epoch)
        for i , b in enumerate(batched):
            b0, b1 =b
            # print(b0[0:10,0:2], b0[-4:,0:2])
            # exit()
            # print(b0)
            # np.savetxt('dir{}.csv'.format(13), b0[0].squeeze().detach().cpu().numpy(), delimiter=',')
            # print('hey')
            # exit()
            bb=b1
            bb0=b1
            if epoch % log == 0 and epoch != 0:
                if i == 0:
                    final_image = torch.Tensor(np.array([], dtype= np.float32))
                    final_opacity = torch.Tensor(np.array([],dtype= np.float32))
                    final_depth = torch.Tensor(np.array([],dtype= np.float32))
                fine_nerf.eval() if (fine_nerf is not None) else basic_nerf.eval()
                rand = torch.randint(0, len(batched), (1,))
                with torch.no_grad():
                    rgb, opacity = basic_nerf(embed_coord(b0[...,0:3]), embed_dir(b0[...,3:6]))
                    
                    print(torch.all(opacity < 1e-6), torch.all(rgb < 1e-6) )
                    opac = opacity
                    print('track basic_nerf behaviour-----------------------------------------------------------------')
                    print(torch.all(rgb == rgb[0,0,0]), torch.all(opacity == opacity[0,0,0]))
                    # print(opacity)
                    if fine_nerf is not None:
                        if rand.item()//2<i<rand.item()//2+3:
                            b0, b1 = helper.importance_sampling(opacity, b1,fine_samples, batched_rays[i], near, True)
                        else:
                            b0, b1 = helper.importance_sampling(opacity, b1,fine_samples, batched_rays[i], near, False)

                        # b0, b1 = helper.sampling_rays(b, coarse_samples)
                        bb= b1
                        rgb, opacity = fine_nerf(embed_coord(b0[...,0:3]), embed_dir(b0[...,3:6]))
                    # print(torch.isnan(rgb).any(), torch.isinf(rgb).any())

                    final_color, depth = helper.render(rgb, opacity, b1, batched_rays[i])
                    
                    
                    final_image = torch.cat([final_image, final_color], axis = 0)
                    final_opacity = torch.cat([final_opacity, torch.max(opacity, axis = -2)[0]])
                    final_depth = torch.cat([final_depth, depth], axis =0)
                    opac_f =opacity
                    if rand.item()//2<i<rand.item()//2+3:
                        for x in range(rand.item(), rand.item()+2):
                            plt.figure()
                            plt.scatter(bb[x,:].cpu().numpy(), range(0,bb.shape[1]),label='Fine Samples')
                            plt.xlabel('Depth')
                            plt.ylabel('Frequency')
                            plt.legend()
                            plt.title('Distribution of Coarse and Fine Samples')
                            plt.savefig('xxxxxxxxxxxxxxxxxxxx{}{}1.png'.format(epoch,i))
                            plt.close()
                            
                            plt.figure()
                            plt.plot(bb0[x,:].cpu().numpy(), opac[x,:].cpu().numpy()*10, label = 'opacs')
                            # print(opac[x,:].cpu().numpy()*10)
                            plt.xlabel('Depth')
                            plt.ylabel('Frequency')
                            plt.legend()
                            plt.title('Distribution of Coarse and Fine Samples')
                            plt.savefig('xxxxxxxxxxxxxxxxxxxx{}{}2.png'.format(epoch,i))
                            plt.close()
                            
                            plt.figure()
                            plt.scatter(bb[x,:].cpu().numpy(),opac_f[x,:].cpu().numpy()*10, label = 'opacs')
                            # print(opac[x,:].cpu().numpy()*10)
                            plt.xlabel('Depth')
                            plt.ylabel('Frequency')
                            plt.legend()
                            plt.title('Distribution of Coarse and Fine Samples')
                            plt.savefig('xxxxxxxxxxxxxxxxxxxx{}{}3.png'.format(epoch,i))
                            plt.close()
                            
            if epoch % log != 0 or epoch == 0:
                # print('here================================')
                # b0, b1 = helper.sampling_rays(b, coarse_samples)
                # print(b0[0][0:3])
                rgb, opacity = basic_nerf(embed_coord(b0[...,0:3]), embed_dir(b0[...,3:6]))
                # if epoch == 19:
                #     print(rgb)
                #     exit()
                final_color_c, depth_c = helper.render(rgb, opacity, b1, batched_rays[i])
                loss_c = img2mse(final_color_c.unsqueeze(0),batched_pixels[i].unsqueeze(0))
                if torch.all(opacity < 1e-6) or torch.all(rgb < 1e-6):
                    print(torch.all(opacity < 1e-6), torch.all(rgb < 1e-6) )
                    counter = counter + 1
                    print('stop due to error')
                    # exit()
                # lets add fine sampling
                # np.savetxt('pointsx.csv',b0[0].detach().cpu().numpy(), delimiter=',')
                
                if fine_nerf is not None:
                    importance_samples, b1 = helper.importance_sampling(opacity, b1,fine_samples, batched_rays[i], near)
                    
                    rgb, opacity = fine_nerf(embed_coord(importance_samples[...,0:3]), embed_dir(importance_samples[...,3:6]))

                    final_color_f, depth_f = helper.render(rgb, opacity, b1, batched_rays[i])
                    loss_f = img2mse(final_color_f.unsqueeze(0),batched_pixels[i].unsqueeze(0))
                
                    
                final_image = torch.cat([final_image,(final_color_f if (fine_nerf is not None) else final_color_c)], axis = 0)
                # print(batched_pixels[i].unsqueeze(0))
                loss_img = ((loss_f+loss_c) if (fine_nerf is not None) else loss_c)
                # reg_depth = (depth_f if fine_nerf is not None else depth_c)
                # reg_depth_item = torch.mean((reg_depth[:-1, 0]-reg_depth[1:,0])**2,axis = 0)
                # regularizer = torch.mean(torch.abs(opacity)) if epoch < 0 else 0
                loss = loss_img 
                for i in ind : loss_dic[i].append(loss.item())
                
                optimizer.zero_grad()

                print('epoch: {}    loss_f: {}  loss_c:{}    loss_c:{}    psnr:{} lrate: {}'.format(epoch,loss_f.item() if fine_nerf is not None else -1, loss_c.item(), loss.item(),mse2psnr(loss).item(),optimizer.param_groups[0]['lr']))
                loss.backward()
                optimizer.step()
                # loss_s.append(loss)
                # if epoch % 1001 == 0:
                #     print(rgb.shape, b[0].shape)
                #     np.savetxt('rescolor{}.csv'.format(epoch), rgb.reshape(-1,3).detach().cpu().numpy(), delimiter=',')
                #     np.savetxt('respoints{}.csv'.format(epoch), b[0].reshape(-1,6)[..., 0:3].squeeze().detach().cpu().numpy(), delimiter=',')
                break
            
            
        
        decay_rate = 0.1
        decay_steps = 250* 1000
        new_lrate = lrate * (decay_rate ** ((epoch+start)/ decay_steps))
        # if epoch in [2,4,8]:
        #     new_lrate = lrate * 0.5
        #     lrate= new_lrate if epoch == 8 else lrate

        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
            
        if epoch % log == 0 and epoch !=0: loss_s.append(img2mse(final_image.view(-1,3), target.view(-1,3)).item())
        if epoch % log ==0 and epoch !=0 :
            # visualize_loss(loss_s, epoch)
            for i in ind:
                # print(loss_dic[i])
                visualize_loss(loss_dic[i], epoch, i)
            # loss_s = []
            print('vis')
            # np.savetxt('rescolor{}.csv'.format(epoch), rgb.squeeze().detach().cpu().numpy(), delimiter=',')
            # np.savetxt('restar.csv', target.reshape(-1,3).detach().cpu().numpy(), delimiter=',')
            normalized_image = final_image / final_image.max()
            normalized_depth = (final_depth - final_depth.min()) / final_depth.max()
            
            # print(normalized_image.shape, normalized_depth.shape)
            save_image = normalized_image.reshape((len(ind),height, width, 3)).detach().cpu().numpy()
            save_depth = normalized_depth.reshape((len(ind),height, width, 1)).detach().cpu().numpy()
            save_opac = final_opacity.reshape((len(ind),height, width, 1) ).detach().cpu().numpy()
            # print(save_opac)
            for x in range(save_image.shape[0]):
                imageio.imwrite('{} {}.png'.format(epoch, x), (save_image[x, ...]* 255).astype(np.uint8))
                imageio.imwrite('opac{} {}.png'.format(epoch, x), (save_opac[x, ...].squeeze(-1)* 255).astype(np.uint8))
                imageio.imwrite('depth{} {}.png'.format(epoch, x), (save_depth[x, ...].squeeze(-1)* 255).astype(np.uint8))

        # Save the checkpoint
        if (epoch+1) % 50000 ==0 and epoch !=0 :
            checkpoint_path = 'model_checkpointv6 {}.pth'.format(epoch)

            torch.save({
                'epoch': epoch,  # Example epoch number
                'model_state_dict': fine_nerf.state_dict() if (fine_nerf != None) else basic_nerf.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,  # Example loss
            }, checkpoint_path)
            # exit()

train('./nerf_synthetic/lego/', dataset_type= 'blender', start = 0, cont=False, rays_batch=1024, coarse_samples=64, fine_samples = 128, lrate=5e-4, precrop_num = 500)
