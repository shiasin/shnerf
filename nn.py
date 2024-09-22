import torch
import torch.nn as nn
import torch.nn.functional as F
import helper

class SinusoidalEncoder(nn.Module):

    def __init__(self, min_deg, max_deg, periodic_fn = [torch.sin, torch.cos]):
        super().__init__()
        self.x_dim = 3
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.periodic_fn = periodic_fn
        self.scales = torch.tensor([2**i for i in range(min_deg, max_deg)])

    @property
    def latent_dim(self):
        return self.x_dim * len(self.periodic_fn) * len(self.scales)
    
    def forward(self, x: torch.Tensor) :
        #  sin (x) sin(y) sin(z)  cos(x) cos(y)  cos(z)  sin(2x) sin(2x)  sin(2z) ...
        xb = torch.flatten((x.unsqueeze(-1) * self.scales[None, None,...]), start_dim=2)
        # print(x)
        return torch.cat([fn(xb) for fn in self.periodic_fn], axis = -1)


class MLP(nn.Module):
    mode_mapping = {
            'base': 1,
            'rgb': 2,
            'opacity': 3
        }

    def __init__(
        self,
        input_dim  = 0,
        num_layers=8,
        hidden_size=256,
        active_fn = 'relu',
        skip = True,
        skip_connect_every=4,
        output_dim = 3, 
        mode = 'base'
    ):
        super(MLP, self).__init__()
        # self.dim_input = input_dim
        self.mode = self.mode_mapping[mode] if mode in self.mode_mapping else -1
        self.active_fn = helper.get_activation(active_fn)

        self.skip = skip
        self.skip_connect = skip_connect_every
        self.layers = nn.ModuleList()
        if num_layers > 0:
            self.layers.append(nn.Linear(input_dim, hidden_size).to(dtype=torch.float64))

        for i in range(1, num_layers):
            if self.skip and (i % self.skip_connect == 0):
                self.layers.append(nn.Linear(input_dim + hidden_size, hidden_size).to(dtype=torch.float64))

            else:
                self.layers.append(nn.Linear(hidden_size, hidden_size).to(dtype=torch.float64))

                
        if not (self.mode == 1):
            self.final = nn.Linear(input_dim if num_layers==0 else hidden_size, output_dim).to(dtype=torch.float64)
            


    def forward(self, x):
        x_skip = x
        # print('-------------in mlp ---------')
        
        for i in range(len(self.layers)):
            if ((self.skip) and (i % self.skip_connect == 0) and (i!= 0) ):
                x = self.layers[i](torch.cat((x, x_skip), dim=-1))
            else:
                x = self.layers[i](x)
            x = self.active_fn(x)

        # self.final  
        if self.mode == 2:
            x = self.final(x) 
            
        elif self.mode == 3:
            x = self.final(x) 
            x = self.active_fn(x)
        
        return x
    
    
class Nerf (nn.Module):
    def __init__(
        self,
        num_layers=8,
        hidden_size=256,
        latent = 256,
        active_fn = 'relu',
        skip = True,
        skip_connect_every=4,
        view_direction = True,
        encode_coord_min =0,
        encode_coord_max = 10,
        encode_dir_min =0,
        encode_dir_max = 4,
        encode_periodic = [torch.sin, torch.cos]
    ):      
        super(Nerf, self).__init__()
        self.encoding_coord = SinusoidalEncoder(encode_coord_min, encode_coord_max, encode_periodic)
        self.dir_input_size = 0
        self.view_direction = view_direction
        if self.view_direction:
            self.encoding_dir = SinusoidalEncoder(encode_dir_min, encode_dir_max, encode_periodic)
            self.dir_input_size = self.encoding_dir.latent_dim
            # print('-----------')
            # print(self.dir_input_size)
        self.base = MLP(input_dim=self.encoding_coord.latent_dim, num_layers=num_layers, 
                        hidden_size=hidden_size, active_fn=active_fn, skip=skip, skip_connect_every=skip_connect_every, output_dim=latent,mode = 'base')
        
        self.density = MLP(input_dim=hidden_size, num_layers= 0, output_dim=latent+1, mode='opacity')
        self.rgb = MLP(input_dim=latent+self.dir_input_size, num_layers=1, output_dim=3, hidden_size=hidden_size//2, skip=False, mode = 'rgb') 

        # self._initialize_weights()
        
    def query_density(self, x):
        x_coord = self.encoding_coord(x[..., 0:3])
        # x_dir = self.encoding_dir(x[..., -2:])
        
        x_coord = self.base(x_coord)
        density = self.density(x_coord)
        return density[...,0].unsqueeze(-1)
    
    # def _initialize_weights(self):
    #     for mlp in [self.base, self.density, self.rgb]:
    #         for module in mlp.modules():
    #             if isinstance(module, nn.Linear):
    #                 nn.init.xavier_normal_(module.weight)
    #                 if module.bias is not None:
    #                     nn.init.constant_(module.bias, 0)
                        
    def forward (self, x): 
        print('-------in model----------')
        x_coord = self.encoding_coord(x[..., 0:3])
        
        x_coord = self.base(x_coord)
        # print( "base Nan {}, Inf {}".format(torch.isnan(x_coord).any(), torch.isinf(x_coord).any()))
        # density, latent = F.relu(self.density(x_coord)[..., 0].unsqueeze(-1)), self.density(x_coord)[..., 1:]
        den = self.density(x_coord)
        density, latent = den[...,0].unsqueeze(-1)/ (torch.max(den[...,0].unsqueeze(-1), dim=-2, keepdim=True)[0]+1e-20) , den [...,1:]
        if self.view_direction:
            # print(torch.cat(x_coord, x_dir, axis=-1))
            x_dir = self.encoding_dir(x[..., -3:])
            rgb = F.sigmoid(self.rgb(torch.cat([latent, x_dir], axis=-1)))
        else:
            rgb = F.sigmoid(self.rgb(latent))
            
        # print( "rgb Nan {}, Inf {}".format(torch.isnan(rgb).any(), torch.isinf(rgb).any()))

        return rgb, density


