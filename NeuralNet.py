import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from utils import varying_multiply, form_complex_conjugate_block, ShiftedTrajectoryData

class Autoencoder(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 encoder_widths: list[int],
                 decoder_widths: list[int],
                 device) :
      super().__init__()    
      self.in_dim = in_dim
      self.out_dim = out_dim
      self.device = device
      self.encoder = self.fconn_layers(encoder_widths)
      self.decoder = self.fconn_layers(decoder_widths)

    def fconn_layers(self, widths: list[int]):
      layers = []
      for i in range(len(widths)-1):
        lin = nn.Linear(widths[i], widths[i+1], bias=True, dtype=torch.float32, device = self.device)
        self.init(lin.weight, widths[i])
        self.init(lin.bias, widths[i+1])

        layers.append(lin)
        layers.append(self.nonlinearity())

      layers.pop(-1)
      return nn.Sequential(*layers)

    def nonlinearity(self):
      return nn.ReLU()

    def init(self, w, shape):
      nn.init.uniform_(w, -1.0/np.sqrt(np.float64(shape)), 1.0/np.sqrt(np.float64(shape)))

    def forward_encoder(self, X):
      n_t, n_s, n_d = X.shape

      #Flatten X to pass through the encoder
      Xf = X.reshape(-1, n_d)

      #Pass through encoder
      Yf = self.encoder(Xf)

      #Reshape Y to get the encoded output for each sample and timestep
      n_l = Yf.size(-1)
      Y = Yf.reshape(n_t, n_s, n_l)

      return Y

    def forward_decoder(self, Y):
      n_t, n_s, n_l = Y.shape

      # Flatten Y to pass through the decoder
      Yf = Y.reshape(-1, n_l)

      # Pass through decoder
      Xf_tilde = self.decoder(Yf)

      # Reshape Xf_tilde to get reconstructed output for each sample
      n_d = Xf_tilde.size(-1)
      Xf_tilde = Xf_tilde.reshape(n_t, n_s, n_d)

      return Xf_tilde

    def forward(self, X):
      Y = self.forward_encoder(X)
      X_tilde = self.forward_decoder(Y)

      return Y, X_tilde
    
class OmegaNet(nn.Module):
    def __init__(self,
                 num_complex_pairs: int,
                 widths_omega_complex: list[int],
                 num_real: int,
                 widths_omega_real: list[int],
                 device):
      super().__init__()
      self.num_complex_pairs = num_complex_pairs
      self.num_real = num_real
      self.widths_omega_complex = widths_omega_complex
      self.widths_omega_real = widths_omega_real
      self.device = device

      # Module dictionary for storing omega_nets
      self.omega_nets = nn.ModuleDict()

      for j in range(self.num_complex_pairs):
          self.omega_nets[f"OC{j+1}"] = self.fconn_layers(self.widths_omega_complex)
      
      for j in range(self.num_real):
          self.omega_nets[f"OR{j+1}"] = self.fconn_layers(self.widths_omega_real)

    def fconn_layers(self, widths: list[int]):
      layers = []
      for i in range(len(widths)-1):
        lin = nn.Linear(widths[i], widths[i+1], bias=True, dtype=torch.float32)
        self.init(lin.weight,widths[i])
        self.init(lin.bias,widths[i+1])

        layers.append(lin)
        layers.append(self.nonlinearity())

      layers.pop(-1)
      return nn.Sequential(*layers)

    def nonlinearity(self):
      return nn.ReLU()

    def init(self, w, shape):
      nn.init.uniform_(w, -1.0/np.sqrt(np.float64(shape)), 1.0/np.sqrt(np.float64(shape)))
#      breakpoint()

    def forward(self, Y):
       n_s, n_l = Y.shape

       # Build a tuple of omegas
       omegas = []
       for j in range( self.num_complex_pairs ):
         net = self.omega_nets[f"OC{j+1}"]
         radius = torch.sum((Y[:,2*j:2*j+2]).pow(2), dim=1, keepdim=True)
         omegaf = net(radius)
         omegas.append(omegaf)

       for j in range( self.num_real ):
         net = self.omega_nets[f"OR{j+1}"]
         omegaf = net(Y[:,2*self.num_complex_pairs+j].unsqueeze(1))
         omegas.append(omegaf)
        
       return torch.cat(omegas, dim=1)

    
class DeepKoopmanNN(nn.Module):
    def __init__(
        self,
        params,
        device='cpu',
    ):
        # This isn't really the typical way you'd lay out a pytorch module;
        # usually, you separate building the model and training it more.
        # This layout is like what we did before, though, and it'll do.
        super().__init__()

        self.edcoder_depth = int((params['d'] - 4) / 2)
        self.edcoder_layers = len(params['widths'])


        self.encoder_widths = params['widths'][0:self.edcoder_depth+2]
        self.decoder_widths = params['widths'][self.edcoder_depth+2:self.edcoder_layers]

        self.encoder_decoder = Autoencoder(self.encoder_widths[0],
                                           self.decoder_widths[-1],
                                           self.encoder_widths,
                                           self.decoder_widths,
                                           device=device)

        self.widths_omega_complex = params['widths_omega_complex']
        self.widths_omega_real = params['widths_omega_real']
        self.num_complex_pairs = params['num_complex_pairs']
        self.num_real = params['num_real']

        self.omega_net = OmegaNet(num_complex_pairs=self.num_complex_pairs,
                                  widths_omega_complex=self.widths_omega_complex,
                                  num_real=self.num_real,
                                  widths_omega_real=self.widths_omega_real,
                                  device=device)
        
        self.device = device
        self.shifts_middle = params['shifts_middle']
        self.num_shifts_middle = params['num_shifts_middle']
        self.num_shifts = params['num_shifts']
        self.shifts = params['shifts']
        self.delta_t = params['delta_t']
        self.params = params

        # self.hidden_layer_sizes = hidden_layer_sizes
        # self.learning_rate = learning_rate
        # self.max_iter = max_iter
        # self.init_scale = init_scale
        # self.batch_size = batch_size
        # self.weight_decay = weight_decay
        
        # self.momentum = momentum

        # if X is not None and y is not None:
        #     self.fit(X, y)

    def cast(self, ary):
        # pytorch defaults everything to float32, unlike numpy which defaults to float64.
        # it's easier to keep everything the same,
        # and most ML uses don't really need the added precision...
        # you could use torch.set_default_dtype,
        # or pass dtype parameters everywhere you create a tensor, if you do want float64
        return torch.as_tensor(ary, dtype=torch.get_default_dtype(), device=self.device)

    def forward(self, X):
        """
        X: Tensor of shape [n_time, n_samples, input_dim]
        returns:
          y_list   : list of decoded predictions at each shift - self.shifts
          g_list   : list of latent codes at each shift - self.shifts_middle
        """
        n_t, n_s, n_d = X.shape

        # print(X.device)

        # Construct the encoding at each of the shifts
        g_all = self.encoder_decoder.forward_encoder(X)

        # print(g_all.device)

        # print(g_all.shape)
        g_shift = g_all[[0] + self.shifts_middle]

        # print(X.device, g_all.device, g_shift.device)

        n_x_tilde = len(self.shifts) + 1
        x_tilde = torch.zeros(n_x_tilde, n_s, n_d, device=X.device)

        # Begin decoding
        x_tilde[0] = self.encoder_decoder.forward_decoder(g_shift[0].unsqueeze(0)).squeeze(0)

        adv    = g_shift[0]                 # current latent [n_s, n_l]

        for idx, shift in enumerate(sorted(self.shifts), start=1):
            # recompute omegas for next step
            omegas = self.omega_net(adv)         # dict/list of omegas
            
#            print(torch.max(omegas).item(), torch.min(omegas).item(), torch.sum(torch.isnan(omegas)).item())
#            breakpoint()
            # advance sequence by omega
            adv = varying_multiply(
                adv, omegas,
                self.delta_t,
                self.num_real,
                self.num_complex_pairs
            )
            
            torch.sum(torch.isnan(adv)).item()
            # decode this new latent state
            x_tilde[idx] = self.encoder_decoder.forward_decoder(adv.unsqueeze(0)).squeeze(0)


        return x_tilde, g_shift



    '''
    shape of x:
    [num_shifts + 1 = 30, d = 3, (len_time - num_shifts) * (num_trajectories)]
    It represents the input for all the trajectories * time series and across all the shifts.

    shape of y:
    [num_shifts + 1 = 30, d = 3, (len_time - num_shifts) * (num_trajectories)]
    It represents the phi_inv (K^m phi(x_i))
    So the decoded version of the stepped through time of the encoded x
    The first dimension defines how many steps move through time

    shape of g_list:
    [num_middle_shifts = len_time - 1 = 120, d = 3, (len_time - num_shifts) * (num_trajectories)]
    It represents the list of output of encoder for each shift (encoding each step in x)
    In the loss notation, it is K^m phi(xi)
    The first dimension represents how many steps we moved forward

    '''

    def define_loss_torch(self, x, y, g_list):
      eps = 1e-5
      params = self.params
      device = x.device

      # Reconstruction loss (t=0)
      if params['relative_loss']:
          denom1 = x[0].pow(2).mean(dim=1).mean() + eps
      else:
          denom1 = torch.tensor(1.0, device=device)

      loss1 = params['recon_lam'] * F.mse_loss(y[0], x[0]) / denom1
#      print(x.shape, y.shape, g_list.shape)


      # One‑step prediction loss (shifts > 0)
      loss2 = torch.tensor(0.0, device=device)
      if self.num_shifts > 0:
          x_shifts = x[self.shifts]
          y_preds  = y[1: 1+self.num_shifts]
          

#          print(y_preds)
#          breakpoint()
          mse_shift = ((y_preds - x_shifts).pow(2)).mean(dim=(1,2))

          if params['relative_loss']:
              denom2 = x_shifts.pow(2).mean(dim=(1,2))+eps
          else:
              denom2 = torch.ones_like(mse_shift)
         
          loss2 = (1.0-params['recon_lam']) * (mse_shift/denom2).mean()
        # (-params['recon_lam'])

      # # Linearity loss in latent space
      loss3 = torch.tensor(0.0, device=device)
      # if self.num_shifts_middle > 0: 
      #     # step through middle shifts
      #     omegas   = self.omega_net(g_list[0])
      #     next_lat = varying_multiply(
      #         g_list[0], omegas, self.delta_t,
      #         self.num_real, self.num_complex_pairs
      #     )
      #     count = 0
      #     for t in range(max(self.shifts_middle)-1):
      #         if (t+1) in self.shifts_middle:
      #             if params['relative_loss']:
      #                 denom3 = g_list[count+1].pow(2).mean(dim=1).mean() + eps
      #             else:
      #                 denom3 = torch.tensor(1.0, device=device)
      #             loss3 += params['mid_shift_lam'] * F.mse_loss(next_lat, g_list[count+1]) / denom3
      #             count += 1
      #         # advance latent
      #         omegas   = self.omega_net(next_lat)
      #         next_lat = varying_multiply(
      #             next_lat, omegas, self.delta_t,
      #             self.num_real, self.num_complex_pairs
      #         )
      #     loss3 /= self.num_shifts_middle

      # Linf penalties
      if params['relative_loss']:
          L1 = x[0].abs().amax(dim=1).amax() + eps
          L2 = x[1].abs().amax(dim=1).amax() + eps
      else:
          L1 = L2 = torch.tensor(1.0, device=device)

      Linf1 = (y[0] - x[0]).abs().amax(dim=1).amax() / L1
      Linf2 = (y[1] - x[1]).abs().amax(dim=1).amax() / L2
      loss_Linf = params['Linf_lam'] * (Linf1 + Linf2)

      # loss = loss1 + loss2 + loss3 + loss_Linf
      #print(x.shape)
      #print(loss1.item(), loss2.item(), loss3.item(), loss_Linf.item())
      loss = loss1 + loss2 + loss3 + loss_Linf

      # Add regularization here
      loss_L1 = torch.tensor(0.0, device=loss.device)
      loss_L2 = torch.tensor(0.0, device=loss.device)
      for name, parameter in self.named_parameters():
        if 'bias' not in name and params['L1_lam']:
          loss_L1 += (params['L1_lam']*torch.sum(torch.abs(parameter)))
        if 'bias' not in name and params['L2_lam']:
          loss_L2 += (params['L2_lam']*torch.sum(parameter**2))


      loss = loss + loss_L1 + loss_L2
      return loss, loss1, loss2, loss3, loss_Linf

    def loss_function(self):
        # 3) Return a (yhat, y) → scalar fn for use in fit()
        def _loss_fn(yhat, y_true):
            x_tilde, g_shift = yhat
            return self.define_loss_torch(x_tilde, y_true, g_shift)
        return _loss_fn

    def init(self, weight):
        nn.init.normal_(weight, mean=0, std=self.init_scale)

    def nonlinearity(self):
        # return nn.Tanh()
        return nn.ReLU()
