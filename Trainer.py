import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn.functional as F
from NeuralNet import DeepKoopmanNN
from utils import ShiftedTrajectoryData
import copy
from pathlib import Path

class Trainer():
  def __init__(self, params, device):
      model = DeepKoopmanNN(params, device=device).to(device)
      # try:
      #     self.model = torch.compile(model, backend="nvfuser")
      # except Exception:
      #     self.model = model
      self.model = model
      self.dataset = ShiftedTrajectoryData(params, device)
      self.valdataset = ShiftedTrajectoryData(params,device)
      self.optimizer = self.make_optimizer(params)
      self.scheduler = self.make_scheduler(params)
      self.loss_fn = self.model.loss_function()
      self.params = params

      # Load the training data
      train_tensor = []
      for fileId in range(1, params['data_train_len']+1):
        ds = ShiftedTrajectoryData(params, device='cpu')
        ds.load_data(fileId=fileId, train_flag=1)

        train_tensor.append(ds.data_tensor.to(device, non_blocking=True))

      all_train = torch.cat(train_tensor, dim=1)
      all_train = all_train.permute(1,0,2).contiguous()

      full_ds = TensorDataset(all_train)
      self.train_loader = DataLoader(full_ds, batch_size=params['batch_size'], shuffle=True, pin_memory=False)
      self.scaler = GradScaler(device)

#        ds.data_tensor = ds.data_tensor.to(device, non_blocking=True) 
#        loader = DataLoader(ds,
#                       batch_size=params['batch_size'],
#                      shuffle=True,
#                        num_workers=0,       # no extra workers, data already in GPU
#                        pin_memory=False)    # irrelevant once on GPU
#        self.train_loaders.append(loader)

      # Load the validation data
      ds_val = ShiftedTrajectoryData(params, device='cpu')
      ds_val.load_data(fileId=0, train_flag=0)
      ds_val.data_tensor = ds_val.data_tensor.to(device, non_blocking=True)
      self.val_loader = DataLoader(ds_val,
                                   batch_size=params['batch_size'],
                                   shuffle=False,
                                   num_workers=0,
                                   pin_memory=False)

      self.best_val_loss = float('inf')
      self.best_checkpoint = None
      self.checkpoint_dir = Path(params['model_path']).parent
      self.checkpoint_pre = Path(params['model_path']).stem
      self.writer = SummaryWriter(log_dir=self.model.params.get('tensorboard_logdir','runs/koopman'))
      
      self.device = device
  
  def _snapshot_best(self, epoch, val_loss) : 
      self.best_val_loss = val_loss
      self.best_epoch = epoch
      self.best_model = copy.deepcopy(self.model)
      self.best_optim = copy.deepcopy(self.optimizer)
      self.best_sched = copy.deepcopy(self.scheduler)

      print(f"snapped best at epoch {epoch}, val_loss={val_loss:.6e}")


  def _save_checkpoint(self, epoch, val_loss):
    ckpt_path = self.checkpoint_dir / f"koopman_epoch{epoch:04d}.pt"
    ckpt = {
        'epoch': epoch,
        'model_state': self.best_model.state_dict(),
        'optim_state': self.best_optim.state_dict(),
        'sched_state': self.best_sched.state_dict(),
        'best_val_loss': self.best_val_loss
    }

    torch.save(ckpt, ckpt_path)
    self.best_val_loss = val_loss
    print(f"Saved checkpoint to {ckpt_path}")

  def load_checkpoint(self, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=self.device)
    self.model.load_state_dict(ckpt['model_state'])
    self.optimizer.load_state_dict(ckpt['optim_state'])
    self.scheduler.load_state_dict(ckpt['sched_state'])
    self.best_val_loss =  ckpt['best_val_loss']
    self.best_epoch = ckpt['epoch']

    self.best_model = copy.deepcopy(self.model)
    self.best_optim = copy.deepcopy(self.optimizer)
    self.best_sched = copy.deepcopy(self.scheduler)
    start_epoch = ckpt['epoch']+1
    self.best_checkpoint = ckpt
    print(f"Loaded checkpoint from {ckpt_path}")
    return start_epoch

  def make_optimizer(self, params):
    learning_rate = params['learning_rate']
    weight_decay = params['weight_decay']
    return torch.optim.Adam( self.model.parameters(),lr=learning_rate, weight_decay=weight_decay )

  def make_scheduler(self, params):
    return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.99)
  
  def pretrain_autoencoder(self, epochs):
        # 1) Freeze OmegaNet
        for p in self.model.omega_net.parameters():
            p.requires_grad = False

        # 2) Re‑init optimizer over only the autoencoder
        ae_params = list(self.model.encoder_decoder.parameters())
        ae_opt = torch.optim.Adam(ae_params,
                                  lr=self.params['learning_rate'],
                                  weight_decay=self.params['weight_decay'])

        # 3) Pretrain loop (just reconstruction loss)
        for epoch in range(1, epochs+1):
            total_loss = 0.0
            for (X_batch,) in self.train_loader:
                X = X_batch.permute(1,0,2)       # [time, batch, dim]
                # forward through AE
                with autocast(self.device):
                    _, X_tilde = self.model.encoder_decoder(X)
                    loss = F.mse_loss(X_tilde, X)
                ae_opt.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(ae_opt)
                self.scaler.update()
                total_loss += loss.item()
            print(f"[AE‑pretrain] Epoch {epoch}/{epochs}, loss={total_loss/len(self.train_loader):.6e}")

        # 4) Unfreeze OmegaNet
        for p in self.model.omega_net.parameters():
            p.requires_grad = True

        # 5) Replace your Trainer’s optimizer with full‑model optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.params['learning_rate'],
                                          weight_decay=self.params['weight_decay'])
        # and reset scheduler if you like:
        self.scheduler = self.make_scheduler(self.params)
      
  def train_one_epoch(self):
      self.model.train()
      total_loss = 0
      total_L1, total_L2, total_L3 = 0, 0, 0
      for (X_batch,) in self.train_loader:
          X = X_batch.permute(1, 0, 2)
          y = X
          self.optimizer.zero_grad()
          with autocast(self.device):
              y_pred = self.model(X)
              loss, loss1, loss2, loss3, loss_Linf = self.loss_fn(y_pred, y)
          self.scaler.scale(loss).backward()
#          self.scaler.unscale_(self.optimizer)
#          y_pred = self.model(X)
#          loss, loss1, loss2, loss3, loss_Linf = self.loss_fn(y_pred, y)
          
#          loss.backward()
#          torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
          self.scaler.step(self.optimizer)
          self.scaler.update()
#          self.optimizer.step()
          total_loss += loss.item()
          total_L1 += loss1.item()
          total_L2 += loss2.item()
          total_L3 += loss3.item()
      n_batches = len(self.train_loader)
      return total_loss / n_batches, total_L1 /n_batches,\
      total_L2 / n_batches, total_L3 / n_batches
  
  def train(self, epochs, start_epoch=1, patience=5, restart=0, pretrain_ae=0):
    # If requested, run AE pretraining first:
    if pretrain_ae > 0:
        print(f"==> Pretraining autoencoder for {pretrain_ae} epochs …")
        self.pretrain_autoencoder(pretrain_ae)

  
    current = start_epoch
    epochs_no_improvement = 0
    self.best_val_loss = float('inf')

    if restart == 1:
        ckpt_path = self.checkpoint_dir/ f"koopman_epoch{start_epoch:04d}.pt"
        next_epoch = self.load_checkpoint(ckpt_path)
        current = next_epoch
        epochs_no_improvement=0

    while current <= epochs:
        print(f"\n>>> Running epochs {current}–{epochs} (patience={patience})")
        for epoch in range(current, epochs+1):
            epoch_loss = 0.0
            epoch_L1, epoch_L2, epoch_L3 = 0.0, 0.0, 0.0

            # loop over each CSV file data loader
#            for loader in self.train_loaders:
                # train on data per file for one pass
            epoch_loss, epoch_L1, epoch_L2, epoch_L3 = self.train_one_epoch()

            val_loss   = self.evaluate(self.val_loader)
            print(f"Epoch {epoch:3d} \ntrain_loss: {epoch_loss:.6e}, val_loss: {val_loss:.6e}")
            print(f"Components: {epoch_L1:.6e}, {epoch_L2:.6e}, {epoch_L3:.6e}")
            print("-----------------------------")

            self.writer.add_scalar("Loss/train", epoch_loss, epoch)
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            self.writer.add_scalar("Loss1/train", epoch_L1, epoch)
            self.writer.add_scalar("Loss2/train", epoch_L2, epoch)
            self.writer.add_scalar("Loss3/train", epoch_L3, epoch)
            self.writer.add_scalar("LR", self.scheduler.get_last_lr()[0], epoch)
            print(f"Validation Loss improvement = {self.best_val_loss - val_loss:.6e}")

            if val_loss < self.best_val_loss - 1e-8:
                self._snapshot_best(epoch, val_loss)
#                self._save_checkpoint(epoch, val_loss)
                epochs_no_improvement = 0
            else:
                epochs_no_improvement+=1
            if epochs_no_improvement >= patience:
                print(f"No improvement for {patience} epochs. Early stopping.")
                break

            # step the scheduler once per epoch
            self.scheduler.step()

        if epochs_no_improvement >= patience and epoch < epochs:
            self.model = copy.deepcopy(self.best_model)
            self.optimizer = copy.deepcopy(self.best_optim)
            self.scheduler = copy.deepcopy(self.best_sched)
            self.best_val_loss = self.best_val_loss
            next_epoch = self.best_epoch+1
            current = next_epoch
            epochs_no_improvement = 0
            self._save_checkpoint(current-1, self.best_val_loss)
            continue
#           ckpt_path = self.checkpoint_dir / f"koopman_epoch{epoch:04d}.pt"
#           next_epoch = self.load_checkpoint(ckpt_path)
#           current = next_epoch
#           epochs_no_improvement = 0
#           continue


        break
    self.writer.close()
    print(f"Training complete (best val={self.best_val_loss:.6e}).")
    self.model = copy.deepcopy(self.best_model)
    self.optimizer = copy.deepcopy(self.best_optim)
    self.scheduler = copy.deepcopy(self.best_sched)
    self._save_checkpoint(self.best_epoch, self.best_val_loss)


  def evaluate(self, val_loader):
        self.model.eval() 
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for X_batch in val_loader:
                X = X_batch.permute(1,0,2).to(self.device)
                # for an autoencoder y==X
                y = X

                # forward pass
                y_pred = self.model(X)
                loss, loss1, loss2, loss3, loss_Linf = self.loss_fn(y_pred, y)

                # accumulate: multiply by batch size if you want a true average
                batch_size = X.shape[0]
                total_loss += loss.item() * batch_size
                total_samples += batch_size

        avg_loss = total_loss / total_samples
        return avg_loss

  def profile_short_run(self, max_steps=20):
      schedule = torch.profiler.schedule(wait=0, warmup=0,active=max_steps,repeat=1)
      with profile( activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    schedule = schedule,
                    record_shapes=True,
                    profile_memory=True,
                    on_trace_ready=torch.profiler.tensorboard_trace_handler("./runs/quick_profile"),
                    with_stack=False ) as prof:
            for step, (X_batch,) in enumerate(self.train_loader):
                with record_function("batch_forward_backward"):
                        X = X_batch.permute(1,0,2)
                        y_pred = self.model(X)
                        loss,*_ = self.loss_fn(y_pred,X)
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                prof.step()
                if step >= max_steps-1:
                    break
      print("launch `tensorboard --logdir runs/quick_profile` to inspect.")  
