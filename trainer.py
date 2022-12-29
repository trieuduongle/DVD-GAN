import time
import torch
import datetime
import numpy as np

import torch.nn as nn
from torchvision.utils import save_image, make_grid
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau, StepLR, MultiStepLR, OneCycleLR
from tqdm.auto import tqdm

from Module.Generator import Generator
from Module.PatchGANDiscriminator import SNTemporalPatchGANDiscriminator
from utils import *
from metrics import *


class Trainer(object):
    def __init__(self, train_loader, val_loader, test_loader, config):

        # Data loader
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # exact model and loss
        self.model = config.model
        self.adv_loss = config.adv_loss

        # Model hyper-parameters
        self.imsize = config.imsize
        self.g_num = config.g_num
        self.z_dim = config.z_dim
        self.g_chn = config.g_chn
        self.ds_chn = config.ds_chn
        self.dt_chn = config.dt_chn
        self.n_frames = config.n_frames
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.lr_schr = config.lr_schr

        self.lambda_gp = config.lambda_gp
        self.total_epoch = config.total_epoch
        self.d_iters = config.d_iters
        self.g_iters = config.g_iters
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.g_lr = config.g_lr
        self.ds_lr = config.ds_lr
        self.dt_lr = config.dt_lr
        self.lr_decay = config.lr_decay
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.pretrained_model = config.pretrained_model

        self.n_class = config.n_class
        self.k_sample = config.k_sample
        self.dataset = config.dataset
        self.use_tensorboard = config.use_tensorboard
        self.test_batch_size = config.test_batch_size

        # path
        self.image_path = config.image_path
        self.log_path = config.log_path
        self.model_save_path = config.model_save_path
        self.sample_path = config.sample_path

        # epoch size
        self.log_epoch = config.log_epoch
        self.sample_epoch = config.sample_epoch
        self.model_save_epoch = config.model_save_epoch
        self.version = config.version

        # Path
        self.log_path = os.path.join(config.log_path, self.version)
        self.sample_path = os.path.join(config.sample_path, self.version)
        self.model_save_path = os.path.join(config.model_save_path, self.version)

        # CUSTOM
        self.in_shape = config.in_shape
        self.hid_S = config.hid_S
        self.hid_T = config.hid_T
        self.N_S = config.N_S
        self.N_T = config.N_T
        self.pre_seq_length = config.pre_seq_length
        self.aft_seq_length = config.aft_seq_length
        self.lambda_d_s = config.lambda_d_s
        self.lambda_d_t = config.lambda_d_t
        self.image_channels = config.image_channels

        self.device, self.parallel, self.gpus = set_device(config)

        self.real_label_val = 1.0
        self.fake_label_val = 0.0

        self.build_model()

        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.pretrained_model:
            print('load_pretrained_model...')
            self.load_pretrained_model()

    def label_sample(self):
        label = torch.randint(low=0, high=self.n_class, size=(self.batch_size, ))
        # label = torch.LongTensor(self.batch_size, 1).random_()%self.n_class
        # one_hot= torch.zeros(self.batch_size, self.n_class).scatter_(1, label, 1)
        return label.to(self.device)  # , one_hot.to(self.device)

    def wgan_loss(self, real_img, fake_img, tag):

        # Compute gradient penalty
        alpha = torch.rand(real_img.size(0), 1, 1, 1).cuda().expand_as(real_img)
        interpolated = torch.tensor(alpha * real_img.data + (1 - alpha) * fake_img.data, requires_grad=True)
        if tag == 'S':
            out = self.D_s(interpolated)
        else:
            out = self.D_t(interpolated)
        grad = torch.autograd.grad(outputs=out,
                                   inputs=interpolated,
                                   grad_outputs=torch.ones(out.size()).cuda(),
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        grad = grad.view(grad.size(0), -1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

        # Backward + Optimize
        loss = self.lambda_gp * d_loss_gp
        return loss

    def calc_loss(self, x, real_flag, criterion):
        if self.adv_loss != 'wgan-gp' and self.adv_loss != 'hinge':

            labels = self.get_target_label(x, real_flag)
            loss = criterion(x, labels)
        else:
            if real_flag is True:
                x = -x
            if self.adv_loss == 'wgan-gp':
                loss = torch.mean(x)
            elif self.adv_loss == 'hinge':
                loss = torch.nn.ReLU()(1.0 + x).mean()
        return loss

    def select_opt_schr(self):

        self.g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), self.g_lr,
                                            (self.beta1, self.beta2))
        self.ds_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D_s.parameters()), self.ds_lr,
                                             (self.beta1, self.beta2))
        self.dt_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D_t.parameters()), self.dt_lr,
                                             (self.beta1, self.beta2))
        if self.lr_schr == 'const':
            self.g_lr_scher = StepLR(self.g_optimizer, step_size=10000, gamma=1)
            self.ds_lr_scher = StepLR(self.ds_optimizer, step_size=10000, gamma=1)
            self.dt_lr_scher = StepLR(self.dt_optimizer, step_size=10000, gamma=1)
        elif self.lr_schr == 'step':
            self.g_lr_scher = StepLR(self.g_optimizer, step_size=500, gamma=0.98)
            self.ds_lr_scher = StepLR(self.ds_optimizer, step_size=500, gamma=0.98)
            self.dt_lr_scher = StepLR(self.dt_optimizer, step_size=500, gamma=0.98)
        elif self.lr_schr == 'exp':
            self.g_lr_scher = ExponentialLR(self.g_optimizer, gamma=0.9999)
            self.ds_lr_scher = ExponentialLR(self.ds_optimizer, gamma=0.9999)
            self.dt_lr_scher = ExponentialLR(self.dt_optimizer, gamma=0.9999)
        elif self.lr_schr == 'multi':
            self.g_lr_scher = MultiStepLR(self.g_optimizer, [10000, 30000], gamma=0.3)
            self.ds_lr_scher = MultiStepLR(self.ds_optimizer, [10000, 30000], gamma=0.3)
            self.dt_lr_scher = MultiStepLR(self.dt_optimizer, [10000, 30000], gamma=0.3)
        elif self.lr_schr == 'onecycle':
            self.g_lr_scher = OneCycleLR(self.g_optimizer,
                                         max_lr=self.g_lr,
                                         steps_per_epoch=len(self.train_loader),
                                         epochs=self.args.epochs)
            self.ds_lr_scher = OneCycleLR(self.ds_optimizer,
                                         max_lr=self.ds_lr,
                                         steps_per_epoch=len(self.train_loader),
                                         epochs=self.args.epochs)
            self.dt_lr_scher = OneCycleLR(self.dt_optimizer,
                                         max_lr=self.dt_lr,
                                         steps_per_epoch=len(self.train_loader),
                                         epochs=self.args.epochs)
        else:
            self.g_lr_scher = ReduceLROnPlateau(self.g_optimizer, mode='min',
                                                factor=self.lr_decay, patience=100,
                                                threshold=0.0001, threshold_mode='rel',
                                                cooldown=0, min_lr=1e-10, eps=1e-08,
                                                verbose=True
                            )
            self.ds_lr_scher = ReduceLROnPlateau(self.ds_optimizer, mode='min',
                                                 factor=self.lr_decay, patience=100,
                                                 threshold=0.0001, threshold_mode='rel',
                                                 cooldown=0, min_lr=1e-10, eps=1e-08,
                                                 verbose=True
                             )
            self.dt_lr_scher = ReduceLROnPlateau(self.dt_optimizer, mode='min',
                                                 factor=self.lr_decay, patience=100,
                                                 threshold=0.0001, threshold_mode='rel',
                                                 cooldown=0, min_lr=1e-10, eps=1e-08,
                                                 verbose=True
                             )

    def train(self):

        # Start with trained model
        if self.pretrained_model:
            start = self.pretrained_model + 1
        else:
            start = 1

        # Start time
        print("=" * 30, f"\nStart training from epoch {start}...")
        start_time = time.time()


        for epoch in range(start, self.total_epoch):
            start_time = time.time()

            list_ds_loss_real = []
            list_ds_loss_fake = []
            list_ds_loss = []
            list_dt_loss_real = []
            list_dt_loss_fake = []
            list_dt_loss = []
            list_g_s_loss = []
            list_g_t_loss = []
            list_g_loss = []
            list_non_g_loss = []
            list_loss = []

            self.D_s.train()
            self.D_t.train()
            self.G.train()
            
            train_pbar = tqdm(self.train_loader)
            step = 0
            for batch_x, batch_y in train_pbar:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # ================ update D d_iters times ================ #
                for i in range(self.d_iters):


                    # ============= Generate fake video ============== #
                    pred_y = self._predict(batch_x)

                    # ================== Train D_s ================== #
                    ds_out_real = self.D_s(self.merge_temporal_dim_to_batch_dim(batch_y), transpose=False)
                    ds_out_fake = self.D_s(self.merge_temporal_dim_to_batch_dim(pred_y.detach()), transpose=False)
                    ds_loss_real = self.calc_loss(ds_out_real, True, self.ds_criterion)
                    ds_loss_fake = self.calc_loss(ds_out_fake, False, self.ds_criterion)

                    # Backward + Optimize
                    ds_loss = ds_loss_real + ds_loss_fake

                    list_ds_loss.append(ds_loss.item())
                    list_ds_loss_real.append(ds_loss_real.item())
                    list_ds_loss_fake.append(ds_loss_fake.item())

                    self.reset_grad()
                    ds_loss.backward()
                    self.ds_optimizer.step()
                    self.ds_lr_scher.step()

                    # ================== Train D_t ================== #
                    dt_out_real = self.D_t(batch_y)
                    dt_out_fake = self.D_t(pred_y.detach())
                    dt_loss_real = self.calc_loss(dt_out_real, True, self.dt_criterion)
                    dt_loss_fake = self.calc_loss(dt_out_fake, False, self.dt_criterion)

                    # Backward + Optimize
                    dt_loss = dt_loss_real + dt_loss_fake

                    list_dt_loss_real.append(dt_loss_real.item())
                    list_dt_loss_fake.append(dt_loss_fake.item())
                    list_dt_loss.append(dt_loss.item())

                    self.reset_grad()
                    dt_loss.backward()
                    self.dt_optimizer.step()
                    self.dt_lr_scher.step()

                    # ================== Use wgan_gp ================== #
                    # if self.adv_loss == "wgan_gp":
                    #     dt_wgan_loss = self.wgan_loss(real_labels, fake_videos, 'T')
                    #     ds_wgan_loss = self.wgan_loss(real_labels, fake_videos, 'S')
                    #     self.reset_grad()
                    #     dt_wgan_loss.backward()
                    #     ds_wgan_loss.backward()
                    #     self.dt_optimizer.step()
                    #     self.ds_optimizer.step()

                # ==================== update G g_iters time ==================== #

                # for i in range(self.g_iters):

                    # ============= Generate fake video ============== #
                    # apply Gumbel Softmax
                    # if i > 1:
                    #     z = torch.randn(self.batch_size, self.z_dim).to(self.device)
                    #     z_class = self.label_sample()
                    #     fake_videos = self.G(z, z_class)
                    #     fake_videos_sample = sample_k_frames(fake_videos, self.n_frames, self.k_sample)
                    #     fake_videos_downsample = vid_downsample(fake_videos)

                # =========== Train G and Gumbel noise =========== #
                # Compute loss with fake images
                pred_y = self._predict(batch_x)
                g_s_out_fake = self.D_s(self.merge_temporal_dim_to_batch_dim(pred_y), transpose=False)  # Spatial Discrimminator loss
                g_t_out_fake = self.D_t(pred_y)  # Temporal Discriminator loss
                g_s_loss = self.calc_loss(g_s_out_fake, True, self.ds_criterion)
                g_t_loss = self.calc_loss(g_t_out_fake, True, self.dt_criterion)
                g_loss = self.lambda_d_s * g_s_loss + self.lambda_d_t * g_t_loss
                non_g_loss = self.g_criterion(pred_y, batch_y)
                loss = non_g_loss + g_loss

                list_g_s_loss.append(g_s_loss.item())
                list_g_t_loss.append(g_t_loss.item())
                list_g_loss.append(g_loss.item())
                list_non_g_loss.append(non_g_loss.item())
                list_loss.append(loss.item())
                # g_loss = self.calc_loss(g_s_out_fake, True) + self.calc_loss(g_t_out_fake, True)

                # Backward + Optimize
                self.reset_grad()
                loss.backward()
                self.g_optimizer.step()
                self.g_lr_scher.step()

                train_pbar.set_description(
                    f"""ds_loss: {ds_loss:.9f}, dt_loss: {dt_loss:.9f}, g_s_loss: {g_s_loss:.9f}, g_t_loss: {g_t_loss:.9f}, g_loss: {loss:.9f}, non_g_loss: {non_g_loss:.9f}""")

                step = step + 1
                if step == 10:
                    break

            # ==================== print & save part ==================== #
            # Print out log info

            ds_loss_real = np.average(list_ds_loss_real)
            ds_loss_fake = np.average(list_ds_loss_fake)
            ds_loss = np.average(list_ds_loss)
            dt_loss_real = np.average(list_dt_loss_real)
            dt_loss_fake = np.average(list_dt_loss_fake)
            dt_loss = np.average(list_dt_loss)
            g_s_loss = np.average(list_g_s_loss)
            g_t_loss = np.average(list_g_t_loss)
            g_loss = np.average(list_g_loss)
            non_g_loss = np.average(list_non_g_loss)
            loss = np.average(list_loss)

            if epoch % self.log_epoch == 0:
                self.vali()

                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                start_time = time.time()

                log_str = "Epoch: [%d/%d], time: %s, ds_loss: %.9f, dt_loss: %.9f, g_s_loss: %.9f, g_t_loss: %.9f, g_loss: %.9f, non_g_loss: %.9f, loss: %.9f, lr: %.2e, ds_lr: %.2e, dt_lr: %.2e" % \
                    (epoch, self.total_epoch, elapsed, ds_loss, dt_loss, g_s_loss, g_t_loss, g_loss, non_g_loss, loss, self.g_lr_scher.get_lr()[0], self.ds_lr_scher.get_lr()[0], self.dt_lr_scher.get_lr()[0])

                if self.use_tensorboard is True:
                    write_log(self.writer, log_str, epoch, ds_loss_real, ds_loss_fake, ds_loss, dt_loss_real, dt_loss_fake, dt_loss, g_loss, non_g_loss, loss)
                print(log_str)

            # Sample images
            if epoch % self.sample_epoch == 0:
                self.generate_samples(epoch)

            # Save model
            if epoch % self.model_save_epoch == 0:
                torch.save(self.G.state_dict(),
                        os.path.join(self.model_save_path, '{}_G.pth'.format(epoch)))
                torch.save(self.g_optimizer.state_dict(),
                        os.path.join(self.model_save_path, '{}_G_optimizer.pth'.format(epoch)))

                torch.save(self.D_s.state_dict(),
                        os.path.join(self.model_save_path, '{}_Ds.pth'.format(epoch)))
                torch.save(self.ds_optimizer.state_dict(),
                        os.path.join(self.model_save_path, '{}_Ds_optimizer.pth'.format(epoch)))

                torch.save(self.D_t.state_dict(),
                        os.path.join(self.model_save_path, '{}_Dt.pth'.format(epoch)))
                torch.save(self.dt_optimizer.state_dict(),
                        os.path.join(self.model_save_path, '{}_Dt_optimizer.pth'.format(epoch)))

    def build_model(self):

        print("=" * 30, '\nBuild_model...')

        self.G = Generator(tuple(self.in_shape), self.hid_S,
                           self.hid_T, self.N_S, self.N_T).cuda()
        self.D_s = SNTemporalPatchGANDiscriminator(self.image_channels, conv_by='2d').cuda()
        self.D_t = SNTemporalPatchGANDiscriminator(self.image_channels, conv_by='2d').cuda()

        if self.parallel:
            print('Use parallel...')
            print('gpus:', os.environ["CUDA_VISIBLE_DEVICES"])

            self.G = nn.DataParallel(self.G, device_ids=self.gpus)
            self.D_s = nn.DataParallel(self.D_s, device_ids=self.gpus)
            self.D_t = nn.DataParallel(self.D_t, device_ids=self.gpus)

        # self.G.apply(weights_init)
        # self.D.apply(weights_init)

        self.select_opt_schr()

        self.c_loss = torch.nn.CrossEntropyLoss()
        self.g_criterion = torch.nn.MSELoss()
        self.ds_criterion = torch.nn.MSELoss()
        self.dt_criterion = torch.nn.MSELoss()

    def build_tensorboard(self):
        from tensorboardX import SummaryWriter
        # from logger import Logger
        # self.logger = Logger(self.log_path)

        self.writer = SummaryWriter(log_dir=self.log_path)

    def load_pretrained_model(self):
        self.G.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_G.pth'.format(self.pretrained_model))))
        self.g_optimizer.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_G_optimizer.pth'.format(self.pretrained_model))))

        self.D_s.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_Ds.pth'.format(self.pretrained_model))))
        self.ds_optimizer.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_Ds_optimizer.pth'.format(self.pretrained_model))))

        self.D_t.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_Dt.pth'.format(self.pretrained_model))))
        self.dt_optimizer.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_Dt_optimizer.pth'.format(self.pretrained_model))))

        print('loaded trained models (epoch: {})..!'.format(self.pretrained_model))

    def reset_grad(self):
        self.ds_optimizer.zero_grad()
        self.dt_optimizer.zero_grad()
        self.g_optimizer.zero_grad()

    def save_sample(self, data_iter):
        real_images, _ = next(data_iter)
        save_image(denorm(real_images), os.path.join(self.sample_path, 'real.png'))

    def merge_temporal_dim_to_batch_dim(self, inputs):
        in_shape = list(inputs.shape)
        return inputs.view([in_shape[0] * in_shape[1]] + in_shape[2:])

    def _predict(self, batch_x):
        if self.aft_seq_length == self.pre_seq_length:
            pred_y = self.G(batch_x)
        elif self.aft_seq_length < self.pre_seq_length:
            pred_y = self.G(batch_x)
            pred_y = pred_y[:, :self.aft_seq_length]
        elif self.aft_seq_length > self.pre_seq_length:
            pred_y = []
            d = self.aft_seq_length // self.pre_seq_length
            m = self.aft_seq_length % self.pre_seq_length
            
            cur_seq = batch_x.clone()
            for _ in range(d):
                cur_seq = self.G(cur_seq)
                pred_y.append(cur_seq)

            if m != 0:
                cur_seq = self.G(cur_seq)
                pred_y.append(cur_seq[:, :m])
            
            pred_y = torch.cat(pred_y, dim=1)
        return pred_y

    @torch.no_grad()
    def vali(self):
        self.G.eval()
        preds_lst, trues_lst, total_loss = [], [], []
        val_pbar = tqdm(self.val_loader)
        for i, (batch_x, batch_y) in enumerate(val_pbar):
            if i * batch_x.shape[0] > 1000:
                break

            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            pred_y = self.G(batch_x)
            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                 pred_y, batch_y], [preds_lst, trues_lst]))

            loss = self.g_criterion(pred_y, batch_y)
            val_pbar.set_description(
                'vali loss: {:.4f}'.format(loss.mean().item()))
            total_loss.append(loss.mean().item())

        total_loss = np.average(total_loss)
        preds = np.concatenate(preds_lst, axis=0)
        trues = np.concatenate(trues_lst, axis=0)

        mse, mae, ssim, psnr = metric(preds, trues, self.val_loader.dataset.mean, self.val_loader.dataset.std, True)
        print('vali mse:{:.4f}, mae:{:.4f}, ssim:{:.4f}, psnr:{:.4f}'.format(mse, mae, ssim, psnr))
        self.G.train()
        return total_loss

    @torch.no_grad()
    def generate_samples(self, epoch):
        self.G.eval()

        # TODO: improve this one
        for batch_x, batch_y in self.test_loader:
            pred_y = self.G(batch_x.to(self.device))
            break

        batch_x = batch_x.detach().cpu().numpy()
        batch_y = batch_y.detach().cpu().numpy()
        pred_y = pred_y.detach().cpu().numpy()
        
        outputs_and_expectations = torch.cat((pred_y, batch_y), 0)

        if self.use_tensorboard is True:
            self.writer.add_image(f"inputs/Epoch_{epoch}", make_grid(batch_x.data, nrow=self.pre_seq_length), epoch)
            self.writer.add_image(f"outputs/Epoch_{epoch}", make_grid(pred_y.data, nrow=self.aft_seq_length), epoch)
            self.writer.add_image(f"expected/Epoch_{epoch}", make_grid(pred_y.data, nrow=self.aft_seq_length), epoch)
            save_image(batch_x.data, os.path.join(self.sample_path, epoch, "inputs.png"), nrow=self.pre_seq_length)
            save_image(outputs_and_expectations.data, os.path.join(self.sample_path, epoch, "outputs_and_expectations.png"), nrow=self.aft_seq_length)
        else:
            save_image(batch_x.data, os.path.join(self.sample_path, epoch, "inputs.png"), nrow=self.pre_seq_length)
            save_image(outputs_and_expectations.data, os.path.join(self.sample_path, epoch, "outputs_and_expectations.png"), nrow=self.aft_seq_length)
        self.G.train()
    
    def get_target_label(self, input, target_is_real):
        """Get target label.
        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.
        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
        """

        if self.adv_loss in ['wgan', 'wgan_softplus', 'wgan-gp']:
            return target_is_real
        target_val = (
            self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val
