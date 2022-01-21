import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from models import losses
from models.misc_models import define_G, define_D, GANLoss
from models.helpers import compute_metrics


class APS(pl.LightningModule):
    def __init__(self, lr=0.0002, input_size:list =[288, 288], lambda_adv=1.0,
                 lambda_str=10.0, lambda_pix=100.0, num_patches=256, layers=[4, 7, 9]):
        super().__init__()
        self.save_hyperparameters()

        # Define generator (ResNet-based)
        self.net_G = define_G(input_nc=1, output_nc=1, ngf=64,
                                           netG="resnet_9blocks",
                                           norm="instance", use_dropout=False,
                                           init_type="xavier", init_gain=0.02,
                                           no_antialias=False,
                                           no_antialias_up=False)

        # Define discriminator (PatchGAN)
        self.net_D = define_D(input_nc=1, ndf=64, netD='basic',
                              n_layers_D=3, norm="instance",
                              init_type="xavier", init_gain=0.02,
                              no_antialias=False)

        # Define l_str func
        self.l_str_fn = StructuralConsistencyLoss(num_patches=num_patches,
                                               patch_size=64,
                                               input_size=input_size,
                                               layers=layers)

        # Define l_adv func
        self.l_adv_fn = GANLoss(gan_mode="lsgan")

        print(self.hparams)

    def forward(self, real_mri):
        return self.net_G(real_mri)

    def l_pix_fn(self, fake_ct, real_ct, p_patch, ori_hw):
        with torch.no_grad():
            p_pix = torch.nn.functional.interpolate(torch.relu(p_patch),
                                                    ori_hw, align_corners=False,
                                                    mode='bilinear')
            p_pix_min = p_pix.flatten(1).min(1)[0].view(-1, 1, 1, 1)  # min values for each item in the batch
            p_pix_max = p_pix.flatten(1).max(1)[0].view(-1, 1, 1, 1)  # max values for each item in the batch
            p_pix = (p_pix - p_pix_min) / (p_pix_max - p_pix_min)  # minmax normalize

        loss = nn.functional.l1_loss(fake_ct, real_ct, reduction='none') * p_pix
        return loss.mean()

    def _se_step(self, real_ct, real_mri):
        fake_ct = self(real_mri)
        real_ct_reversed = torch.flip(real_ct, [0])

        x = torch.cat([real_mri, real_mri], dim= 0)
        y_hat = torch.cat([fake_ct, real_ct], dim=0)
        y_tilda = torch.cat([real_ct_reversed, real_ct_reversed], dim=0)

        l_c = self.l_str_fn(x, y_hat, y_tilda)  # this func also computes l_c

        return l_c

    def _gen_step(self, real_ct, real_mri):
        fake_ct = self(real_mri)

        # L_adv loss
        p_patch = self.net_D(fake_ct)
        l_adv = self.l_adv_fn(p_patch, True).mean() * self.hparams.lambda_adv

        # L_pix loss
        l_pix = self.l_pix_fn(fake_ct, real_ct, p_patch, real_mri.shape[2:]) * self.hparams.lambda_pix

        # L_str loss
        l_str = self.l_str_fn(real_mri, fake_ct, None) * self.hparams.lambda_str

        return l_adv + l_pix + l_str

    def _disc_step(self, real_ct, real_mri):
        # Fake
        fake_ct = self(real_mri).detach()
        fake_logits = self.net_D(fake_ct)
        fake_loss = self.l_adv_fn(fake_logits, False).mean()

        # Real
        real_logits = self.net_D(real_ct)
        real_loss = self.l_adv_fn(real_logits, True).mean()

        l_disc = (fake_loss + real_loss) * 0.5

        return l_disc

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_ct, real_mri = batch['ct'], batch['in_phase']
        if optimizer_idx == 0: # train the structure encoder
            loss = self._se_step(real_ct, real_mri)
            self.log('train_SE_loss', loss, on_epoch=True)
        elif optimizer_idx == 1:  # train the discriminator
            loss = self._disc_step(real_ct, real_mri)
            self.log('train_D_loss', loss, on_epoch=True)
        elif optimizer_idx == 2:  # train the generator
            loss = self._gen_step(real_ct, real_mri)
            self.log('train_G_loss', loss, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        ct, mri = batch['ct'], batch['in_phase']
        pred = self(mri)

        with torch.no_grad():
            # Add dim to make it same as the ct shape so it can be broadcasted
            ct_min, ct_max = batch['ct_min'], batch['ct_max']
            ct_min = ct_min.view(-1, 1, 1)
            ct_max = ct_max.view(-1, 1, 1)

            mets = compute_metrics(pred.squeeze(1).cpu().numpy(),
                                   ct_min.cpu().numpy(),
                                   ct_max.cpu().numpy(),
                                   ct.squeeze(1).cpu().numpy())
        self.log('val_mae', mets['mae'], on_epoch=True)
        self.log('val_psnr', mets['psnr'], on_epoch=True)

        return {'mae': mets['mae']}

    def test_step(self, batch, batch_idx):
        ct, mri = batch['ct'], batch['in_phase']
        pred = self(mri)

        with torch.no_grad():
            # Add dim to make it same as the ct shape so it can be broadcasted
            ct_min, ct_max = batch['ct_min'], batch['ct_max']
            ct_min = ct_min.view(-1, 1, 1)
            ct_max = ct_max.view(-1, 1, 1)

            # Compute MAE, PSNR, and save visual results
            save_figurename = f"{self.trainer.default_root_dir}/visual/Result {batch_idx}.png"
            mets = compute_metrics(pred.squeeze(1).cpu().numpy(),
                                   ct_min.cpu().numpy(),
                                   ct_max.cpu().numpy(),
                                   ct.squeeze(1).cpu().numpy(),
                                   mri.squeeze(1).cpu().numpy(),
                                   create_figure=True,
                                   save_figurename=save_figurename)
        self.log('test_mae', mets['mae'], on_epoch=True)
        self.log('test_psnr', mets['psnr'], on_epoch=True)

        return {'mae': mets['mae']}

    def configure_optimizers(self):
        assert self.l_str_fn.LSeSim.conv_init
        lr = self.hparams.lr

        net_D_opt = torch.optim.Adam(self.net_D.parameters(), lr=lr)
        net_G_opt = torch.optim.Adam(self.net_G.parameters(), lr=lr)
        net_SE_opt = torch.optim.Adam(self.l_str_fn.parameters(), lr=lr)

        return net_SE_opt, net_D_opt, net_G_opt


class StructuralConsistencyLoss(nn.Module):
    def __init__(self, num_patches=256, patch_size=64, input_size=[288, 288],
                 layers=[4, 7, 9]):
        super().__init__()

        self.structure_encoder = losses.VGG16()

        # Re-use the code from LSeSim
        self.LSeSim = losses.SpatialCorrelativeLoss('cos', num_patches,
                                                    patch_size, True, True)

        # Run a dummy data to initialize the 1x1 convolution operations
        self.layers = layers  # layer id to extract features from
        dummy_fea = torch.randn([1, 1] + input_size)
        self(dummy_fea, dummy_fea, None)

    def forward(self, src, tgt, other=None):
        n_layers = len(self.layers)
        feats_src = self.structure_encoder(src, self.layers, encode_only=True)
        feats_tgt = self.structure_encoder(tgt.float(), self.layers, encode_only=True)
        if other is not None:
            feats_oth = self.structure_encoder(
                torch.flip(other.float(), [2, 3]), self.layers,
                encode_only=True)
        else:
            feats_oth = [None for _ in range(n_layers)]

        total_loss = 0.0
        for i, (feat_src, feat_tgt, feat_oth) in enumerate(zip(feats_src, feats_tgt, feats_oth)):
            loss = self.LSeSim.loss(feat_src, feat_tgt, feat_oth, i)
            total_loss += loss.mean()

        if not self.LSeSim.conv_init:
            self.LSeSim.update_init_()

        return total_loss / n_layers