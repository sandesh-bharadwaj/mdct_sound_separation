from tqdm import tqdm

import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F
import wandb
import numpy as np
import random
import os
from natsort import natsorted
from glob import glob
from utils import torch_mdct, torch_imdct, calc_loss
from metrics import calc_metrics


class Solver:
    def __init__(self,
                 args,
                 model,
                 device,
                 optimizer,
                 train_loader,
                 valid_loader=None,
                 test_loader=None,
                 scheduler=None,
                 augmentation=True
                 ):
        stem_list = ['mix', 'drums', 'bass', 'other', 'vocals']
        stem_dict = {stem: i for i, stem in enumerate(stem_list)}
        self.target_stem_idx = stem_dict[args.target_stem]

        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        if self.device == torch.device('cuda:0'):
            self.model = nn.DataParallel(self.model)

        # if augmentation:
        #     self.augments = augment(args).to(device)
        # else:
        #     self.augments = nn.Identity.to(device)

        self.model.to(device)

        self.l1Loss = nn.L1Loss(reduction='mean')

    def train(self):
        start_epoch = self.args.start_epoch + 1 if self.args.resume else 0
        for epoch in range(start_epoch, self.args.max_epochs):
            self.train_epoch(epoch=epoch)
            if epoch % self.args.valid_per == 0:
                if self.valid_loader is not None:
                    self.valid_epoch(epoch=epoch)
            if epoch % self.args.ckpt_per == 0:
                try:
                    state_dict = self.model.module.state_dict()
                except AttributeError:
                    state_dict = self.model.state_dict()
                ckpt = dict(net=state_dict, opt=self.optimizer.state_dict())
                torch.save(ckpt, os.path.join(self.args.save_dir, 'ckpt', str(epoch).zfill(4) + '.pt'))
                ckpt_list = natsorted(glob(os.path.join(self.args.save_dir, 'ckpt', '*')))
                if len(ckpt_list) > 4:
                    os.remove(ckpt_list[0])

    def train_epoch(self, epoch):
        self.model.train()
        scaler = torch.cuda.amp.GradScaler()
        metrics = dict(sdr_1=[], sdr_2=[])
        for idx, batch in enumerate(tqdm(self.train_loader, desc=f'Train Epoch: {epoch}')):
            self.optimizer.zero_grad()

            mixture, sources, target_stems = [i.to(self.device) for i in batch]
            gt_masks = self.calculate_gt_masks_mixOfMix(sources, target_stems)
            # print("GTMASKS:", gt_masks.shape)
            # print("Target stems:",target_stems.shape)
            mix_args = list(mixture.shape)
            mixture_spec = self.prepare_input_wav(mixture)

            with torch.cuda.amp.autocast():
                pred_mask = self.model(mixture_spec)
                if self.args.use_weighted_loss:
                    mixture_spec_ = rearrange(mixture_spec, 'b f t c -> (b c) f t')
                    weight = torch.log1p(torch.abs(mixture_spec_))
                    weight = torch.clamp(weight, 1e-3, 10)
                else:
                    weight = torch.ones_like(mixture_spec)

                est_specs, est_wavs = self.extract_output(mixture_spec, pred_mask[:2], mix_args)
                # print(est_wavs.shape,est_specs.shape)

                # Compute noise-invariant loss
                loss_00_0 = calc_loss(pred_mask[0], pred_mask[2], gt_masks[0], weight)
                loss_11_1 = calc_loss(pred_mask[1], pred_mask[3], gt_masks[1], weight)
                loss_01_0 = calc_loss(pred_mask[0], pred_mask[3], gt_masks[0], weight)
                loss_10_1 = calc_loss(pred_mask[1], pred_mask[2], gt_masks[1], weight)
                loss_0 = (loss_00_0 + loss_11_1) / 2
                loss_1 = (loss_01_0 + loss_10_1) / 2
                noiseInvLoss = torch.mean(torch.minimum(loss_0, loss_1))

                # noiseInvLoss = torch.mean(
                #     torch.stack(
                #         [
                #             F.binary_cross_entropy_with_logits(torch.sigmoid(est_mask), torch.sigmoid(gt_masks[i]),
                #                                                weight)
                #             for i in range(self.args.num_mixtures)
                #         ]
                #     )
                # )

            """ 
            METRICS 
            """
            metric_scores = [calc_metrics(target_stems[:, i], est_wavs[i]) for i in range(self.args.num_mixtures)]
            metrics['sdr_1'] = metric_scores[0]
            metrics['sdr_2'] = metric_scores[1]

            lr_cur = self.optimizer.state_dict()['param_groups'][0]['lr']
            scaler.scale(noiseInvLoss).backward()
            scaler.unscale_(self.optimizer)
            nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), max_norm=5.)

            scaler.step(self.optimizer)
            scaler.update()

            # print(metrics['sdr_1'], type(metrics['sdr_1']), metrics['sdr_1'].shape)

            """
            WANDB LOGGING
            """
            if not self.args.debug:
                wandb.log({
                    'epoch': epoch,
                    'learning_rate': lr_cur,
                    'train loss': noiseInvLoss.item(),
                    'SDR 1': metrics['sdr_1'],
                    'SDR 2': metrics['sdr_2'],
                })

            if self.args.debug:
                print('loss:', noiseInvLoss.item(), 'cur_lr:', lr_cur, 'epoch:', epoch, 'sdr1:',
                      metrics['sdr_1'], 'sdr2:', metrics['sdr_2'])
                if idx > 2:
                    break

        if self.scheduler is not None:
            self.scheduler.step()

        if epoch % self.args.wandb_per == 0:
            if not self.args.debug:
                target_stems = [rearrange(target_stems[:, i], 'b c t -> b t c').detach().cpu().numpy()[0]
                                for i in range(self.args.num_mixtures)]
                est_wavs = rearrange(est_wavs, 'n b c t -> n b t c').detach().cpu().numpy()[:,0]

                mixture = rearrange(mixture, 'b c t -> b t c').detach().cpu().numpy()[0]
                wandb.log({
                    'Train target wav 1': wandb.Audio(target_stems[0], sample_rate=self.args.sample_rate),
                    'Train target wav 2': wandb.Audio(target_stems[1], sample_rate=self.args.sample_rate),
                    'Train est wav 1': wandb.Audio(est_wavs[0], sample_rate=self.args.sample_rate),
                    'Train est wav 2': wandb.Audio(est_wavs[1], sample_rate=self.args.sample_rate),
                    'Train mix wav': wandb.Audio(mixture, sample_rate=self.args.sample_rate)
                })

    def valid_epoch(self, epoch):
        self.model.eval()
        if not self.args.debug:
            if epoch == 0:
                return
        with torch.no_grad():
            metrics = dict(sdr_track=[])
            for idx, batch in enumerate(tqdm(self.valid_loader, desc=f"Valid Epoch: {epoch}")):
                mixture, sources, target_stems = [i.to(self.device) for i in batch]
                mix_args = list(mixture.shape)
                mixture = self.prepare_input_wav(mixture)
                est_mask = self.model(mixture)
                est_wav = self.extract_output(mixture, est_mask, mix_args)
                target_stems = target_stems[0]
                """
                Validation Metrics
                """
                scores = calc_metrics(target_stems, est_wav)
                metrics['sdr_track'] += scores

                if not self.args.debug:
                    pass

                if self.args.debug:
                    print("VALIDATION")
                    print('epoch:', epoch,
                          'validation_sdr:', scores)
                    if idx > 5:
                        break

            if not self.args.debug:
                wandb.log({
                    'valid metric sdr epoch': np.mean(metrics['sdr_track']).item()
                })

    def prepare_input_wav(self, wav):
        wav = rearrange(wav, 'b c t -> (b c) t')
        spec = torch_mdct(wav, window_length=self.args.n_fft, window_type='kbd')
        spec = rearrange(spec, '(b c) f t -> b f t c', c=self.args.channels)
        return spec

    def extract_output(self, spec, mask, mix_args):
        b, c, t = mix_args
        spec = rearrange(spec, 'b f t c -> (b c) f t', b=b, c=c)
        est_specs = []
        est_wavs = []
        for i in range(mask.shape[0]):
            est_spec = mask[i] * spec
            est_wav = torch_imdct(est_spec, sample_length=t, window_length=self.args.n_fft,
                                  window_type='kbd')
            est_wav = rearrange(est_wav, '(b c) t -> b c t', b=b, c=c)
            est_wav = est_wav[:, :, :t]
            est_spec = rearrange(est_spec, '(b c) f t -> b c f t', c=c)
            est_specs.append(est_spec)
            est_wavs.append(est_wav)
        est_specs = torch.stack(est_specs)
        est_wavs = torch.stack(est_wavs)
        return est_specs, est_wavs

    def calculate_gt_masks_mixOfMix(self, sources, target_stems):
        gt_masks = []
        for j in range(self.args.num_mixtures):
            mix_spec = rearrange(self.prepare_input_wav(sources[:, j]), 'b f t c -> (b c) f t')
            target_spec = rearrange(self.prepare_input_wav(target_stems[:, j]), 'b f t c -> (b c) f t')
            gt_mask = target_spec / mix_spec
            gt_masks.append(gt_mask)
        gt_masks = torch.stack(gt_masks)
        return gt_masks
