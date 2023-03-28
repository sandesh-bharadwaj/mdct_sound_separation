from tqdm import tqdm

import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F
import wandb
import numpy as np
import random
from utils import augment
import os
from natsort import natsorted
from glob import glob
from utils import torch_mdct, torch_imdct
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
            target_masks = self.calculate_gt_masks_mixOfMix(sources, target_stems)
            mix_args = list(mixture.shape)
            mixture_spec = self.prepare_input_wav(mixture)

            with torch.cuda.amp.autocast():
                est_mask = self.model(mixture_spec)
                if self.args.use_weighted_loss:
                    mixture_spec_ = rearrange(mixture_spec, 'b f t c -> (b c) f t')
                    weight = torch.log1p(torch.abs(mixture_spec_))
                    weight = torch.clamp(weight, 1e-3, 10)
                else:
                    weight = torch.ones_like(mixture_spec)

                est_spec, est_wav = self.extract_output(mixture_spec, est_mask, mix_args)

                # Compute noise-invariant loss
                noiseInvLoss = torch.mean(
                    torch.stack(
                        [
                            F.binary_cross_entropy_with_logits(torch.sigmoid(est_mask), torch.sigmoid(target_masks[i]),
                                                               weight)
                            for i in range(self.args.num_mixtures)
                        ]
                    )
                )

            """ 
            METRICS 
            """
            metric_scores = [calc_metrics(target_stems[i], est_wav) for i in range(self.args.num_mixtures)]
            metrics['sdr_1'] = metric_scores[0]
            metrics['sdr_2'] = metric_scores[1]

            lr_cur = self.optimizer.state_dict()['param_groups'][0]['lr']
            scaler.scale(noiseInvLoss).backward()
            scaler.unscale_(self.optimizer)
            nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), max_norm=5.)

            scaler.step(self.optimizer)
            scaler.update()

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
                est_wav = rearrange(est_wav, 'b c t -> b t c').detach().cpu().numpy()[0]
                mix_wav = rearrange(mixture, 'b c t -> b t c').detach().cpu().numpy()[0]
                est_mask = est_mask.detach().cpu().numpy()[0]
                wandb.log({
                    'Train target wav 1': wandb.Audio(target_stems[0], sample_rate=self.args.sample_rate),
                    'Train target wav 2': wandb.Audio(target_stems[1], sample_rate=self.args.sample_rate),
                    'Train est wav': wandb.Audio(est_wav, sample_rate=self.args.sample_rate),
                    'Train mix wav': wandb.Audio(mix_wav, sample_rate=self.args.sample_rate)
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
                          'sdr_track:', scores)
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
        est_spec = mask * spec
        est_wav = torch_imdct(est_spec, sample_length=t, window_length=self.args.n_fft,
                              window_type='kbd')
        est_wav = rearrange(est_wav, '(b c) t -> b c t', b=b, c=c)
        est_wav = est_wav[:, :, :t]
        est_spec = rearrange(est_spec, '(b c) f t -> b c f t', c=c)
        return est_spec, est_wav

    def calculate_gt_masks_mixOfMix(self, sources, target_stems):
        target_masks = []
        for j in range(self.args.num_mixtures):
            mix_spec = rearrange(self.prepare_input_wav(sources[:, j]), 'b f t c -> (b c) f t')
            target_spec = rearrange(self.prepare_input_wav(target_stems[:, j]), 'b f t c -> (b c) f t')
            gt_mask = target_spec / mix_spec
            target_masks.append(gt_mask)
        target_masks = torch.stack(target_masks)
        return target_masks
