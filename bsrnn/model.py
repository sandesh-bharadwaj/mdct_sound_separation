import torch
import torch.nn as nn
import numpy as np
from einops.layers.torch import Rearrange
from einops import rearrange
import lightning.pytorch as pylightning
from utils import createFreqBands, torch_mdct, torch_imdct

'''
This is a modified implementation of the paper Music Source Separation with Band-Split RNN
Instead of using STFT on the audio waveform, we instead apply the Modified DCT Transform or MDCT 
(DCT Type-IV with windowing) as an input transform and replace the standard L1 loss with noise-invariant 
loss introduced in CLIPSep.
'''


# PyTorch Lightning Trainer
class BSRNNLightning(pylightning.LightningModule):
    def __init__(self, target_stem, sample_rate, n_fft, hop_length, lr, batch_size, channels=2,
                 fc_dim=128, num_band_seq_module=12,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        stem_list = ['mix', 'drums', 'bass', 'other', 'vocals']
        stem_dict = {stem: i for i, stem in enumerate(stem_list)}
        self.target_stem_idx = stem_dict[target_stem]

        # FFT params
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.channels = channels

        if target_stem == "drums":
            bands = [1000, 2000, 4000, 8000, 16000]
            num_subBands = [20, 10, 8, 8, 8, 1]
        elif target_stem == "bass":
            bands = [500, 1000, 4000, 8000, 16000]
            num_subBands = [10, 5, 6, 4, 4, 1]
        else:
            # V7 band splitting
            bands = [1000, 4000, 8000, 16000, 20000]
            num_subBands = [10, 12, 8, 8, 2, 1]

        self.bandSplitter = BandSplitModule(sample_rate=sample_rate, n_fft=n_fft, channels=self.channels,
                                            fc_dim=fc_dim, bands=bands, num_subBands=num_subBands)
        self.bandSeqModeling = nn.Sequential(
            *[BandSequenceModelingModule(channels=channels, fc_dim=fc_dim,
                                         num_subBands=num_subBands)
              for _ in range(num_band_seq_module)]
        )
        self.maskEstimator = MaskEstimationModule(sample_rate=sample_rate, n_fft=n_fft, channels=channels,
                                                  fc_dim=fc_dim,
                                                  bands=bands, num_subBands=num_subBands)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = 0
        return loss

    def configure_optimizers(self, lr=1e-3):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer


class BSRNN(nn.Module):
    def __init__(self, target_stem, sample_rate, n_fft, hop_length, channels=2,
                 fc_dim=128, num_band_seq_module=12, num_mixtures=2,
                 **kwargs):
        super().__init__()
        stem_list = ['mix', 'drums', 'bass', 'other', 'vocals']
        stem_dict = {stem: i for i, stem in enumerate(stem_list)}
        self.target_stem_idx = stem_dict[target_stem]

        # FFT params
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.channels = channels
        self.num_mixtures = num_mixtures

        if target_stem == "drums":
            bands = [1000, 2000, 4000, 8000, 16000]
            num_subBands = [20, 10, 8, 8, 8, 1]
        elif target_stem == "bass":
            bands = [500, 1000, 4000, 8000, 16000]
            num_subBands = [10, 5, 6, 4, 4, 1]
        else:
            # V7 band splitting
            bands = [1000, 4000, 8000, 16000, 20000]
            num_subBands = [10, 12, 8, 8, 2, 1]

        self.bandSplitter = BandSplitModule(sample_rate=sample_rate, n_fft=n_fft, channels=self.channels,
                                            fc_dim=fc_dim, bands=bands, num_subBands=num_subBands)
        self.bandSeqModeling = nn.Sequential(
            *[BandSequenceModelingModule(channels=channels, fc_dim=fc_dim,
                                         num_subBands=num_subBands)
              for _ in range(num_band_seq_module)]
        )
        self.maskEstimator = MaskEstimationModule(sample_rate=sample_rate, n_fft=n_fft, channels=channels,
                                                  fc_dim=fc_dim,
                                                  bands=bands, num_subBands=num_subBands, num_mixtures=num_mixtures)

    def forward(self, spec_):
        # b, c, t = wav.shape
        # wav = rearrange(wav, 'b c t -> (b c) t')
        # spec = torch_mdct(wav, window_length=self.n_fft, window_type='kbd')
        # spec_ = rearrange(spec, '(b c) f t -> b f t c', c=self.channels)
        out = self.bandSplitter(spec_)
        out = self.bandSeqModeling(out)
        out = self.maskEstimator(out)

        # est_spec = mask * spec
        #
        # est_wav = torch_imdct(est_spec, sample_length=t, window_length=self.n_fft,
        #                       window_type='kbd')
        # est_wav = rearrange(est_wav, '(b c) t -> b c t', b=b, c=c)
        # est_wav = est_wav[:, :, :t]
        # est_spec = rearrange(est_spec, '(b c) f t -> b c f t', c=self.channels)
        return out


class BandSplitModule(nn.Module):
    def __init__(self, sample_rate, n_fft, channels=2,
                 fc_dim=128, bands=[1000, 4000, 8000, 16000, 20000],
                 num_subBands=[10, 12, 8, 8, 2, 1]):
        super().__init__()

        self.bands = createFreqBands(sample_rate, n_fft, bands, num_subBands)
        self.band_intervals = self.bands[1:] - self.bands[:-1]
        self.channels = channels

        self.layer_list = nn.ModuleList([
            nn.Sequential(
                Rearrange('b f t c -> b t (f c)', c=self.channels),
                nn.LayerNorm(band_interval * channels),
                nn.Linear(channels * band_interval, channels * fc_dim),
                Rearrange('b t n -> b n 1 t')
            )
            for band_interval in self.band_intervals
        ])

    def forward(self, spec):
        # spec format: (b, f, t, channel) #Mono or stereo channels
        spec_bands = [spec[:, self.bands[i]:self.bands[i + 1]] for i in range(len(self.bands) - 1)]
        outputs = []
        for spec_band, layer in zip(spec_bands, self.layer_list):
            output = layer(spec_band)
            outputs.append(output)
        outputs = torch.cat(outputs, dim=-2)
        return outputs


class BandSequenceModelingModule(nn.Module):
    def __init__(self, fc_dim, channels=2, num_subBands=[10, 12, 8, 8, 2, 1]):
        super().__init__()
        fc_dim = channels * fc_dim
        hidden_dim = fc_dim  # Dim for BLSTM
        num_totalSubBands = sum(num_subBands)

        # RNN across T
        self.blstm_seq = nn.Sequential(
            Rearrange('b n k t -> b k n t'),
            nn.GroupNorm(num_groups=num_totalSubBands, num_channels=num_totalSubBands),
            Rearrange('b k n t -> (b k) t n'),
            nn.LSTM(input_size=fc_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True),
            ExtractLSTMOutput(),
            nn.Linear(2 * hidden_dim, fc_dim),
            Rearrange('(b k) t n -> b n k t', k=num_totalSubBands)
        )

        # RNN across K
        self.blstm_band = nn.Sequential(
            Rearrange('b n k t -> b k n t'),
            nn.GroupNorm(num_groups=num_totalSubBands, num_channels=num_totalSubBands),
            Rearrange('b k n t -> (b t) k n'),
            nn.LSTM(input_size=fc_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True),
            ExtractLSTMOutput(),
            nn.Linear(2 * hidden_dim, fc_dim)
        )

    def forward(self, z):
        z_cap = self.blstm_seq(z) + z
        out = self.blstm_band(z_cap)
        out = rearrange(out, '(b t) k n -> b n k t', b=z_cap.size(0))
        out = out + z_cap
        return out


class MaskEstimationModule(nn.Module):
    def __init__(self, sample_rate, n_fft, channels=2, fc_dim=128,
                 bands=[1000, 4000, 8000, 16000, 20000], num_subBands=[10, 12, 8, 8, 2, 1], num_mixtures=2):
        super().__init__()
        self.bands = createFreqBands(sample_rate, n_fft, bands, num_subBands)
        self.band_intervals = self.bands[1:] - self.bands[:-1]
        self.channels = channels
        self.num_mixtures = num_mixtures
        fc_dim = fc_dim * channels
        hidden_dim = self.num_mixtures * 4 * fc_dim

        self.layer_list = nn.ModuleList([
            nn.Sequential(
                Rearrange('b n t -> b t n'),
                nn.LayerNorm(fc_dim),
                nn.Linear(fc_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 2 * num_mixtures * band_interval * channels)
            ) for band_interval in self.band_intervals
        ])

    def forward(self, q):
        outputs = []
        for i in range(len(self.band_intervals)):
            output = self.layer_list[i](q[:, :, i, :])
            output = rearrange(output, 'b t (f c) -> (b c) f t', c=2*self.num_mixtures*self.channels)
            outputs.append(output)
        outputs = torch.cat(outputs, dim=-2)
        outputs = rearrange(outputs, '(b c n) f t -> n (b c) f t', c=self.channels, n=2*self.num_mixtures)
        return outputs


# Class to extract LSTM output separately
class ExtractLSTMOutput(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def forward(x):
        output, _ = x
        return output


if __name__ == '__main__':
    from torchinfo import summary

    sec = 6
    nfft = 2048
    hop = 512
    sr = 44100
    lr = None
    bs = None

    bsrnn = BSRNN('vocals', sr, nfft, hop)  # , num_band_seq_module=3
    wav_len = sr * sec
    wav_len = int(wav_len / hop) * hop
    wav = torch.rand((1, 2, wav_len))
    wav = rearrange(wav, 'b c t -> (b c) t')
    spec = torch_mdct(wav, window_length=nfft, window_type='kbd')
    print(spec.shape)
    spec = rearrange(spec, '(b c) f t -> b f t c', c=2)

    summary(bsrnn, input_size=spec.shape)
