import torch
import torch.nn as nn
import numpy as np
from einops.layers.torch import Rearrange
from einops import rearrange

'''
This is a modified implementation of the paper Music Source Separation with Band-Split RNN
Instead of using STFT on the audio waveform, we instead apply the Modified DCT Transform (DCT Type-IV) as 
an input transform and replace the standard L1 loss with noise-invariant loss introduced in CLIPSep.
'''

class BSRNN(nn.Module):
    def __init__(self, target_stem, sample_rate, n_fft, hop_length, channels=2,
                 fc_dim=128, num_band_seq_module=12, model_type="vocals",
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

        if model_type == "drums":
            pass
        elif model_type == "bass":
            pass
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

    def forward(self, wav):
        b, c, t = wav.shape
        wav = rearrange(wav, 'b c t -> (b c) t')
        spec = torch_mdct(wav, window_length=self.n_fft, window_type='kbd')
        # spec = torch.stft(wav, n_fft=self.n_fft, hop_length=self.hop_length,
        #                   window=torch.hann_window(self.n_fft).to(wav.device), return_complex=False)
        spec_ = rearrange(spec, '(b c) f t -> b f t c', c=self.channels)
        z = self.bandSplitter(spec_)
        q = self.bandSeqModeling(z)
        mask = self.maskEstimator(q)

        # cspec = torch.view_as_complex(spec)
        # cmask = torch.view_as_complex(mask)
        # est_cspec = cmask * cspec
        # est_wav = torch.istft(est_cspec, n_fft=self.n_fft, hop_length=self.hop_length,
        #                       window=torch.hann_window(self.n_fft).to(wav.device))

        #MDCT change
        est_cspec = mask * spec
        est_wav = torch_imdct(est_cspec, sample_length=6 * self.sample_rate, window_length=self.n_fft,
                              window_type='kbd')
        # est_spec = torch.view_as_real(est_cspec)
        est_spec = est_cspec
        est_wav = rearrange(est_wav, '(b c) t -> b c t', b=b, c=c)
        est_spec = rearrange(est_spec, '(b c) f t -> b c f t', c=self.channels)
        return est_spec, est_wav, torch.abs(mask)


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
                 bands=[1000, 4000, 8000, 16000, 20000], num_subBands=[10, 12, 8, 8, 2, 1]):
        super().__init__()
        self.bands = createFreqBands(sample_rate, n_fft, bands, num_subBands)
        self.band_intervals = self.bands[1:] - self.bands[:-1]
        num_total_subBands = len(self.bands) - 1
        self.channels = channels
        fc_dim = fc_dim * channels
        hidden_dim = 4 * fc_dim

        self.layer_list = nn.ModuleList([
            nn.Sequential(
                Rearrange('b n t -> b t n'),
                nn.LayerNorm(fc_dim),
                nn.Linear(fc_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, band_interval * channels)
            ) for band_interval in self.band_intervals
        ])

    def forward(self, q):
        # print(q.shape)
        outputs = []
        for i in range(len(self.band_intervals)):
            output = self.layer_list[i](q[:, :, i, :])
            output = rearrange(output, 'b t (f c) -> (b c) f t', c=self.channels)
            outputs.append(output)
        mask = torch.cat(outputs, dim=-2)
        return mask


# Class to extract LSTM output separately
class ExtractLSTMOutput(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def forward(x):
        output, _ = x
        return output


def createFreqBands(sample_rate, n_fft, bands, num_subBands):
    bands = [0] + bands + [int(sample_rate / 2)]
    bands = np.array(bands) * n_fft / sample_rate

    freq_bands = []
    for i in range(len(num_subBands)):
        start_freq = int(bands[i])
        end_freq = int(bands[i + 1])
        num_bands = num_subBands[i]
        interval = (end_freq - start_freq) / num_bands
        for n in range(num_bands):
            freq_bands.append(int(start_freq + interval * n))
    freq_bands.append(int(n_fft-1)//2)
    return np.array(freq_bands)


class mdctFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_audio, window_length, window_type="kbd"):
        if window_type == "kbd":
            alpha = 5
            kbd_win_func = np.kaiser(int(window_length / 2) + 1, alpha * torch.pi)
            kbd_cumSum_win_func = np.cumsum(kbd_win_func[1:int(window_length / 2)])
            win_func = np.sqrt(np.concatenate((kbd_cumSum_win_func, np.flip(kbd_cumSum_win_func, [0])), 0)
                               / np.sum(kbd_win_func))
            print("KBD Window:", window_length)
        # Window function used in Vorbis audio coding
        elif window_type == "vorbis":
            win_func = np.sin(np.pi / 2 * np.pow(np.sin(np.pi / window_length *
                                                        np.arange(0.5, window_length + 0.5)), 2))
        input_audio_np = input_audio.cpu().detach().numpy()
        result = []
        if input_audio_np.shape[0] > 1:
            for i in range(input_audio_np.shape[0]):
                # print("batch: "+str(i//2)+" channel: "+str(i%2))
                result.append(mdct(input_audio_np[i], win_func))
        else:
            result.append(mdct(input_audio_np, win_func))
        return input_audio.new(np.stack(result))


class imdctFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_mdct, sample_length, window_length, window_type="kbd"):
        if window_type == "kbd":
            alpha = 5
            kbd_win_func = np.kaiser(int(window_length / 2) + 1, alpha * torch.pi)
            kbd_cumSum_win_func = np.cumsum(kbd_win_func[1:int(window_length / 2)])
            win_func = np.sqrt(np.concatenate((kbd_cumSum_win_func, np.flip(kbd_cumSum_win_func, [0])), 0)
                               / np.sum(kbd_win_func))
        # Window function used in Vorbis audio coding
        elif window_type == "vorbis":
            win_func = np.sin(np.pi / 2 * np.pow(np.sin(np.pi / window_length *
                                                        np.arange(0.5, window_length + 0.5)), 2))

        input_spec = input_mdct.cpu().detach().numpy()
        result = []
        for i in range(input_spec.shape[0]):
            result.append(imdct(input_spec[i], window_function=win_func))
        return input_mdct.new(np.stack(result))


def torch_mdct(input_audio, window_length, window_type="kbd"):
    return mdctFunc.apply(input_audio, window_length, window_type)


def torch_imdct(input_mdct, sample_length, window_length, window_type='kbd'):
    return imdctFunc.apply(input_mdct, sample_length, window_length, window_type)


# From Zaf-Python
def mdct(audio_signal, window_function):
    """
    Compute the modified discrete cosine transform (MDCT) using the fast Fourier transform (FFT).
    Inputs:
        audio_signal: audio signal (number_samples,)
        window_function: window function (window_length,)
    Output:
        audio_mdct: audio MDCT (number_frequencies, number_times)
    Example: Compute and display the MDCT as used in the AC-3 audio coding format.
        # Import the needed modules
        import numpy as np
        import zaf
        import matplotlib.pyplot as plt
        # Read the audio signal (normalized) with its sampling frequency in Hz, and average it over its channels
        audio_signal, sampling_frequency = zaf.wavread("audio_file.wav")
        audio_signal = np.mean(audio_signal, 1)
        # Compute the Kaiser-Bessel-derived (KBD) window as used in the AC-3 audio coding format
        window_length = 512
        alpha_value = 5
        window_function = np.kaiser(int(window_length/2)+1, alpha_value*np.pi)
        window_function2 = np.cumsum(window_function[1:int(window_length/2)])
        window_function = np.sqrt(np.concatenate((window_function2, window_function2[int(window_length/2)::-1]))
                                /np.sum(window_function))
        # Compute the MDCT
        audio_mdct = zaf.mdct(audio_signal, window_function)
        # Display the MDCT in dB, seconds, and Hz
        number_samples = len(audio_signal)
        plt.figure(figsize=(14, 7))
        zaf.specshow(np.absolute(audio_mdct), number_samples, sampling_frequency, xtick_step=1, ytick_step=1000)
        plt.title("MDCT (dB)")
        plt.tight_layout()
        plt.show()
    """

    # Get the number of samples and the window length in samples
    number_samples = len(audio_signal)
    window_length = len(window_function)

    # Derive the step length and the number of frequencies (for clarity)
    step_length = int(window_length / 2)
    number_frequencies = int(window_length / 2)

    # Derive the number of time frames
    number_times = int(np.ceil(number_samples / step_length)) + 1

    # Zero-pad the start and the end of the signal to center the windows
    audio_signal = np.pad(
        audio_signal,
        (step_length, (number_times + 1) * step_length - number_samples),
        "constant",
        constant_values=0,
    )

    # Initialize the MDCT
    audio_mdct = np.zeros((number_frequencies, number_times))

    # Prepare the pre-processing and post-processing arrays
    preprocessing_array = np.exp(
        -1j * np.pi / window_length * np.arange(0, window_length)
    )
    postprocessing_array = np.exp(
        -1j
        * np.pi
        / window_length
        * (window_length / 2 + 1)
        * np.arange(0.5, window_length / 2 + 0.5)
    )

    # Loop over the time frames
    # (Do the pre and post-processing, and take the FFT in the loop to avoid storing twice longer frames)
    i = 0
    # print("num_times:", number_times)
    for j in range(number_times):
        # Window the signal
        # print(audio_signal[i:i+window_length].shape)
        # print(window_function.shape)
        audio_segment = audio_signal[i: i + window_length] * window_function
        i = i + step_length

        # Compute the Fourier transform of the windowed segment using the FFT after pre-processing
        audio_segment = np.fft.fft(audio_segment * preprocessing_array)

        # Truncate to the first half before post-processing (and take the real to ensure real values)
        audio_mdct[:, j] = np.real(
            audio_segment[0:number_frequencies] * postprocessing_array
        )

    return audio_mdct


# From Zaf-Python
def imdct(audio_mdct, window_function):
    """
    Compute the inverse modified discrete cosine transform (MDCT) using the fast Fourier transform (FFT).

    Inputs:
        audio_mdct: audio MDCT (number_frequencies, number_times)
        window_function: window function (window_length,)
    Output:
        audio_signal: audio signal (number_samples,)

    Example: Verify that the MDCT is perfectly invertible.
        # Import the needed modules
        import numpy as np
        import zaf
        import matplotlib.pyplot as plt

        # Read the audio signal (normalized) with its sampling frequency in Hz, and average it over its channels
        audio_signal, sampling_frequency = zaf.wavread("audio_file.wav")
        audio_signal = np.mean(audio_signal, 1)

        # Compute the MDCT with a slope function as used in the Vorbis audio coding format
        window_length = 2048
        window_function = np.sin(np.pi/2*pow(np.sin(np.pi/window_length*np.arange(0.5, window_length+0.5)), 2))
        audio_mdct = zaf.mdct(audio_signal, window_function)

        # Compute the inverse MDCT
        audio_signal2 = zaf.imdct(audio_mdct, window_function)
        audio_signal2 = audio_signal2[0:len(audio_signal)]

        # Compute the differences between the original signal and the resynthesized one
        audio_differences = audio_signal-audio_signal2
        y_max = np.max(np.absolute(audio_differences))

        # Display the original and resynthesized signals, and their differences in seconds
        xtick_step = 1
        plt.figure(figsize=(14, 7))
        plt.subplot(3, 1, 1), zaf.sigplot(audio_signal, sampling_frequency, xtick_step)
        plt.ylim(-1, 1), plt.title("Original signal")
        plt.subplot(3, 1, 2), zaf.sigplot(audio_signal2, sampling_frequency, xtick_step)
        plt.ylim(-1, 1), plt.title("Resyntesized signal")
        plt.subplot(3, 1, 3), zaf.sigplot(audio_differences, sampling_frequency, xtick_step)
        plt.ylim(-y_max, y_max), plt.title("Original - resyntesized signal")
        plt.tight_layout()
        plt.show()
    """

    # Get the number of frequency channels and time frames
    number_frequencies, number_times = np.shape(audio_mdct)

    # Derive the window length and the step length in samples (for clarity)
    window_length = 2 * number_frequencies
    step_length = number_frequencies

    # Derive the number of samples for the signal
    number_samples = step_length * (number_times + 1)

    # Initialize the audio signal
    audio_signal = np.zeros(number_samples)

    # Prepare the pre-processing and post-processing arrays
    preprocessing_array = np.exp(
        -1j
        * np.pi
        / (2 * number_frequencies)
        * (number_frequencies + 1)
        * np.arange(0, number_frequencies)
    )
    postprocessing_array = (
            np.exp(
                -1j
                * np.pi
                / (2 * number_frequencies)
                * np.arange(
                    0.5 + number_frequencies / 2,
                    2 * number_frequencies + number_frequencies / 2 + 0.5,
                )
            )
            / number_frequencies
    )

    # Compute the Fourier transform of the frames using the FFT after pre-processing (zero-pad to get twice the length)
    audio_mdct = np.fft.fft(
        audio_mdct * preprocessing_array[:, np.newaxis],
        n=2 * number_frequencies,
        axis=0,
    )

    # Apply the window function to the frames after post-processing (take the real to ensure real values)
    audio_mdct = 2 * (
            np.real(audio_mdct * postprocessing_array[:, np.newaxis])
            * window_function[:, np.newaxis]
    )

    # Loop over the time frames
    i = 0
    for j in range(number_times):
        # Recover the signal with the time-domain aliasing cancellation (TDAC) principle
        audio_signal[i: i + window_length] = (
                audio_signal[i: i + window_length] + audio_mdct[:, j]
        )
        i = i + step_length

    # Remove the zero-padding at the start and at the end of the signal
    audio_signal = audio_signal[step_length: -step_length - 1]

    return audio_signal


if __name__ == '__main__':
    from torchinfo import summary

    sec = 6
    nfft = 2048
    hop = 512
    sr = 44100
    lr = None
    bs = None

    bsrnn = BSRNN('vocals', sr, nfft, hop) # , num_band_seq_module=3
    wav_len = sr * sec
    wav_len = int(wav_len / hop) * hop
    summary(bsrnn, input_size=(2, 2, wav_len))
