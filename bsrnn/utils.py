import numpy as np
import torch
import torch_dct
import torch.nn as nn


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
    freq_bands.append(int(n_fft - 1) // 2)
    return np.array(freq_bands)

"""
MDCT and IMDCT Functions
"""

class mdctFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_audio, window_length, window_type="kbd"):
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
        input_audio_np = input_audio.cpu().detach().numpy()
        result = []
        if input_audio_np.shape[0] > 1:
            for i in range(input_audio_np.shape[0]):
                result.append(mdct(input_audio_np[i], win_func))
        else:
            result.append(mdct(input_audio_np, win_func))
        return input_audio.new(np.stack(result))


class imdctFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_mdct, sample_length, window_length, window_type="kbd"):
        # Kaiser–Bessel-derived (KBD) window
        if window_type == "kbd":
            alpha = 5
            kbd_win_func = np.kaiser(int(window_length / 2) + 1, alpha * torch.pi)
            kbd_cumSum_win_func = np.cumsum(kbd_win_func[1:int(window_length / 2)])
            win_func = np.sqrt(np.concatenate((kbd_cumSum_win_func, np.flip(kbd_cumSum_win_func, [0])), 0)
                               / np.sum(kbd_win_func))
        # Vorbis window
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
    for j in range(number_times):
        # Window the signal
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

"""
STDCT/ISTDCT Functions
"""

# Torch implementation of Short-Time DCT, requires torch_dct library
def torch_stdct(signals, frame_length, frame_step, window=torch.hamming_window):
    """Compute Short-Time Discrete Cosine Transform of `signals`.
    No padding is applied to the signals.
    Parameters
    ----------
    signal : Time-domain input signal(s), a `[..., n_samples]` tensor.
    frame_length : Window length and DCT frame length in samples.
    frame_step : Number of samples between adjacent DCT columns.
    window : Window to use for DCT.  Either a window tensor (see documentation for `torch.stft`),
        or a window tensor constructor, `window(frame_length) -> Tensor`.
        Default: hamming window.
    Returns
    -------
    dct : Real-valued F-T domain DCT matrix/matrixes, a `[..., frame_length, n_frames]` tensor.
    """
    framed = signals.unfold(-1, frame_length, frame_step)
    if callable(window):
        window = window(frame_length).to(framed)
    if window is not None:
        framed = framed * window
    return torch_dct.dct(framed, norm="ortho").transpose(-1, -2)


# Torch implementation of Short-Time DCT, requires torch_dct library
def torch_istdct(dcts, *, frame_step, frame_length=None, window=torch.hamming_window):
    """Compute Inverse Short-Time Discrete Cosine Transform of `dct`.
    Parameters other than `dcts` are keyword-only.
    Parameters
    ----------
    dcts : DCT matrix/matrices from `torch_stdct`
    frame_step : Number of samples between adjacent DCT columns (should be the
        same value that was passed to `torch_stdct`).
    frame_length : Ignored.  Window length and DCT frame length in samples.
        Can be None (default) or same value as passed to `torch_stdct`.
    window : Window to use for DCT.  Either a window tensor (see documentation for `torch.stft`),
        or a window tensor constructor, `window(frame_length) -> Tensor`.
        Default: hamming window.
    Returns
    -------
    signals : Time-domain signal(s) reconstructed from `dcts`, a `[..., n_samples]` tensor.
        Note that `n_samples` may be different from the original signals' lengths as passed to `torch_stdct`,
        because no padding is applied.
    """
    *_, frame_length2, n_frames = dcts.shape
    assert frame_length in {None, frame_length2}
    signals = torch_overlap_add(
        torch_dct.idct(dcts.transpose(-1, -2), norm="ortho").transpose(-1, -2),
        frame_step=frame_step,
    )
    if callable(window):
        window = window(frame_length2).to(signals)
    if window is not None:
        window_frames = window[:, None].expand(-1, n_frames)
        window_signal = torch_overlap_add(window_frames, frame_step=frame_step)
        signals = signals / window_signal
    return signals


# Overlap-add required for STDCT
def torch_overlap_add(framed, *, frame_step, frame_length=None):
    """Overlap-add ("deframe") a framed signal.
    Parameters other than `framed` are keyword-only.
    Parameters
    ----------
    framed : Tensor of shape `(..., frame_length, n_frames)`.
    frame_step : Overlap to use when adding frames.
    frame_length : Ignored.  Window length and DCT frame length in samples.
        Can be None (default) or same value as passed to `torch_stdct`.
    Returns
    -------
    deframed : Overlap-add ("deframed") signal.
        Tensor of shape `(..., (n_frames - 1) * frame_step + frame_length)`.
    """
    *rest, frame_length2, n_frames = framed.shape
    assert frame_length in {None, frame_length2}
    return torch.nn.functional.fold(
        framed.reshape(-1, frame_length2, n_frames),
        output_size=(((n_frames - 1) * frame_step + frame_length2), 1),
        kernel_size=(frame_length2, 1),
        stride=(frame_step, 1),
    ).reshape(*rest, -1)

"""
Audio Augmentation stuff
Similar to Demucs augmentations
"""
class FlipChannels(nn.Module):
    """
    Flip channels
    """
    def forward(self, wav):
        batch, sources, channels, time = wav.size()
        if self.training and wav.size(2) == 2:
            left = torch.randint(2, (batch, sources, 1, 1), device=wav.device)
            left = left.expand(-1,-1,-1,time)
            right = 1-left
            wav = torch.cat([wav.gather(2,left),wav.gather(2,right)], dim=2)
        return wav

class FlipSign(nn.Module):
    """
    Random sign flip.
    """
    def forward(self, wav):
        batch, sources, channels, time = wav.size()
        if self.training:
            signs = torch.randint(2, (batch, sources, 1, 1), device=wav.device, dtype=torch.float32)
            wav = wav * (2 * signs - 1)
        return wav

# Call augment in your train method
def augment():
    augments = [FlipChannels(), FlipSign()]
    return torch.nn.Sequential(*augments)



# Noise Invariant Loss Calculation
def calc_loss(x, n, gt, weight):
    if n is None:
        pred_mask = torch.sigmoid(x)
    elif x is None:
        pred_mask = torch.sigmoid(n)
    else:
        pred_mask = 1 - nn.functional.relu(1-(torch.sigmoid(x) + torch.sigmoid(n)))
    BCELoss = nn.BCEWithLogitsLoss(weight=weight, reduction="none")
    print(x.shape,n.shape,gt.shape,pred_mask.shape)
    return torch.mean(
        BCELoss(pred_mask, gt),
        (1,2)
    )