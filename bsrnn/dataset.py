import argparse
import tqdm
import random
from pathlib import Path
from typing import Optional, Union, Tuple, List, Any, Callable

import torch
import torch.utils.data
import torchaudio


def load_info(path: str) -> dict:
    """Load audio metadata
    this is a backend_independent wrapper around torchaudio.info
    Args:
        path: Path of filename
    Returns:
        Dict: Metadata with
        `samplerate`, `samples` and `duration` in seconds
    """
    # get length of file in samples
    if torchaudio.get_audio_backend() == "sox":
        raise RuntimeError("Deprecated backend is not supported")

    info = {}
    si = torchaudio.info(str(path))
    info["samplerate"] = si.sample_rate
    info["samples"] = si.num_frames
    info["channels"] = si.num_channels
    info["duration"] = info["samples"] / info["samplerate"]
    return info


def load_audio(
        path: str,
        start: float = 0.0,
        dur: Optional[float] = None,
        info: Optional[dict] = None,
):
    """Load audio file
    Args:
        path: Path of audio file
        start: start position in seconds, defaults on the beginning.
        dur: end position in seconds, defaults to `None` (full file).
        info: metadata object as called from `load_info`.
    Returns:
        Tensor: torch tensor waveform of shape `(num_channels, num_samples)`
    """
    # loads the full track duration
    if dur is None:
        # we ignore the case where start!=0 and dur=None
        # since we have to deal with fixed length audio
        sig, rate = torchaudio.load(path)
        return sig, rate
    else:
        if info is None:
            info = load_info(path)
        num_frames = int(dur * info["samplerate"])
        frame_offset = int(start * info["samplerate"])
        sig, rate = torchaudio.load(path, num_frames=num_frames, frame_offset=frame_offset)
        return sig, rate


def aug_from_str(list_of_function_names: list):
    if list_of_function_names:
        return Compose([globals()["_augment_" + aug] for aug in list_of_function_names])
    else:
        return lambda audio: audio


class Compose(object):
    """Composes several augmentation transforms.
    Args:
        augmentations: list of augmentations to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            audio = t(audio)
        return audio


def _augment_gain(audio: torch.Tensor, low: float = 0.25, high: float = 1.25) -> torch.Tensor:
    """Applies a random gain between `low` and `high`"""
    g = low + torch.rand(1) * (high - low)
    return audio * g


def _augment_mag(audio: torch.Tensor) -> torch.Tensor:
    """ Flip signal magnitude randomly"""
    flip = 2 * torch.randint(2, (1,), dtype=torch.float32) - 1
    return audio * flip


def _augment_channelswap(audio: torch.Tensor) -> torch.Tensor:
    """Swap channels of stereo signals with a probability of p=0.5"""
    if audio.shape[0] == 2 and torch.tensor(1.0).uniform_() < 0.5:
        return torch.flip(audio, [0])
    else:
        return audio


def _augment_force_stereo(audio: torch.Tensor) -> torch.Tensor:
    # for multichannel > 2, we drop the other channels
    if audio.shape[0] > 2:
        audio = audio[:2, ...]

    if audio.shape[0] == 1:
        # if we have mono, we duplicate it to get stereo
        audio = torch.repeat_interleave(audio, 2, dim=0)

    return audio


class UnmixDataset(torch.utils.data.Dataset):
    _repr_indent = 4

    def __init__(
            self,
            root: Union[Path, str],
            sample_rate: float,
            seq_duration: Optional[float] = None,
            source_augmentations: Optional[Callable] = None,
    ) -> None:
        self.root = Path(args.root).expanduser()
        self.sample_rate = sample_rate
        self.seq_duration = seq_duration
        self.source_augmentations = source_augmentations

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        body += self.extra_repr().splitlines()
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)

    def extra_repr(self) -> str:
        return ""


class SDXDataset(UnmixDataset):
    def __init__(
            self,
            root: str,
            split: str = "train",
            target_file: str = "vocals.wav",
            interferer_files: List[str] = ["bass.wav", "drums.wav"],
            seq_duration: Optional[float] = None,
            random_chunks: bool = False,
            random_track_mix: bool = False,
            source_augmentations: Optional[Callable] = lambda audio: audio,
            sample_rate: float = 44100.0,
            seed: int = 42,
    ) -> None:
        """A dataset that assumes audio sources to be stored
        in track folder where each track has a fixed number of sources.
        For each track the users specifies the target file-name (`target_file`)
        and a list of interferences files (`interferer_files`).
        A linear mix is performed on the fly by summing the target and
        the inferers up.
        Due to the fact that all tracks comprise the exact same set
        of sources, the random track mixing augmentation technique
        can be used, where sources from different tracks are mixed
        together. Setting `random_track_mix=True` results in an
        unaligned dataset.
        When random track mixing is enabled, we define an epoch as
        when the the target source from all tracks has been seen and only once
        with whatever interfering sources has randomly been drawn.
        This dataset is recommended to be used for small/medium size
        for example like the MUSDB18 or other custom source separation
        datasets.
        Example
        =======
        train/1/vocals.wav ---------------\
        train/1/drums.wav (interferer1) ---+--> input
        train/1/bass.wav -(interferer2) --/
        train/1/vocals.wav -------------------> output
        """
        self.root = Path(root).expanduser()
        self.split = split
        self.sample_rate = sample_rate
        self.seq_duration = seq_duration
        self.random_track_mix = random_track_mix
        self.random_chunks = random_chunks
        self.source_augmentations = source_augmentations
        # set the input and output files (accept glob)
        self.target_file = target_file
        self.interferer_files = interferer_files
        self.source_files = self.interferer_files + [self.target_file]
        self.seed = seed
        random.seed(self.seed)

        self.tracks = list(self.get_tracks())
        if not len(self.tracks):
            raise RuntimeError("No tracks found")

    def __getitem__(self, index):
        # first, get target track
        track_path = self.tracks[index]["path"]
        min_duration = self.tracks[index]["min_duration"]
        if self.random_chunks:
            # determine start seek by target duration
            start = random.uniform(0, min_duration - self.seq_duration)
        else:
            start = 0

        # assemble the mixture of target and interferers
        audio_sources = []
        # load target
        target_audio, _ = load_audio(
            track_path / self.target_file, start=start, dur=self.seq_duration
        )
        target_audio = self.source_augmentations(target_audio)
        audio_sources.append(target_audio)
        # load interferers
        for source in self.interferer_files:
            # optionally select a random track for each source
            if self.random_track_mix:
                random_idx = random.choice(range(len(self.tracks)))
                track_path = self.tracks[random_idx]["path"]
                if self.random_chunks:
                    min_duration = self.tracks[random_idx]["min_duration"]
                    start = random.uniform(0, min_duration - self.seq_duration)

            audio, _ = load_audio(track_path / source, start=start, dur=self.seq_duration)
            audio = self.source_augmentations(audio)
            audio_sources.append(audio)

        stems = torch.stack(audio_sources)
        # # apply linear mix over source index=0
        x = stems.sum(0)
        # target is always the first element in the list
        y = stems[0]
        return x, y

    def __len__(self):
        return len(self.tracks)

    def get_tracks(self):
        """Loads input and output tracks"""
        p = Path(self.root, self.split)
        for track_path in tqdm.tqdm(p.iterdir()):
            if track_path.is_dir():
                source_paths = [track_path / s for s in self.source_files]
                if not all(sp.exists() for sp in source_paths):
                    print("Exclude track ", track_path)
                    continue

                if self.seq_duration is not None:
                    infos = list(map(load_info, source_paths))
                    # get minimum duration of track
                    min_duration = min(i["duration"] for i in infos)
                    if min_duration > self.seq_duration:
                        yield ({"path": track_path, "min_duration": min_duration})
                else:
                    yield ({"path": track_path, "min_duration": None})


class CLIPSDXDataset(UnmixDataset):
    def __init__(
            self,
            root: str,
            split: str = "train",
            target_file: str = "vocals.wav",
            interferer_files: List[str] = ["bass.wav", "drums.wav"],
            seq_duration: Optional[float] = None,
            random_chunks: bool = False,
            random_track_mix: bool = False,
            source_augmentations: Optional[Callable] = lambda audio: audio,
            sample_rate: float = 44100.0,
            seed: int = 42,
            num_mixtures: int = 2,
    ) -> None:
        """A dataset that assumes audio sources to be stored
        in track folder where each track has a fixed number of sources.
        For each track the users specifies the target file-name (`target_file`)
        and a list of interferences files (`interferer_files`).
        A linear mix is performed on the fly by summing the target and
        the inferers up.
        Due to the fact that all tracks comprise the exact same set
        of sources, the random track mixing augmentation technique
        can be used, where sources from different tracks are mixed
        together. Setting `random_track_mix=True` results in an
        unaligned dataset.
        When random track mixing is enabled, we define an epoch as
        when the the target source from all tracks has been seen and only once
        with whatever interfering sources has randomly been drawn.
        This dataset is recommended to be used for small/medium size
        for example like the MUSDB18 or other custom source separation
        datasets.
        Example
        =======
        train/1/vocals.wav ---------------\
        train/1/drums.wav (interferer1) ---+--> input
        train/1/bass.wav -(interferer2) --/
        train/1/vocals.wav -------------------> output
        """
        self.root = Path(root).expanduser()
        self.split = split
        self.sample_rate = sample_rate
        self.seq_duration = seq_duration
        self.random_track_mix = random_track_mix
        self.random_chunks = random_chunks
        self.source_augmentations = source_augmentations
        self.num_mixtures = num_mixtures if split == "train" else 1
        # set the input and output files (accept glob)
        self.target_file = target_file
        self.interferer_files = interferer_files
        self.source_files = self.interferer_files + [self.target_file]
        self.seed = seed
        random.seed(self.seed)

        self.tracks = list(self.get_tracks())
        if not len(self.tracks):
            raise RuntimeError("No tracks found")

    def __getitem__(self, index):
        # first, get target track
        track_path = self.tracks[index]["path"]
        min_duration = self.tracks[index]["min_duration"]
        if self.random_chunks:
            # determine start seek by target duration
            start = random.uniform(0, min_duration - self.seq_duration)
        else:
            start = 0

        # assemble the mixture of target and interferers
        mixtures = []
        targets = []
        sources = []

        for _ in range(self.num_mixtures):
            audio_sources = []
            # load target
            target_audio, _ = load_audio(
                track_path / self.target_file, start=start, dur=self.seq_duration
            )
            target_audio = self.source_augmentations(target_audio)
            audio_sources.append(target_audio)
            # load interferers
            for source in self.interferer_files:
                # optionally select a random track for each source
                if self.random_track_mix:
                    random_idx = random.choice(range(len(self.tracks)))
                    track_path = self.tracks[random_idx]["path"]
                    if self.random_chunks:
                        min_duration = self.tracks[random_idx]["min_duration"]
                        start = random.uniform(0, min_duration - self.seq_duration)

                audio, _ = load_audio(track_path / source, start=start, dur=self.seq_duration)
                audio = self.source_augmentations(audio)
                audio_sources.append(audio)

            stems = torch.stack(audio_sources)
            # # apply linear mix over source index=0
            x = stems.sum(0)
            # target is always the first element in the list
            y = stems[0]
            sources.append(x)
            targets.append(y)

        mixtures = torch.stack(sources)
        mixtures = mixtures.sum(0)
        sources = torch.stack(sources)
        targets = torch.stack(targets)

        return mixtures, sources, targets

    def __len__(self):
        return len(self.tracks)

    def get_tracks(self):
        """Loads input and output tracks"""
        p = Path(self.root, self.split)
        for track_path in tqdm.tqdm(p.iterdir()):
            if track_path.is_dir():
                source_paths = [track_path / s for s in self.source_files]
                if not all(sp.exists() for sp in source_paths):
                    print("Exclude track ", track_path)
                    continue

                if self.seq_duration is not None:
                    infos = list(map(load_info, source_paths))
                    # get minimum duration of track
                    min_duration = min(i["duration"] for i in infos)
                    if min_duration > self.seq_duration:
                        yield ({"path": track_path, "min_duration": min_duration})
                else:
                    yield ({"path": track_path, "min_duration": None})
