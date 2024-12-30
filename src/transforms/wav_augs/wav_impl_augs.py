import torch
import torch.nn as nn
from torch_audiomentations import (
    AddColoredNoise,
    HighPassFilter,
    LowPassFilter,
    PitchShift,
)


class ApplyPitchShift(nn.Module):
    def __init__(self, sample_rate, *args, **kwargs):
        super(ApplyPitchShift, self).__init__()
        self.transform = PitchShift(sample_rate=sample_rate, *args, **kwargs)

    def forward(self, audio_waveform):
        audio_waveform = audio_waveform.unsqueeze(1)
        return self.transform(audio_waveform).squeeze(1)


class AddNoiseAug(nn.Module):
    def __init__(self, sample_rate, *args, **kwargs):
        super(AddNoiseAug, self).__init__()
        self.noise_augment = AddColoredNoise(sample_rate=sample_rate, *args, **kwargs)

    def forward(self, audio_waveform):
        audio_waveform = audio_waveform.unsqueeze(1)
        return self.noise_augment(audio_waveform).squeeze(1)


class ApplyHighPassFilter(nn.Module):
    def __init__(self, sample_rate, min_cutoff_freq, max_cutoff_freq, *args, **kwargs):
        super(ApplyHighPassFilter, self).__init__()
        self.hp_filter = HighPassFilter(
            sample_rate=sample_rate,
            min_cutoff_freq=min_cutoff_freq,
            max_cutoff_freq=max_cutoff_freq,
            *args, **kwargs
        )

    def forward(self, audio_waveform):
        audio_waveform = audio_waveform.unsqueeze(1)
        return self.hp_filter(audio_waveform).squeeze(1)


class ApplyLowPassFilter(nn.Module):
    def __init__(self, sample_rate, min_cutoff_freq, max_cutoff_freq, *args, **kwargs):
        super(ApplyLowPassFilter, self).__init__()
        self.lp_filter = LowPassFilter(
            sample_rate=sample_rate,
            min_cutoff_freq=min_cutoff_freq,
            max_cutoff_freq=max_cutoff_freq,
            *args, **kwargs
        )

    def forward(self, audio_waveform):
        audio_waveform = audio_waveform.unsqueeze(1)
        return self.lp_filter(audio_waveform).squeeze(1)
