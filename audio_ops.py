import numpy as np

SAMPLE_RATE = 48000


def append_audio(previous, segment, overlap_samples):
    if previous is None:
        return segment
    overlap = min(overlap_samples, previous.shape[-1] // 2, segment.shape[-1] // 2)
    if overlap == 0:
        return np.concatenate((previous, segment), axis=-1)
    fade = np.linspace(0, 1, overlap, dtype=np.float32)
    blend = previous[..., -overlap:] * (1 - fade) + segment[..., :overlap] * fade
    return np.concatenate((previous[..., :-overlap], blend, segment[..., overlap:]), axis=-1)


def finish_audio(waveform, samples, normalize, crop_mode="start"):
    if crop_mode not in ("start", "peak"):
        raise ValueError(f"Unknown crop mode: {crop_mode}")
    if waveform.shape[-1] < samples:
        raise ValueError("Woosh output is shorter than the requested duration.")
    if not np.isfinite(waveform).all():
        raise ValueError("Woosh generated NaN/Inf audio. Check the backend and model weights.")
    if crop_mode == "peak" and waveform.shape[-1] > samples:
        peaks = np.abs(waveform).max(axis=1).argmax(axis=-1)
        preroll = min(round(0.03 * SAMPLE_RATE), samples // 4)
        starts = np.clip(peaks - preroll, 0, waveform.shape[-1] - samples)
        waveform = np.stack([item[:, start:start + samples] for item, start in zip(waveform, starts)]).astype(np.float32)
        fade_samples = min(round(0.002 * SAMPLE_RATE), samples // 8)
        if fade_samples > 1:
            fade = np.linspace(0, 1, fade_samples, dtype=np.float32)
            waveform[..., :fade_samples] *= fade
            waveform[..., -fade_samples:] *= fade[::-1]
    else:
        waveform = waveform[..., :samples].astype(np.float32, copy=True)
    if normalize:
        peak = np.maximum(1.0, np.max(np.abs(waveform), axis=(-2, -1), keepdims=True))
        waveform /= peak
    return waveform
