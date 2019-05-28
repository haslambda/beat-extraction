"""
extractor.py
"""
import wave
import math
import contextlib
import numpy as np
from json import dumps
from sys import argv

fname = argv[1]
outname = 'filtered.wav'
cutoff_freq = 400.0
thresold = 1500


def running_mean(x, window_size):
    """
    running_mean
    """
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size


def interpret_wav(raw_bytes, n_frames, n_channels, sample_width,
                  interleaved=True):
    """
    interpret_wav
    """
    if sample_width == 1:
        dtype = np.uint8
    elif sample_width == 2:
        dtype = np.int16
    else:
        raise ValueError("Only supports 8 and 16 bit audio formats.")

    channels = np.fromstring(raw_bytes, dtype=dtype)

    if interleaved:
        channels.shape = (n_frames, n_channels)
        channels = channels.T
    else:
        channels.shape = (n_channels, n_frames)

    return channels


with contextlib.closing(wave.open(fname, 'rb')) as spf:
    rate = spf.getframerate()
    amp_width = spf.getsampwidth()
    channels = spf.getnchannels()
    frames = spf.getnframes()

    signal = spf.readframes(frames*channels)
    spf.close()
    channels = interpret_wav(signal, frames, channels, amp_width, True)

    freq_ratio = (cutoff_freq/rate)
    N = int(math.sqrt(0.196196 + freq_ratio**2)/freq_ratio)

    filtered: np.ndarray = running_mean(channels[0], N).astype(channels.dtype)

    wav_file = wave.open(outname, "w")
    wav_file.setparams((1, amp_width, rate, frames, spf.getcomptype(), spf.getcompname()))
    wav_file.writeframes(filtered.tobytes('C'))
    wav_file.close()

    duration = frames / float(rate)
    segments = np.array_split(filtered, duration * 4)
    processed = np.zeros_like(filtered)
    start = 0

    for segment in segments:
        strongest = np.amax(segment)
        strongest_arr = segment.copy()
        strongest_arr.fill(strongest)
        processed[start:start + strongest_arr.size] = strongest_arr
        start += strongest_arr.size

    right_shifted = np.delete(np.insert(processed, 0, 0), -1)
    diff = right_shifted - processed
    beats = np.where(diff >= thresold)[0] / diff.size * duration
    frame_drop = 0

    print(dumps(beats.tolist()))
