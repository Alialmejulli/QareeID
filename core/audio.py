import os
import numpy as np
import soundfile as sf
import librosa

TARGET_SR = 16000


def load_audio(path: str) -> tuple[np.ndarray, int]:
    if not os.path.isfile(path):
        raise ValueError(f"Audio file not found: {path}")

    ext = os.path.splitext(path)[1].lower()

    try:
        if ext in (".mp3", ".m4a", ".aac", ".ogg"):
            audio, sr = _load_via_pydub(path)
        else:
            audio, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception as e:
        raise ValueError(f"Cannot load audio file '{path}': {e}") from e

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak

    return audio.astype(np.float32), TARGET_SR


def _load_via_pydub(path: str) -> tuple[np.ndarray, int]:
    from pydub import AudioSegment
    seg = AudioSegment.from_file(path)
    seg = seg.set_channels(1).set_sample_width(2)
    sr = seg.frame_rate
    samples = np.array(seg.get_array_of_samples(), dtype=np.float32) / 32768.0
    return samples, sr


def slice_audio(audio: np.ndarray, sr: int, segment_sec: float = 5.0) -> list[np.ndarray]:
    seg_len = int(sr * segment_sec)
    min_len = int(sr * 1.0)

    if len(audio) < min_len:
        return [audio]

    segments = []
    for start in range(0, len(audio), seg_len):
        chunk = audio[start: start + seg_len]
        if len(chunk) >= min_len:
            segments.append(chunk)
    return segments
