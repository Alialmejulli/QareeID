import numpy as np
import torch

_model = None


def _get_model():
    global _model
    if _model is None:
        from speechbrain.inference.classifiers import EncoderClassifier
        print("[embedder] Loading ECAPA-TDNN model (first run downloads ~150 MB)...")
        _model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"},
        )
        _model.eval()
        print("[embedder] Model ready.")
    return _model


def get_embedding(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    model = _get_model()
    wav = torch.from_numpy(audio).unsqueeze(0)  # (1, T)
    with torch.no_grad():
        emb = model.encode_batch(wav)           # (1, 1, 192)
    emb = emb.squeeze().numpy()                 # (192,)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb.astype(np.float32)


def average_embedding(embeddings: list[np.ndarray]) -> np.ndarray:
    arr = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr.astype(np.float32)
