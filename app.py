import os
import sys
import tempfile
import threading
from pathlib import Path

import webview
from dotenv import load_dotenv

load_dotenv()

DEBUG = os.getenv("QAREE_DEBUG", "0") == "1"

# Set working directory to the exe/script location so relative paths
# (data/quranid.db, pretrained_models/) resolve correctly regardless of
# where the user launched the process from.
if hasattr(sys, '_MEIPASS'):
    os.chdir(os.path.dirname(sys.executable))
else:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Ensure project root is on path so `core.*` imports resolve
sys.path.insert(0, str(Path(__file__).parent))


def resource_path(relative_path):
    """Get absolute path to resource — works for dev and PyInstaller."""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


class API:
    def __init__(self):
        self._stop_event = threading.Event()

    def identify(self, file_path: str) -> dict:
        import os
        if DEBUG: print(f"[DEBUG] Working dir: {os.getcwd()}")
        if DEBUG: print(f"[DEBUG] DB exists: {os.path.exists('data/quranid.db')}")
        if DEBUG: print(f"[DEBUG] Model exists: {os.path.exists('pretrained_models/spkrec-ecapa-voxceleb')}")
        if DEBUG: print(f"[DEBUG] File exists: {os.path.exists(file_path)}")
        try:
            from core.audio import load_audio, slice_audio
            from core.embedder import get_embedding, average_embedding
            from core import database as db
            from core import matcher

            db.init_db()

            enrolled = db.get_all_embeddings()
            if not enrolled:
                return {"success": False, "error": "No reciters enrolled yet"}

            audio, sr = load_audio(file_path)
            segments = slice_audio(audio, sr, segment_sec=5.0)
            embeddings = [get_embedding(seg) for seg in segments]
            query_emb = average_embedding(embeddings)

            ranked = matcher.match(query_emb, enrolled)
            result = matcher.interpret(ranked)

            if result.get("confidence") == "low" and result.get("score", 0) > 0.25:
                result["confidence"] = "high"
                result["needs_claude"] = False

            method = "cosine"
            if result.get("needs_claude"):
                claude_result = matcher.ask_claude(file_path, ranked)
                # fix display_name in case Claude chose a different candidate
                chosen_name = claude_result.get("name", ranked[0]["name"])
                for r in ranked:
                    if r["name"] == chosen_name:
                        claude_result["display_name"] = r["display_name"]
                        break
                result.update(claude_result)
                method = claude_result.get("method", "claude")

            # arabic_name lookup from reciters table
            reciters_info = {r["name"]: r for r in db.list_reciters()}

            top_name = result.get("name") or ""
            top_info = reciters_info.get(top_name, {})

            db.save_history(
                file_path,
                top_name or "unknown",
                result.get("score", 0.0),
                method,
            )

            top5 = []
            for r in ranked[:5]:
                info = reciters_info.get(r["name"], {})
                top5.append({
                    "name": r.get("display_name") or r["name"],
                    "arabic_name": info.get("arabic_name", ""),
                    "score": round(float(r["score"]), 4),
                })

            result["confidence"] = (result.get("confidence") or "unknown").upper()

            return {
                "success": True,
                "name": result.get("display_name") or top_name or "Unknown",
                "arabic_name": top_info.get("arabic_name", ""),
                "score": round(float(result.get("score", 0.0)), 4),
                "confidence": result["confidence"],
                "top5": top5,
            }

        except Exception as e:
            import traceback
            print(f"[ERROR] identify() failed: {e}")
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def listen(self, duration: int = 15) -> dict:
        self._stop_event.clear()
        try:
            import sounddevice as sd
            import soundfile as sf
            import numpy as np
            import queue as _queue
            from core.audio import slice_audio
            from core.embedder import get_embedding, average_embedding
            from core import database as db
            from core import matcher

            db.init_db()
            enrolled = db.get_all_embeddings()
            if not enrolled:
                return {"success": False, "error": "No reciters enrolled yet"}

            SR = 16000
            q = _queue.Queue()

            def _cb(indata, frames, time_info, status):
                q.put(indata.copy())

            # Try each device config in order — first one that opens wins
            stream = None
            native_sr = SR
            for cfg in [
                dict(device=32, samplerate=SR, extra_settings=sd.WasapiSettings(exclusive=False)),
                dict(device=32, samplerate=48000),
                dict(samplerate=SR),
            ]:
                try:
                    stream = sd.InputStream(channels=1, dtype='float32', callback=_cb, **cfg)
                    stream.start()
                    native_sr = cfg.get('samplerate', SR)
                    if DEBUG: print(f"[DEBUG] Input: device {cfg.get('device','default')} @ {native_sr}Hz")
                    break
                except Exception as e:
                    if DEBUG: print(f"[DEBUG] Config failed: {e}")
                    stream = None

            if stream is None:
                return {"success": False, "error": "No working audio input found"}

            frames = []
            total = 0
            target = duration * native_sr
            EARLY_STOP_SCORE = 0.65
            CHECK_EVERY = native_sr * 3   # probe every 3s
            MIN_CHECK = native_sr * 5     # don't probe before 5s
            next_check = MIN_CHECK
            try:
                while total < target:
                    if self._stop_event.is_set():
                        break
                    try:
                        chunk = q.get(timeout=2.0)
                        frames.append(chunk)
                        total += len(chunk)
                    except _queue.Empty:
                        if DEBUG: print("[DEBUG] Audio stream timeout — no data for 2s")
                        break

                    if total >= next_check:
                        next_check = total + CHECK_EVERY
                        probe = np.concatenate(frames).flatten()
                        if native_sr != SR:
                            from scipy.signal import resample_poly
                            import math
                            g = math.gcd(SR, native_sr)
                            probe = resample_poly(probe, SR // g, native_sr // g).astype(np.float32)
                        segs = slice_audio(probe, SR, segment_sec=5.0)
                        if segs:
                            embs = [get_embedding(seg) for seg in segs]
                            quick_ranked = matcher.match(average_embedding(embs), enrolled)
                            if quick_ranked and quick_ranked[0]["score"] >= EARLY_STOP_SCORE:
                                if DEBUG: print(f"[DEBUG] Early stop at {total/native_sr:.1f}s — score {quick_ranked[0]['score']:.3f}")
                                break
            finally:
                stream.stop()
                stream.close()

            if self._stop_event.is_set():
                return {"success": False, "stopped": True}

            if not frames:
                return {"success": False, "error": "No audio captured"}

            audio = np.concatenate(frames).flatten()

            if native_sr != SR:
                from scipy.signal import resample_poly
                import math
                g = math.gcd(SR, native_sr)
                audio = resample_poly(audio, SR // g, native_sr // g).astype(np.float32)

            rms = float(np.sqrt(np.mean(audio ** 2)))
            if DEBUG: print(f"[DEBUG] Audio length: {len(audio)} samples ({len(audio)/SR:.1f}s)  RMS: {rms:.4f}")

            if 0.001 < rms < 0.02:
                gain = min(0.02 / rms, 10.0)
                audio = audio * gain
                if DEBUG: print(f"[DEBUG] Applied gain: {gain:.1f}x")

            # Same pipeline as CLI — raw array, no load_audio(), no noise reduction
            segments = slice_audio(audio, SR, segment_sec=5.0)
            if not segments:
                return {"success": False, "error": "Recording too short"}

            if DEBUG: print(f"[DEBUG] {len(segments)} segment(s). Identifying...")
            embeddings = [get_embedding(seg) for seg in segments]
            query_emb = average_embedding(embeddings)

            ranked = matcher.match(query_emb, enrolled)
            result = matcher.interpret(ranked)

            if result.get("confidence") == "low" and result.get("score", 0) > 0.25:
                result["confidence"] = "high"
                result["needs_claude"] = False

            method = "cosine"
            if result.get("needs_claude"):
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp_path = tmp.name
                tmp.close()
                sf.write(tmp_path, audio, SR)
                claude_result = matcher.ask_claude(tmp_path, ranked)
                chosen_name = claude_result.get("name", ranked[0]["name"])
                for r in ranked:
                    if r["name"] == chosen_name:
                        claude_result["display_name"] = r["display_name"]
                        break
                result.update(claude_result)
                method = claude_result.get("method", "claude")
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

            reciters_info = {r["name"]: r for r in db.list_reciters()}
            top_name = result.get("name") or ""
            top_info = reciters_info.get(top_name, {})

            db.save_history("[microphone]", top_name or "unknown", result.get("score", 0.0), method)

            top5 = []
            for r in ranked[:5]:
                info = reciters_info.get(r["name"], {})
                top5.append({
                    "name": r.get("display_name") or r["name"],
                    "arabic_name": info.get("arabic_name", ""),
                    "score": round(float(r["score"]), 4),
                })

            result["confidence"] = (result.get("confidence") or "unknown").upper()
            if DEBUG: print(f"[DEBUG] Score: {result.get('score')}  Confidence: {result.get('confidence')}  Name: {top_name}")

            return {
                "success": True,
                "name": result.get("display_name") or top_name or "Unknown",
                "arabic_name": top_info.get("arabic_name", ""),
                "score": round(float(result.get("score", 0.0)), 4),
                "confidence": result["confidence"],
                "top5": top5,
            }

        except Exception as exc:
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(exc)}

    def stop_listen(self) -> dict:
        self._stop_event.set()
        return {"success": True}

    def get_history(self) -> dict:
        try:
            from core import database as db
            db.init_db()
            rows = db.get_history(limit=50)
            return {"success": True, "history": [dict(r) for r in rows]}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def get_reciters(self) -> dict:
        try:
            from core import database as db

            db.init_db()
            reciters = db.list_reciters()
            # convert sqlite Row dicts to plain dicts so pywebview can serialise them
            return {"success": True, "reciters": [dict(r) for r in reciters]}

        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def open_file_dialog(self) -> dict:
        file_types = (
            "Audio Files (*.mp3;*.wav;*.flac;*.m4a;*.ogg;*.aac)",
            "All files (*.*)",
        )
        result = webview.windows[0].create_file_dialog(
            webview.OPEN_DIALOG,
            allow_multiple=False,
            file_types=file_types,
        )
        if result:
            return {"success": True, "path": result[0]}
        return {"success": False}


if __name__ == "__main__":
    from core.embedder import get_embedding
    import numpy as np
    try:
        get_embedding(np.zeros(16000, dtype=np.float32))
    except Exception:
        pass

    api = API()
    window = webview.create_window(
        "QareeID",
        resource_path("ui.html"),
        js_api=api,
        width=900,
        height=650,
        resizable=False,
    )
    webview.start()
