import os
import sys
import tempfile
from pathlib import Path

import webview
from dotenv import load_dotenv

load_dotenv()

# Ensure project root is on path so `core.*` imports resolve
sys.path.insert(0, str(Path(__file__).parent))


class API:
    def identify(self, file_path: str) -> dict:
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

            if result.get("confidence") == "low" and result.get("score", 0) > 0.35:
                result["confidence"] = "high"

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

        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def listen(self, duration: int = 15) -> dict:
        try:
            import sounddevice as sd
            import soundfile as sf
            import numpy as np

            SR = 16000
            recording = sd.rec(
                int(duration * SR), samplerate=SR, channels=1, dtype="float32"
            )
            sd.wait()
            audio = recording.squeeze()

            rms = float(np.sqrt(np.mean(audio ** 2)))
            if rms < 0.02:
                return {
                    "success": True,
                    "confidence": "UNKNOWN",
                    "score": 0.0,
                    "name": None,
                    "arabic_name": None,
                    "display_name": None,
                    "top5": [],
                }

            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_path = tmp.name
            tmp.close()

            sf.write(tmp_path, audio, SR)

            result = self.identify(tmp_path)
            if result.get("success"):
                if result.get("confidence") == "LOW" and result.get("score", 0) > 0.35:
                    result["confidence"] = "HIGH"
                result["confidence"] = str(result.get("confidence", "UNKNOWN")).upper()

            try:
                os.unlink(tmp_path)
            except OSError:
                pass

            return result

        except Exception as exc:
            return {"success": False, "error": str(exc)}

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
        str(Path(__file__).parent / "ui.html"),
        js_api=api,
        width=900,
        height=650,
        resizable=False,
    )
    webview.start()
