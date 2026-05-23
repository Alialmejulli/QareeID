import os
import json
import numpy as np

CONFIDENT_THRESHOLD = 0.50
UNCERTAIN_THRESHOLD = 0.15


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def match(query_emb: np.ndarray, enrolled: list[dict]) -> list[dict]:
    scores: dict[str, list[float]] = {}
    display_names: dict[str, str] = {}
    for item in enrolled:
        sim = _cosine(query_emb, item["embedding"])
        scores.setdefault(item["name"], []).append(sim)
        display_names[item["name"]] = item.get("display_name") or item["name"]

    ranked = [
        {"name": name, "display_name": display_names[name], "score": float(np.mean(sims))}
        for name, sims in scores.items()
    ]
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked


def interpret(ranked: list[dict]) -> dict:
    if not ranked:
        return {"name": None, "display_name": None, "score": 0.0, "confidence": "unknown", "needs_claude": False}

    top = ranked[0]
    score = top["score"]
    dn = top["display_name"]

    if score >= CONFIDENT_THRESHOLD:
        return {"name": top["name"], "display_name": dn, "score": score, "confidence": "high", "needs_claude": False}
    elif score >= UNCERTAIN_THRESHOLD:
        return {"name": top["name"], "display_name": dn, "score": score, "confidence": "low", "needs_claude": True}
    else:
        return {"name": None, "display_name": None, "score": score, "confidence": "unknown", "needs_claude": False}


def ask_claude(_audio_path: str, ranked: list[dict], transcript: str = "") -> dict:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key or api_key == "your_api_key_here":
        return {"name": ranked[0]["name"], "score": ranked[0]["score"],
                "confidence": "low", "method": "cosine_fallback"}

    candidates = "\n".join(
        f"  {i + 1}. {r['name']} — {r['score']:.3f}"
        for i, r in enumerate(ranked[:5])
    )
    transcript_line = f"\nWhisper transcript of the audio:\n{transcript}\n" if transcript else ""

    prompt = (
        "You are an expert on Quran reciters (Qaris). Based on voice analysis, "
        "the top candidates are:\n\n"
        f"{candidates}\n"
        f"{transcript_line}\n"
        "Pick the single most likely reciter. Consider vocal style, tone, and typical "
        "characteristics of each reciter. Reply with JSON only, no explanation outside the JSON:\n"
        '{"name": "<exact name from the list>", "reasoning": "<one sentence>"}'
    )

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        result = json.loads(text)
        chosen = result.get("name", ranked[0]["name"])
        reasoning = result.get("reasoning", "")
        score = next((r["score"] for r in ranked if r["name"] == chosen), ranked[0]["score"])
        return {"name": chosen, "score": score, "reasoning": reasoning,
                "confidence": "claude", "method": "claude"}
    except Exception:
        return {"name": ranked[0]["name"], "score": ranked[0]["score"],
                "confidence": "low", "method": "cosine_fallback"}
