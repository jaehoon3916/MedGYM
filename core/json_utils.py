from __future__ import annotations

import json
import re


def safe_json_load(s: str) -> dict:
    """Parse a JSON object from an LLM response.

    Tolerant of markdown code fences (```json ... ```) and surrounding prose:
    falls back to extracting the first {...} block. Returns {} on failure.
    """
    if not s:
        return {}
    t = s.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}
    return {}
