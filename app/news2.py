"""NEWS2 (National Early Warning Score 2, Royal College of Physicians, 2017) from the nurse's observations.

A published, nationally used scoring chart, copied here as written (SpO2 scale 1). It is shown to staff next to the routing card
and is NOT used for routing in this build. Missing observations are left out and the score says which ones are missing.
"""
import re

BANDS = {   # parameter: list of (upper bound inclusive, points), first match wins
    "rr":   [(8, 3), (11, 1), (20, 0), (24, 2), (999, 3)],
    "spo2": [(91, 3), (93, 2), (95, 1), (100, 0)],
    "sbp":  [(90, 3), (100, 2), (110, 1), (219, 0), (999, 3)],
    "hr":   [(40, 3), (50, 1), (90, 0), (110, 1), (130, 2), (999, 3)],
    "temp": [(35.0, 3), (36.0, 1), (38.0, 0), (39.0, 1), (99, 2)],
}
NEEDED = ["rr", "spo2", "o2", "sbp", "hr", "acvpu", "temp"]


def _num(s):
    m = re.search(r"\d+(\.\d+)?", str(s or ""))
    return float(m.group()) if m else None


def score(vitals):
    v = dict(vitals or {})
    v["sbp"] = _num(v.get("bp"))                  # "128/80" -> systolic 128
    parts, missing = {}, []
    for k in NEEDED:
        if k in BANDS:
            x = _num(v.get(k))
            if x is None:
                missing.append(k)
                continue
            parts[k] = next(p for hi, p in BANDS[k] if x <= hi)
        elif k == "o2":
            if v.get("o2") in ("air", "oxygen"):
                parts[k] = 2 if v["o2"] == "oxygen" else 0
            else:
                missing.append(k)
        elif k == "acvpu":
            if v.get("acvpu"):
                parts[k] = 0 if v["acvpu"] == "A" else 3
            else:
                missing.append(k)
    if not parts:
        return None
    total = sum(parts.values())
    # RCP clinical risk: 0-4 low; a 3 in any single parameter low-medium; 5-6 medium; 7+ high
    risk = "high" if total >= 7 else "medium" if total >= 5 else "low-medium" if 3 in parts.values() else "low"
    return {"total": total, "risk": risk, "parts": parts, "missing": missing}


def qsofa(vitals):
    """qSOFA (Sepsis-3, Seymour/Singer et al., JAMA 2016): one point each for breathing rate >= 22, altered mentation, systolic BP <= 100.
    2 or more = higher risk of a poor outcome from infection. In the UK NICE prefers NEWS2 for adults; shown here as the
    international quick check. Altered mentation is read from ACVPU (anything other than Alert)."""
    v = vitals or {}
    rr, sbp, a = _num(v.get("rr")), _num(v.get("bp")), v.get("acvpu")
    if rr is None or sbp is None or not a:
        return None
    pts = (rr >= 22) + (sbp <= 100) + (a != "A")
    return {"total": int(pts), "flag": pts >= 2}
