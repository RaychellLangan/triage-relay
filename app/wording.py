"""Staff-screen wording: one translator field -> a short everyday line, instead of the dataset's question text.

    "E_54_@_V_181" -> "Pain feels shooting"        (not "Characterize your pain: haunting")
    "E_55_@_V_29"  -> "Pain in: left upper arm"     (not "Do you feel pain somewhere: biceps(L)")

Reuses the everyday vocabulary written for the natural-language training data (data/natural.py), always its first phrasing,
so the same field always reads the same way. Display only: routing never sees these words.
"""
import natural as N


class _First:
    """stands in for a random generator: always the first phrasing"""
    @staticmethod
    def choice(xs):
        return xs[0]


def _cap(s):
    return s[:1].upper() + s[1:] if s else s


def _meaning(ev, val):
    if isinstance(val, str) and val.startswith("V_"):
        return ev["value_meaning"].get(val, {}).get("en", val)
    return str(val)


def _yn(m):
    return {"Y": "yes", "N": "no"}.get(m, m)


SCALE = {"E_56": "Pain {}/10", "E_58": "Can point to exactly where it hurts: {}/10", "E_59": "How suddenly it came on: {}/10",
         "E_134": "Rash pain {}/10", "E_132": "Rash swelling {}/10", "E_136": "Itching {}/10"}
PLACE = {"E_55": "Pain in: {}", "E_57": "Spreads to: {}", "E_133": "Affected area: {}", "E_152": "Swelling at: {}"}


# display-only words for body places the training vocabulary leaves in the dataset's terms ('breast(R)' = one side of the chest)
DISPLAY_PLACE = {"breast": "{s} side of the chest"}


def _place(meaning):
    base, _, side = meaning.strip().partition("(")
    say = DISPLAY_PLACE.get(base.strip())
    if say:
        return say.format(s={"L)": "left", "R)": "right"}.get(side, "")).strip()
    return N.loc_words(meaning, _First)


def staff_line(code, evid):
    name, _, val = code.partition("_@_")
    ev = evid.get(name)
    if not ev:
        return code
    if not val:
        if name == "E_53":
            return "Has pain related to this visit"
        return _cap(N.SAY[name][0]) if name in N.SAY else _cap(N.first_person(ev["question_en"]))
    m = _meaning(ev, val)
    if name in PLACE:
        if m == "nowhere":
            return "Pain does not spread" if name == "E_57" else "No particular place"
        return PLACE[name].format(_place(m))
    if name == "E_54":
        return "Pain feels " + N.PAIN_WORD.get(m, [m])[0] if m != "NA" else "Pain: no particular kind"
    if name in SCALE:
        return SCALE[name].format(m)
    if name == "E_204":
        return "No travel abroad" if m == "N" else f"Travelled to {m} (last 4 weeks)"
    if name == "E_130":
        return f"Rash colour: {m}"
    if name == "E_135":
        return f"Lesion larger than 1cm: {_yn(m)}"
    if name == "E_131":
        return f"Lesions peel: {_yn(m)}"
    return _cap(N.first_person(ev["question_en"])) + ": " + m
