"""How people actually say it: renders a DDXPlus patient's answers as everyday first-person language, for training the translator
to read real desk-side descriptions ("burning pain in my chest, about 7 out of 10, it goes down my left arm") instead of only the
dataset's own question wording. Written for this project; several phrasings per item, chosen at random.

    natural_bits(groups, evid, rng) -> list of short phrases, one per answered question (pain questions merged into one sentence)
"""
import re

# ---- body locations in everyday words ----------------------------------------------------------------------------------
LOC = {
    "hypochondrium": ["upper belly under my {s} ribs", "under my {s} ribs"], "epigastric": ["upper stomach", "top of my stomach", "pit of my stomach"],
    "iliac fossa": ["lower {s} belly", "lower {s} side of my tummy"], "iliac wing": ["{s} hip bone"], "iliac crest": ["{s} hip bone"],
    "belly": ["belly", "tummy", "stomach"], "pharynx": ["throat"], "trachea": ["windpipe", "throat"], "occiput": ["back of my head"],
    "back of head": ["back of my head"], "top of the head": ["top of my head"], "forehead": ["forehead"], "temple": ["{s} temple"],
    "tonsil": ["{s} tonsil", "throat"], "flank": ["{s} side"], "renal fossa": ["{s} lower back"], "posterior chest wall": ["{s} side of my back", "upper back"],
    "side of the chest": ["{s} side of my chest"], "lower chest": ["lower chest", "chest"], "upper chest": ["upper chest", "chest"],
    "popliteal fossa": ["back of my {s} knee"], "ischio jambier": ["{s} hamstring", "back of my {s} thigh"], "palace": ["roof of my mouth"],
    "commissure": ["corner of my mouth"], "vermilion": ["lip"], "scapula": ["{s} shoulder blade"], "trapezius": ["top of my {s} shoulder"],
    "cervical spine": ["neck", "back of my neck"], "thoracic spine": ["upper back", "middle of my back"], "lumbar spine": ["lower back"],
    "thyroid cartilage": ["front of my throat"], "tibia": ["{s} shin"], "dorsal aspect of the foot": ["top of my {s} foot"],
    "dorsal aspect of the hand": ["back of my {s} hand"], "dorsal aspect of the wrist": ["{s} wrist"], "palmar face of the wrist": ["{s} wrist"],
    "palmar side of the forearm": ["{s} forearm"], "lateral side of the foot": ["side of my {s} foot"], "posterior aspect of the ankle": ["back of my {s} ankle"],
    "axilla": ["{s} armpit"], "biceps": ["{s} upper arm"], "triceps": ["back of my {s} arm"], "under the jaw": ["under my jaw"],
    "internal cheek": ["inside of my cheek"], "sole": ["sole of my {s} foot"], "coccyx": ["tailbone"], "pubis": ["above my groin"],
    "side of the neck": ["{s} side of my neck"], "back of the neck": ["back of my neck"],
}


def loc_words(meaning, rng):
    m = re.match(r"^(.*?)\s*\((L|R)\)$", meaning.strip())
    base, side = (m.group(1), {"L": "left", "R": "right"}[m.group(2)]) if m else (meaning.strip(), "")
    opts = LOC.get(base)
    if opts:
        return rng.choice(opts).format(s=side).replace("  ", " ").strip()
    return f"{side} {base}".strip()


PAIN_WORD = {"heartbreaking": ["tearing"], "haunting": ["shooting", "stabbing"], "tedious": ["nagging", "dull"], "sensitive": ["tender", "sore"],
             "a knife stroke": ["stabbing", "knife-like"], "tugging": ["pulling"], "burning": ["burning"], "a cramp": ["cramping", "crampy"],
             "heavy": ["heavy", "pressing", "crushing"], "a pulse": ["throbbing", "pounding"], "violent": ["severe", "really strong"],
             "sharp": ["sharp"], "sickening": ["sickening"], "scary": ["frightening", "scary"], "exhausting": ["draining", "exhausting"]}

# ---- the most common yes/no questions, in everyday words (first person) -------------------------------------------------
SAY = {
    "E_66": ["I'm short of breath", "I can't catch my breath", "struggling to breathe", "breathless"],
    "E_201": ["I've got a cough", "coughing a lot", "I keep coughing"], "E_79": ["I smoke", "I'm a smoker"],
    "E_91": ["I've got a fever", "running a temperature", "feel feverish", "high temperature"],
    "E_181": ["blocked nose", "runny nose", "stuffy nose"], "E_129": ["a rash", "something on my skin", "my skin's come up red"],
    "E_124": ["I have asthma", "asthmatic, I use an inhaler"], "E_50": ["sweating a lot", "I've been really sweaty"],
    "E_78": ["I drink a lot", "I drink too much alcohol"], "E_148": ["feel sick", "nauseous", "feel like I'm going to throw up"],
    "E_70": ["I'm overweight", "I'm quite heavy for my height"], "E_45": ["coughing up blood", "blood when I cough"],
    "E_48": ["I live with lots of people", "big household"], "E_151": ["swelling", "something's swollen"],
    "E_77": ["coughing up green phlegm", "coughing up a lot of phlegm"], "E_49": ["I work in a nursery", "I'm around a daycare"],
    "E_97": ["sore throat", "my throat hurts", "my throat's killing me"], "E_144": ["aching all over", "muscles ache everywhere"],
    "E_82": ["dizzy, like I might faint", "feel faint", "lightheaded, almost passed out"], "E_89": ["always tired", "exhausted, can't sleep properly"],
    "E_41": ["someone at home has the same thing", "been around someone with this"], "E_69": ["I'm diabetic", "I have diabetes"],
    "E_155": ["my heart's racing", "palpitations", "heart beating funny"], "E_104": ["high blood pressure", "I take blood pressure tablets"],
    "E_226": ["I get allergies easily", "allergy prone"], "E_105": ["I've had a heart attack before", "I have angina"],
    "E_214": ["wheezy", "wheezing when I breathe out"], "E_123": ["I have COPD", "lung disease, COPD"],
    "E_88": ["so tired I can't get out of bed", "wiped out, stuck in bed"], "E_222": ["people smoke around me", "secondhand smoke at home"],
    "E_218": ["worse when I walk or exert myself, better when I rest", "gets worse with exercise"], "E_208": ["I'm underweight", "very thin"],
    "E_227": ["my immune system is weak", "I'm immunosuppressed"], "E_220": ["hurts when I breathe in deeply", "painful to take a deep breath"],
    "E_51": ["diarrhoea", "running to the toilet a lot"], "E_76": ["a bit dizzy", "slightly lightheaded"],
    "E_209": ["my jabs are up to date", "vaccinations up to date"], "E_116": ["had a cold recently", "just got over a cold"],
    "E_106": ["I have heart failure"], "E_118": ["I've had pneumonia before"], "E_161": ["not hungry", "lost my appetite", "get full quickly"],
    "E_217": ["worse lying down, better sitting up", "can't lie flat"], "E_162": ["losing weight without trying", "lost weight recently"],
    "E_125": ["I have acid reflux", "I've been told I have reflux"], "E_194": ["a high pitched noise when I breathe in", "whistling when I breathe in"],
    "E_62": ["I take stimulants", "I use speed"], "E_71": ["high cholesterol", "on statins"], "E_9": ["swollen glands", "lumps in my neck"],
    "E_175": ["just feel generally unwell", "feel rubbish all over"], "E_225": ["heart problems run in my family"],
    "E_0": ["had a virus recently", "been ill with a virus"], "E_146": ["I'm on blood thinners"], "E_24": ["I've had anaemia before"],
    "E_196": ["had an operation recently", "surgery last month"], "E_113": ["kidney failure"], "E_140": ["black poo", "my stools are black"],
    "E_7": ["I don't eat well", "poor diet"], "E_26": ["anaemia runs in my family"], "E_159": ["I passed out", "blacked out", "fainted"],
    "E_94": ["chills", "shivering", "the shivers"], "E_157": ["pins and needles in my hands and feet and round my mouth"],
    "E_109": ["I've had a blood clot in my leg before", "I've had DVT"], "E_212": ["my voice has gone hoarse", "my voice sounds different"],
    "E_22": ["I have a heart valve problem"], "E_219": ["worse at night"], "E_120": ["nasal polyps"], "E_154": ["really pale"],
    "E_179": ["blood in my poo", "bright red blood in my stool"], "E_16": ["feel anxious", "really anxious", "I'm worried", "really worried"],
    "E_84": ["weak arms and legs", "my legs feel weak"], "E_172": ["my eyelids are drooping", "hard to keep my eyes open"],
    "E_207": ["I live in a city"], "E_103": ["can't smell anything", "lost my sense of smell"], "E_44": ["I take steroids"],
    "E_31": ["severe COPD"], "E_177": ["numbness or tingling", "pins and needles"], "E_167": ["I'm pregnant", "might be pregnant"],
    "E_61": ["I inject drugs"], "E_87": ["asthma runs in my family"], "E_65": ["hard to swallow", "it hurts to swallow", "food gets stuck"],
    "E_3": ["I've had pericarditis before"], "E_182": ["green snot", "yellow mucus from my nose"], "E_143": ["I exercise a lot"],
    "E_191": ["I used to smoke", "ex-smoker"], "E_2": ["I'm HIV positive"], "E_176": ["weakness in my arm or face", "one side feels weak"],
    "E_198": ["I work on a farm"], "E_121": ["deviated septum"], "E_12": ["severe food allergy"], "E_169": ["itchy nose and throat"],
    "E_42": ["I ate something I'm allergic to", "touched something I'm allergic to"], "E_203": ["bad coughing fits"],
    "E_173": ["heartburn", "burning that comes up into my throat", "acid coming up"], "E_215": ["worse after I eat"],
    "E_170": ["really itchy eyes"], "E_14": ["chest pain even when resting"], "E_210": ["I threw up blood", "vomited something like coffee grounds"],
    "E_126": ["cirrhosis of the liver"], "E_111": ["I feel like I'm dying", "terrified something awful is happening"],
    "E_81": ["chronic anxiety", "I worry all the time"], "E_25": ["cluster headaches run in my family"], "E_34": ["I have cancer"], "E_86": ["allergies and eczema run in my family"],
}


def first_person(q):
    q = q.strip().rstrip("?").rstrip(":")
    q = re.sub(r"\byour\b", "my", q, flags=re.I)
    q = re.sub(r"\byou are\b", "I am", q, flags=re.I)
    q = re.sub(r"\byou\b", "I", q, flags=re.I)
    q = re.sub(r"^(do|does|did|have|has|are|is|was)\s+I\s+", "", q, flags=re.I)
    return q[0].lower() + q[1:] if q else q


HEAD = {"back of head", "top of the head", "forehead", "occiput", "temple"}     # pain here can be said as "headache"


def natural_bits(groups, evid, rng):
    """groups: {question code: [items]} -> everyday phrases. The pain questions (E_53..E_59) become one sentence."""
    bits = []
    vals = lambda code: [it.partition("_@_")[2] for it in groups.get(code, []) if "_@_" in it]          # noqa: E731
    meaning = lambda code, v: evid[code]["value_meaning"].get(v, {}).get("en", v)                       # noqa: E731
    if "E_53" in groups or "E_55" in groups:
        where = [loc_words(meaning("E_55", v), rng) for v in vals("E_55") if meaning("E_55", v) != "nowhere"]
        kind = [rng.choice(PAIN_WORD.get(meaning("E_54", v), [meaning("E_54", v)])) for v in vals("E_54") if meaning("E_54", v) != "NA"]
        bases = {re.sub(r"\s*\((L|R)\)$", "", meaning("E_55", v).strip()) for v in vals("E_55")} - {"nowhere"}
        # pain only in the head or only in an ear: people often just say "headache" / "earache"
        word = ("headache" if bases <= HEAD else "earache" if bases == {"ear"} else None) if bases and rng.random() < 0.5 else None
        s = (rng.choice(["", "a "]) + " ".join(kind[:2]) + " " if kind else rng.choice(["", "bad ", "a "])) + (word or "pain")
        if where:
            s += " in my " + " and ".join(dict.fromkeys(where[:3]))
        sev = vals("E_56")
        if sev and sev[0].isdigit():
            n = int(sev[0])
            s += rng.choice([f", about {n} out of 10", f", {n}/10", f", maybe a {n}"]) if rng.random() < 0.8 else (", really bad" if n >= 7 else ", not too bad")
        rad = [loc_words(meaning("E_57", v), rng) for v in vals("E_57") if meaning("E_57", v) != "nowhere"]
        if rad:
            s += rng.choice([", it goes to my ", ", spreads to my ", ", it moves into my "]) + " and ".join(dict.fromkeys(rad[:2]))
        fast = vals("E_59")
        if fast and fast[0].isdigit() and rng.random() < 0.7:
            s += ", came on suddenly" if int(fast[0]) >= 6 else ", came on slowly"
        bits.append(s.strip())
    for code, items in groups.items():
        if code in ("E_53", "E_54", "E_55", "E_56", "E_57", "E_58", "E_59"):
            continue
        ev = evid[code]
        v = [it.partition("_@_")[2] for it in items if "_@_" in it]
        if code in SAY:
            bits.append(rng.choice(SAY[code]))
        elif ev["data_type"] == "B" or not v:
            bits.append(first_person(ev["question_en"]))
        else:
            words = [meaning(code, x) for x in v]
            if code == "E_204":
                bits.append("no travel abroad" if words == ["N"] else "I've been to " + ", ".join(words))
            elif code in ("E_133", "E_152"):
                bits.append(first_person(ev["question_en"]) + " " + ", ".join(loc_words(w, rng) for w in words if w != "nowhere"))
            else:
                bits.append(first_person(ev["question_en"]) + " " + ", ".join(words))
    return [b for b in bits if b]
