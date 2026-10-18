"""The ward card: messy intake text in, a prebaked card out. Every word on the card is a slot filled from data; nothing is
generated. Not a diagnosis.

usage: run_env.sh route_from_fields/ward_card.py "<messy intake text>"        (prints the card and its JSON)

Chain: translator (writes the patient's fields) -> counted seesaw (department + urgency planks) -> transformer router (second
reader on the text) -> verdict seesaw (safe to route automatically?) -> card.
"""
import os as _os
ROOT = _os.environ.get("TRIAGE_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # the repo folder

import json
import os
import re
import sys

import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "train"))
sys.path.insert(0, os.path.join(HERE, "..", "data"))
import common                                  # noqa: E402
import convert_ddxplus as conv                 # noqa: E402
from seesaw_router import FieldRouter          # noqa: E402
import verdict as V                            # noqa: E402

ROOT = ROOT + ""
CFG = {"translator": f"{ROOT}/runs/m_tiny_translator/model", "router": f"{ROOT}/runs/m_tiny_router/model",
       "counted": f"{ROOT}/runs/route_from_fields/counted_table.json",
       "levers": f"{ROOT}/runs/route_from_fields/verdict/levers.json",
       "evidences": f"{ROOT}/data_raw/ddxplus_en/release_evidences.json"}
CODE = re.compile(r"E_\d+(?:_@_(?:V_\d+|\d+))?")


class Chain:
    def __init__(self, cfg=CFG, device=None):
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.ttok = AutoTokenizer.from_pretrained(cfg["translator"])
        self.tmod = GPT2LMHeadModel.from_pretrained(cfg["translator"]).to(self.dev).eval()
        self.rtok = AutoTokenizer.from_pretrained(cfg["router"])
        self.rmod = GPT2LMHeadModel.from_pretrained(cfg["router"]).to(self.dev).eval()
        self.seesaw = FieldRouter(cfg["counted"])
        self.counted = json.load(open(cfg["counted"], encoding="utf-8"))
        self.levers = json.load(open(cfg["levers"], encoding="utf-8"))["levers"]
        self.evid = conv.load_evidences(cfg["evidences"])
        self.dept_ids = common.first_token_ids(self.rtok, common.DEPARTMENTS)

    @torch.no_grad()
    def translate(self, text):
        common.TASK = "translator"
        ids = self.ttok(common.prompt_of({"text": text}))["input_ids"]
        x = torch.tensor([ids], device=self.dev)
        g = self.tmod.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=320, do_sample=False,
                               pad_token_id=self.ttok.eos_token_id, eos_token_id=self.ttok.eos_token_id,
                               output_scores=True, return_dict_in_generate=True)
        new = g.sequences[0, len(ids):].tolist()
        probs = [torch.softmax(s[0].float(), -1)[t].item() for s, t in zip(g.scores, new)]
        fields, conf, acc, low = [], {}, "", 1.0
        for t, p in zip(new, probs):
            if t == self.ttok.eos_token_id:
                break
            piece = self.ttok.decode([t])
            if piece.strip().startswith("E_") and CODE.fullmatch(re.sub(r"\s*@\s*", "@", acc.strip()) or "x"):
                code = re.sub(r"\s*@\s*", "@", acc.strip())
                fields.append(code)
                conf[code] = round(low, 3)
                acc, low = "", 1.0
            acc += piece
            low = min(low, p)
        code = re.sub(r"\s*@\s*", "@", acc.strip())
        if CODE.fullmatch(code or "x"):
            fields.append(code)
            conf[code] = round(low, 3)
        return sorted(set(fields)), conf

    @torch.no_grad()
    def router_read(self, text):
        common.TASK = "router"
        ids = self.rtok(common.prompt_of({"text": text}))["input_ids"]
        logits = common.logits_at(self.rmod, torch.tensor([ids], device=self.dev), torch.ones(1, len(ids), dtype=torch.long,
                                  device=self.dev), torch.tensor([len(ids) - 1], device=self.dev))
        p = torch.softmax(logits[0, self.dept_ids].float(), -1).tolist()
        k = max(range(14), key=p.__getitem__)
        return common.DEPARTMENTS[k], p[k]

    def plain(self, code):
        name, _, val = code.partition("_@_")
        ev = self.evid.get(name)
        if not ev:
            return code
        q = ev["question_en"].strip().rstrip("?:")
        return f"{q}: {conv.answer_text(ev, val)}" if val else q

    def would_settle(self, fields, a, b, k=3):
        """unanswered yes/no questions that pull hardest between department a and department b"""
        lean = self.counted["kinds"]["department"]["lean"]
        have = {f.partition("_@_")[0] for f in fields}
        cand = [(abs(per.get(a, 0) - per.get(b, 0)), f) for f, per in lean.items()
                if "_@_" not in f and f not in have and self.evid.get(f, {}).get("data_type") == "B"]
        return [self.evid[f]["question_en"].strip() for _, f in sorted(cand, reverse=True)[:k]]

    def card(self, text):
        fields, conf = self.translate(text)
        dept = self.seesaw.route(fields, "department")
        urg = self.seesaw.route(fields, "urgency")
        r_pick, r_conf = self.router_read(text)
        known = set(self.counted["kinds"]["department"]["lean"])
        facts = {"evidence": 1.0 if any(f in known for f in fields) else 0.0,
                 "department_calm": -1.0 if dept[0]["fight"] else 1.0,
                 "router_agrees": 1.0 if r_pick == dept[0]["label"] else -1.0,
                 "translator_sure": round(2 * min(conf.values()) - 1, 4) if conf else 0.0,
                 "router_sure": round(2 * r_conf - 1, 4)}
        v = V.verdict_of(facts, self.levers)
        # no cut-offs: the two runners-up are always shown with their own lean, and so are the two most urgent planks
        heads_up = [{"department": x["label"], "gap": x["gap_to_winner"]} for x in dept[1:3]]   # force gap to the winner (NEXICOG 13 Oct)
        severe = {x["label"]: x["tilt"] for x in urg if x["label"] in ("Immediate", "Very urgent")}
        why = []
        if facts["evidence"] <= 0:
            why.append("nothing in the intake the system can use")
        else:
            why.append("the department evidence is " + ("contested" if dept[0]["fight"] else "clear"))
            why.append("a second reader " + ("agrees" if facts["router_agrees"] > 0 else f"says {r_pick}"))
            weak = [f for f, c in conf.items() if c < 0.6]
            if weak:
                why.append(f"{len(weak)} field(s) read with low confidence")
        return {
            "verdict": v["verdict"], "verdict_lean": v["tilt"],
            "route_to": dept[0]["label"] if facts["evidence"] > 0 else None,
            "urgency": urg[0]["label"] if facts["evidence"] > 0 else None,
            "heads_up": heads_up, "urgent_planks": severe, "why": why,
            "patient_said": [{"field": f, "plain": self.plain(f), "confidence": conf.get(f)} for f in fields],
            "ask_next": self.would_settle(fields, dept[0]["label"], dept[1]["label"]) if v["verdict"] not in ("yes",) else [],
            "second_reader": {"department": r_pick, "confidence": round(r_conf, 3)},
            "footer": "Not a diagnosis. Routing suggestion only. Synthetic training data (DDXPlus, CC-BY).",
        }


def render(c):
    lines = [f"VERDICT: {c['verdict'].upper()}   (lean {c['verdict_lean']:+.2f})"]
    if c["route_to"]:
        routed = c["verdict"] in ("yes", "maybe yes")
        lines.append(("ROUTE TO: " if routed else "SUGGESTION ONLY, NOT ROUTED: ") + f"{c['route_to']}   URGENCY: {c['urgency']}")
    if c["route_to"]:
        lines.append("HEADS-UP: " + ", ".join(f"{h['department']} (gap {h['gap']:.1f})" for h in c["heads_up"])
                     + "   | urgent planks: " + ", ".join(f"{k} {v:+.2f}" for k, v in c["urgent_planks"].items()))
    lines.append("WHY: " + "; ".join(c["why"]))
    lines.append("PATIENT SAID:")
    for p in c["patient_said"]:
        flag = "  <- check" if p["confidence"] is not None and p["confidence"] < 0.6 else ""
        lines.append(f"  - {p['plain']}  [{p['confidence']}]{flag}")
    if c["ask_next"]:
        lines.append("ASK NEXT:")
        lines += [f"  ? {q}" for q in c["ask_next"]]
    lines.append(c["footer"])
    return "\n".join(lines)


if __name__ == "__main__":
    chain = Chain(device=os.environ.get("PROBE_DEVICE"))
    c = chain.card(sys.argv[1])
    print(render(c))
    print(json.dumps(c, indent=1, ensure_ascii=False))
