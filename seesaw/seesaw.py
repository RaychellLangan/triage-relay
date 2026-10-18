"""
seesaw.py -- an expert system rebuilt as seesaws
===============================================
Counted levers, a sum and tanh/atanh: a different arithmetic from MYCIN-style certainty factors. Plain, publishable maths only: levers, a sum, tanh/atanh.

THE SHAPE
  * Every question the expert answers is ONE SEESAW (a "beam") with two ends,
    e.g. "the supplier keeps their word" <-> "the supplier lets us down".
    Opposite answers sit on the two ends of the same plank, so they cannot both
    be believed: evidence for one IS evidence against the other.
  * Every seesaw RESTS LEVEL. Level means "don't know". Nothing tips it except
    a push.
  * A LEVER puts a fact (or another seesaw's tilt) onto a plank. A lever has a
    push for when its driver reads "yes" and a push for when it reads "no", so
    a "no" is evidence too, not just the absence of a "yes". "Probably" pushes
    half as hard as "yes"; "don't know" weighs nothing.
  * A lever may have GATES: conditions that must hold for it to push at all
    ("a line is running dry" gates the courier question). A gate lets through
    only the part of a fact that says yes, in proportion -- "probably running
    dry" lets half the push through, "don't know" lets none through. No cut-off.

THE ARITHMETIC (why it adds up reliably)
  Each push p (between -1 and +1) is turned into a turning force
  t = atanh(p), the forces are summed, and the tilt is tanh(sum).
  * Order does not matter: it is a plain sum.
  * Never past certain: tanh never reaches 1.
  * Equal and opposite pushes cancel exactly to level.
  * One push alone gives back exactly its own size, so a lever with weight 0.6
    on a "yes" fact tilts the plank to 0.6, same as a MYCIN rule of 0.6 would.
  This is ordinary log-odds adding (atanh(x) = half the log-odds of (1+x)/2).

WHAT A SEESAW REPORTS
  tilt  -- which way and how far it leans, -1..+1
  load  -- how much weight is sitting on it at all (sum of |forces|)
  torn  -- how much of that weight cancels out, 0..1
           (0 = every push agrees, 1 = perfectly balanced against itself)
  contest -- the LOSING side's weight on this plank, plus contest carried up
           from any contested plank it reads (so a question downstream of a
           fight is marked contested, not "don't know"). Contest travels down
           a lever at the same rate as the lean does.
  A plank FALLS THROUGH to a person when its contest outweighs its net push:
  the losing side is heavier than the margin it lost by. That compares two
  measured weights; no number is chosen.
  So a LEVEL plank can mean two different things, and they are kept apart:
    empty and level  -> "don't know" (silent until proven otherwise)
    loaded and level -> a genuine conflict inside the domain: this is the only
                        thing that should fall through to a person.

There are NO thresholds in the reasoning. The only numbers are the lever
weights, taken from the 30 MYCIN rules (see seesaw_rules.py for each one's
source), and the page's words ("probably", "leaning") which are for display
only and change nothing.
"""
from __future__ import annotations

import math
from typing import Dict, List, Any

EDGE = 0.999999   # atanh(1) is infinite; a push is kept a hair inside certain


def force_of(push: float) -> float:
    p = max(-EDGE, min(EDGE, push))
    return math.atanh(p)


def add_pushes(pushes: List[float]) -> float:
    """The seesaw's addition: order-free, never past certain."""
    return math.tanh(sum(force_of(p) for p in pushes))


def carry_rate(tilt: float) -> float:
    """A plank hands its lean down a lever as a tilt, which is its force
    squashed. Its contest must travel at the same rate, or a clear lean
    upstream arrives downstream looking like a fight. Level -> rate 1."""
    f = force_of(tilt)
    return tilt / f if f != 0 else 1.0


def fights(x) -> bool:
    """The losing side is heavier than the margin it lost by."""
    return x["contest"] > 0 and x["contest"] > abs(force_of(x["tilt"]))


def gate_value(spec: str, values: Dict[str, float]) -> float:
    """'name' lets through the yes-part of name; '-name' the no-part."""
    if spec.startswith("-"):
        return max(0.0, -values.get(spec[1:], 0.0))
    return max(0.0, values.get(spec, 0.0))


class Seesaws:
    def __init__(self, kb: Dict[str, Any]):
        self.kb = kb
        self.beams = {b["name"]: b for b in kb["beams"]}
        self.levers_on = {b["name"]: [] for b in kb["beams"]}
        for lv in kb["levers"]:
            self.levers_on[lv["beam"]].append(lv)
        self.order = self._settle_order()

    def _settle_order(self) -> List[str]:
        """Each seesaw after every seesaw it reads (they form no loop)."""
        done, order = set(), []

        def visit(name, path=()):
            if name in done:
                return
            if name in path:
                raise ValueError("seesaws read each other in a loop: " + " -> ".join(path + (name,)))
            for lv in self.levers_on[name]:
                for dep in [lv["driver"]] + [g.lstrip("-") for g in lv.get("gates", [])]:
                    if dep in self.beams:
                        visit(dep, path + (name,))
            done.add(name)
            order.append(name)

        for b in self.kb["beams"]:
            visit(b["name"])
        return order

    def lever_push(self, lv, values) -> float:
        d = values.get(lv["driver"], 0.0)
        push = lv["if_yes"] * d if d > 0 else lv["if_no"] * (-d)
        for g in lv.get("gates", []):
            push *= gate_value(g, values)
        return push

    def run(self, facts: Dict[str, float]) -> Dict[str, Any]:
        """facts: {fact: -1..+1}; anything missing is "don't know" (0)."""
        values = {k: max(-1.0, min(1.0, float(v))) for k, v in facts.items()}
        out = {}
        for name in self.order:
            parts = []
            for lv in self.levers_on[name]:
                p = self.lever_push(lv, values)
                if p != 0.0:
                    parts.append({"lever": lv["id"], "push": p, "force": force_of(p)})
            # math.fsum adds exactly, so equal and opposite pushes land on a
            # true zero instead of a crumb of float rounding
            total = math.fsum(x["force"] for x in parts)
            load = sum(abs(x["force"]) for x in parts)
            tilt = math.tanh(total)
            torn = 1.0 - abs(total) / load if load > 0 else 0.0
            # contest: the losing side's weight here, plus the contest carried
            # up each lever from a contested plank it reads. Facts carry none.
            # It is what keeps "nobody knows" apart from "the evidence fights".
            inherited = 0.0
            came_from: Dict[str, float] = {}           # which plank each bit of contest came from
            for lv in self.levers_on[name]:
                gates = lv.get("gates", [])
                dv = values.get(lv["driver"], 0.0)
                # each side's weight goes through its own arm of the lever
                active_arm = abs(lv["if_yes"]) if dv > 0 else abs(lv["if_no"])
                open_ = 1.0
                for g in gates:
                    open_ *= gate_value(g, values)
                if lv["driver"] in out:                       # contested driver
                    d = out[lv["driver"]]
                    losing_arm = (abs(lv["if_no"]) if d["tilt"] > 0 else abs(lv["if_yes"]) if d["tilt"] < 0
                                  else max(abs(lv["if_yes"]), abs(lv["if_no"])))
                    c = losing_arm * open_ * d["contest"] * carry_rate(d["tilt"])
                    inherited += c
                    came_from[lv["driver"]] = came_from.get(lv["driver"], 0.0) + c
                for i, g in enumerate(gates):                 # contested gate
                    gname = g.lstrip("-")
                    if gname in out:
                        rest = 1.0
                        for j, h in enumerate(gates):
                            if j != i:
                                rest *= gate_value(h, values)
                        gd = out[gname]
                        c = (active_arm * abs(dv) * rest
                             * gd["contest"] * carry_rate(gd["tilt"]))
                        inherited += c
                        came_from[gname] = came_from.get(gname, 0.0) + c
            # the losing side's weight, counted once
            losing = min(sum(x["force"] for x in parts if x["force"] > 0),
                         -sum(x["force"] for x in parts if x["force"] < 0))
            contest = losing + inherited
            values[name] = tilt
            out[name] = {"tilt": tilt, "load": load, "torn": torn, "contest": contest, "parts": parts,
                         "own_fight": losing, "came_from": came_from,
                         "for_high": sum(x["force"] for x in parts if x["force"] > 0),
                         "for_low": -sum(x["force"] for x in parts if x["force"] < 0)}
        self.values = values
        return out

    # -- the route the ball rolled ---------------------------------------
    def route(self, result, beam: str) -> List[Dict[str, Any]]:
        """Follow the heaviest push back from a seesaw to the facts that tipped
        it. Read backwards it is the path the ball rolled down."""
        path, seen, cur = [], set(), beam
        while cur in self.beams and cur not in seen:
            seen.add(cur)
            parts = result[cur]["parts"]
            if not parts:
                break
            top = max(parts, key=lambda x: abs(x["force"]))
            lv = next(l for l in self.levers_on[cur] if l["id"] == top["lever"])
            path.append({"beam": cur, "lever": lv["id"], "driver": lv["driver"], "push": top["push"]})
            cur = lv["driver"]
        return path

    # -- where a fight started, and what is on each side ---------------------
    def fight_origin(self, result, beam: str) -> str:
        """A plank can be contested because a plank it reads is. Walk back
        along the biggest share of contest until reaching the plank where the
        fight is its own (its losing side outweighs what it inherited)."""
        cur, seen = beam, set()
        while cur not in seen:
            seen.add(cur)
            x = result[cur]
            if not x["came_from"]:
                return cur
            src, amount = max(x["came_from"].items(), key=lambda kv: kv[1])
            if x["own_fight"] >= amount:
                return cur
            cur = src
        return cur

    def _facts_behind(self, result, lever_id: str, lean: float, seen=()):
        """The facts that made one push, traced down through planks. For a
        plank driver, the facts are its own pushes in the direction it leaned."""
        lv = next(l for l in self.kb["levers"] if l["id"] == lever_id)
        d = lv["driver"]
        if d not in self.beams:
            v = self.values.get(d, 0.0)
            return [{"fact": d, "value": v, "force": lean}]
        if d in seen:
            return []
        up = result[d]
        sign = 1 if up["tilt"] > 0 else -1 if up["tilt"] < 0 else 0
        out = []
        for p in up["parts"]:
            if sign == 0 or (p["force"] > 0) == (sign > 0):
                out += self._facts_behind(result, p["lever"], lean * abs(p["force"]) / max(up["load"], 1e-12),
                                          seen + (d,))
        return out

    def fight_sides(self, result, beam: str) -> Dict[str, Any]:
        """The origin of a fight and the facts on each side of it."""
        origin = self.fight_origin(result, beam)
        sides = {"high": [], "low": []}
        for p in result[origin]["parts"]:
            side = "high" if p["force"] > 0 else "low"
            sides[side] += self._facts_behind(result, p["lever"], abs(p["force"]))
        for k in sides:                                     # one line per fact, heaviest first
            merged: Dict[str, Dict[str, Any]] = {}
            for f in sides[k]:
                m = merged.setdefault(f["fact"], {"fact": f["fact"], "value": f["value"], "force": 0.0})
                m["force"] += f["force"]
            sides[k] = sorted(merged.values(), key=lambda f: -f["force"])
        return {"origin": origin, "high": sides["high"], "low": sides["low"]}

    # -- what would tip it -------------------------------------------------
    def what_would_tip(self, facts: Dict[str, float], beam: str) -> List[Dict[str, Any]]:
        """For every fact that is "don't know", how far this seesaw would move
        if it turned out yes, or no. Nothing is asked; this only says which
        unknowns matter, biggest first."""
        base = self.run(facts)[beam]["tilt"]
        out = []
        for f in self.kb["facts"]:
            n = f["name"]
            if abs(facts.get(n, 0.0)) > 0:
                continue
            up = self.run({**facts, n: 1.0})[beam]["tilt"] - base
            dn = self.run({**facts, n: -1.0})[beam]["tilt"] - base
            if abs(up) + abs(dn) > 0:
                out.append({"fact": n, "if_yes": up, "if_no": dn})
        self.run(facts)                       # leave state as the caller left it
        return sorted(out, key=lambda x: -(abs(x["if_yes"]) + abs(x["if_no"])))
