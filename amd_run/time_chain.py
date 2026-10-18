"""How fast is the whole relay per case on this machine? translator (writes the fields) + router read + seesaw routing.

usage: python time_chain.py <translator dir> <router dir> <messy file> <out.json> [n]
Runs a warm-up batch first, then times n cases in batches of 32 (translator + router on the GPU, seesaws on the CPU).
"""
import json
import os
import platform
import re
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "train"))
sys.path.insert(0, os.path.join(ROOT, "src", "route_from_fields"))
import common                                   # noqa: E402
from seesaw_router import FieldRouter           # noqa: E402

def cpu_name():
    """the processor's own name (the chain's CPU part runs on it; with TRIAGE_DEVICE=cpu, everything does)"""
    try:
        if sys.platform == "win32":
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, "HARDWARE" + chr(92) + "DESCRIPTION" + chr(92) + "System" + chr(92) + "CentralProcessor" + chr(92) + "0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor()


CODE = re.compile(r"E_\d+(?:_@_(?:V_\d+|\d+))?")


def main(tdir, rdir, data, out, n=1000):
    n = int(n)
    dev = os.environ.get("TRIAGE_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")   # TRIAGE_DEVICE=cpu: processor only
    ttok, rtok = AutoTokenizer.from_pretrained(tdir), AutoTokenizer.from_pretrained(rdir)
    ttok.padding_side = "left"
    tmod = AutoModelForCausalLM.from_pretrained(tdir).to(dev).eval()       # GPT-2 or any causal LM (the Gemma translator arm)
    rmod = AutoModelForCausalLM.from_pretrained(rdir).to(dev).eval()
    seesaw = FieldRouter(os.path.join(ROOT, "runs", "route_from_fields", "counted_table.json"))
    texts = [json.loads(l)["text"] for l in open(data, encoding="utf-8")][:n]
    dept_ids = common.first_token_ids(rtok, common.DEPARTMENTS)

    @torch.no_grad()
    def run(batch):
        common.TASK = "translator"
        enc = ttok([common.prompt_of({"text": t}) for t in batch], return_tensors="pt", padding=True).to(dev)
        enc.pop("token_type_ids", None)
        g = tmod.generate(**enc, max_new_tokens=320, do_sample=False, pad_token_id=ttok.eos_token_id, eos_token_id=ttok.eos_token_id)
        decoded = [re.sub(r"\s*@\s*", "@", ttok.decode(row[enc["input_ids"].shape[1]:], skip_special_tokens=True)) for row in g]
        common.TASK = "router"
        ids = [rtok(common.prompt_of({"text": t}))["input_ids"] for t in batch]
        w = max(len(x) for x in ids)
        x = torch.full((len(ids), w), rtok.eos_token_id, dtype=torch.long)
        att = torch.zeros_like(x)
        for j, s in enumerate(ids):
            x[j, :len(s)] = torch.tensor(s)
            att[j, :len(s)] = 1
        common.logits_at(rmod, x.to(dev), att.to(dev), torch.tensor([len(s) - 1 for s in ids], device=dev))[:, dept_ids]
        for d in decoded:
            seesaw.route(CODE.findall(d), "department")
        if dev == "cuda":
            torch.cuda.synchronize()

    run(texts[:32])                                   # warm-up
    t0 = time.time()
    for i in range(0, len(texts), 32):
        run(texts[i:i + 32])
    secs = time.time() - t0
    rep = {"machine": platform.node(), "device": dev, "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else None, "cpu": cpu_name(),
           "cpu_threads": torch.get_num_threads(), "torch": torch.__version__,
           "hip": getattr(torch.version, "hip", None), "cases": len(texts), "seconds": round(secs, 2),
           "ms_per_case": round(1000 * secs / len(texts), 2), "cases_per_second": round(len(texts) / secs, 1)}
    print(json.dumps(rep), flush=True)
    json.dump(rep, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:6])
