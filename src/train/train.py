"""Train a router and stop it before it goes bang.

    run_env.sh train/train.py --arm tiny              --out runs/tiny
    run_env.sh train/train.py --arm gpt2              --out runs/gpt2       --gpt2 <stock gpt2 dir>
    run_env.sh train/train.py --arm gpt2_frozen_head  --out runs/gpt2_fh    --gpt2 <stock gpt2 dir>

THE STOPPING RULE IS MEASUREMENTS, NOT AN EPOCH COUNT. Before step 1 the start is recorded (weights, held-out answer loss,
English loss for the GPT-2 arms). At every measurement the run reports how close it is to EVERY limit at once:

    english_cost   = English loss rise as a share of the start          limit --english_allowance  (GPT-2 arms; a BUDGET)
    readout_moved  = relative distance of the word table / tied head    limit --readout_ceiling    (GPT-2 arms)
    val_rise       = held-out answer loss above the best seen so far    limit --val_allowance      (all arms; a BUDGET)
    steps          = steps taken                                        limit --max_steps
    minutes        = time spent, checked at each measurement            limit --max_minutes (optional; for paid GPU time)

The first limit crossed stops the run, and the run ROLLS BACK to the best checkpoint that was still inside every limit
(stopping AT the crossing would ship the first checkpoint known to be too far). The budgets are choices, written down as
choices; nothing in the geometry says 5%.

Also recorded at every measurement and NEVER stopped on: dark-matter ratio and participation ratio at every layer (raw
block outputs, at the cue position), mouth entropy, mouth accuracy, and the KL from the ground-truth differential to the
mouth. A reading you stop on is a reading you have fitted to.
"""
import os as _os
ROOT = _os.environ.get("TRIAGE_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # the repo folder

import argparse
import json
import os
import time

import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

import common
from english_canary import PASSAGES


def parse():
    a = argparse.ArgumentParser()
    a.add_argument("--arm", required=True, choices=["tiny", "gpt2", "gpt2_frozen_head", "hf"])
    a.add_argument("--hf_model", default="google/gemma-3-270m",
                   help="arm hf: any Hugging Face causal LM, by id or local folder (Gemma's weights are gated: accept its terms)")
    a.add_argument("--task", default="router", choices=["router", "translator"])
    a.add_argument("--set", default="ddxplus", help="data file prefix: ddxplus (converted intake) or messy")
    a.add_argument("--out", required=True)
    a.add_argument("--data", default=ROOT + "/data_built")
    a.add_argument("--gpt2", default="gpt2")
    a.add_argument("--max_steps", type=int, default=None)
    a.add_argument("--max_minutes", type=float, default=None, help="stop at the first measurement after this long, then roll back and save as usual")
    a.add_argument("--batch", type=int, default=None)
    a.add_argument("--lr", type=float, default=None)
    a.add_argument("--measure_every", type=int, default=None)
    a.add_argument("--n_val", type=int, default=1024)
    a.add_argument("--n_probe", type=int, default=256)
    a.add_argument("--english_allowance", type=float, default=0.05)
    a.add_argument("--readout_ceiling", type=float, default=0.011)
    a.add_argument("--val_allowance", type=float, default=0.02)
    a.add_argument("--warmup_measures", type=int, default=3, help="val_rise is not counted for the first N measurements")
    a.add_argument("--seed", type=int, default=0)
    a = a.parse_args()
    tiny = a.arm == "tiny"
    a.max_steps = a.max_steps or (20000 if tiny else 4000)
    a.batch = a.batch or (64 if tiny else 16)
    a.lr = a.lr or (3e-4 if tiny else 5e-5)
    a.measure_every = a.measure_every or (250 if tiny else 25 if a.arm == "hf" else 10)
    return a


def headline(m):
    if "val_dept_acc" in m:
        return (f"dept {m['val_dept_acc']:.3f} urg {m['val_urgency_acc']:.3f} H {m['val_mouth_entropy_bits']:.2f}b "
                f"KL {m['val_kl_differential_to_mouth']:.3f}")
    return f"tok {m['val_token_acc']:.3f} exact {m['val_exact']:.3f}"


def main():
    args = parse()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    common.TASK = args.task
    train = common.JsonlLines(os.path.join(args.data, f"{args.set}_train.jsonl"))
    valset = common.JsonlLines(os.path.join(args.data, f"{args.set}_validate.jsonl"))
    val_recs = valset.sample(args.n_val, seed=1)
    probe_recs = (valset.sample(args.n_probe, seed=2), valset.sample(args.n_probe, seed=4))   # (measured, calibration)

    if args.arm == "tiny":
        tok = common.build_word_tokenizer((r["text"] + " " + r.get("codes", "") for r in train.sample(50000, seed=3)),
                                          os.path.join(args.out, "tokenizer"))
        cfg = GPT2Config(vocab_size=len(tok), n_positions=768, n_embd=256, n_layer=6, n_head=8,
                         bos_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
        model = GPT2LMHeadModel(cfg)
        canary = None
    elif args.arm == "hf":
        # a pretrained family other than GPT-2 (built for Gemma 3 270M): same canary, same stops, same rollback
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.hf_model)
        model = AutoModelForCausalLM.from_pretrained(args.hf_model, dtype=torch.float32, attn_implementation="eager")
        canary = [torch.tensor([tok(p)["input_ids"]], device=dev) for p in PASSAGES]
    else:
        tok = GPT2TokenizerFast.from_pretrained(args.gpt2)
        model = GPT2LMHeadModel.from_pretrained(args.gpt2)
        canary = [torch.tensor([tok(p)["input_ids"]], device=dev) for p in PASSAGES]
        if args.arm == "gpt2_frozen_head":
            model.transformer.wte.weight.requires_grad_(False)       # tied to lm_head: the readout stays where it was
    model.to(dev)
    eos = tok.eos_token_id
    dept_ids = common.first_token_ids(tok, common.DEPARTMENTS)
    urg_ids = common.first_token_ids(tok, common.URGENCIES)
    start = {k: v.detach().float().cpu().clone() for k, v in model.named_parameters()}
    n_params = sum(p.numel() for p in model.parameters())

    log_path = os.path.join(args.out, "watch.jsonl")
    log = open(log_path, "a", encoding="utf-8")

    def record(step, loss, t0):
        m = common.measure(model, tok, val_recs, probe_recs, dept_ids, urg_ids, eos, dev, canary)
        m["readout_moved"], m["blocks_moved"] = common.relative_distance(model, start)
        m.update(step=step, train_loss=loss, minutes=round((time.time() - t0) / 60, 2))
        log.write(json.dumps(m) + "\n")
        log.flush()
        return m

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    # Gemma overflows in float16, so the hf arm trains in bfloat16 (no loss scaling needed); tiny and GPT-2 keep float16
    amp_dtype = torch.bfloat16 if args.arm == "hf" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda" and amp_dtype == torch.float16))
    t0 = time.time()
    m0 = record(0, None, t0)
    english0 = m0.get("english_loss")
    best_val = m0["val_answer_loss"]
    good = None                                    # (state on cpu, step, reading) — best checkpoint inside every limit
    n_measures = 0
    print(f"arm {args.arm} task {args.task} set {args.set}: {n_params/1e6:.1f}M params on {dev}; start val loss {best_val:.4f} "
          + headline(m0) + (f", English {english0:.3f}" if english0 else ""), flush=True)

    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(args.seed)).tolist()
    cursor, step, why = 0, 0, "it ran out of steps"
    model.train()
    while step < args.max_steps:
        idx = order[cursor:cursor + args.batch]
        cursor += args.batch
        if cursor >= len(order):
            cursor = 0
        recs = [train[i] for i in idx]
        ids, att, lab, _ = common.encode_batch(tok, recs, eos, dev)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(dev == "cuda")):
            alog, tgt = common.answer_logits(model, ids, att, lab)
            loss = torch.nn.functional.cross_entropy(alog.float(), tgt)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        step += 1

        if step % args.measure_every == 0 or step == args.max_steps:
            m = record(step, round(loss.item(), 4), t0)
            n_measures += 1
            margins = {"steps": step - args.max_steps}
            if english0 is not None:
                m["english_cost"] = (m["english_loss"] - english0) / english0
                margins["english"] = m["english_cost"] - args.english_allowance
                if args.arm == "gpt2":
                    margins["readout"] = m["readout_moved"] - args.readout_ceiling
            m["val_rise"] = (m["val_answer_loss"] - best_val) / best_val
            if n_measures > args.warmup_measures:
                margins["val"] = m["val_rise"] - args.val_allowance
            best_val = min(best_val, m["val_answer_loss"])
            inside = all(v < 0 for k, v in margins.items() if k != "steps") and margins["steps"] <= 0
            print(f"step {step:6d} loss {loss.item():.3f} val {m['val_answer_loss']:.4f} " + headline(m) + " "
                  + (f"Eng {100*m['english_cost']:+.1f}% " if english0 else "")
                  + f"head {100*m['readout_moved']:.3f}% margins " + " ".join(f"{k}{v:+.4f}" for k, v in margins.items()),
                  flush=True)
            if inside and (good is None or m["val_answer_loss"] <= good[2]["val_answer_loss"]):
                good = ({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, step, m)
            crossed = [k for k, v in margins.items() if v >= 0 and k != "steps"]
            if crossed:
                why = "crossed " + ", ".join(crossed)
                break
            if args.max_minutes and (time.time() - t0) / 60 >= args.max_minutes:
                why = f"it ran out of time ({args.max_minutes:g} min)"
                break

    print(f"stopped at step {step} after {(time.time()-t0)/60:.1f} min because {why}", flush=True)
    peak_gb = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else 0.0
    print(f"peak GPU memory allocated: {peak_gb:.2f} GB", flush=True)
    if good is not None:
        model.load_state_dict(good[0])
        print(f"rolled back to step {good[1]} (val {good[2]['val_answer_loss']:.4f}, " + headline(good[2]) + ")")
    model.save_pretrained(os.path.join(args.out, "model"))
    tok.save_pretrained(os.path.join(args.out, "model"))
    report = {"arm": args.arm, "params": n_params, "stopped_at": step, "because": why,
              "kept_step": good[1] if good else step, "kept_reading": good[2] if good else None,
              "start_reading": m0, "peak_gpu_gb": round(peak_gb, 3), "settings": vars(args)}
    json.dump(report, open(os.path.join(args.out, "report.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
