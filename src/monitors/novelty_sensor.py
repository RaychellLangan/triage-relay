"""The novelty sensor: "have I seen anything like this?", read from a router's middle layers. Nothing learned.

    sensor = NoveltySensor(router_dir)          # builds its memory from training texts
    nov, fact = sensor.read(texts)             # raw novelty per text, and the verdict FACT (-1 new .. +1 familiar)
    sensor.remember(texts)                     # learn-as-it-goes: add these cases' states to the memory bank

READING  each middle block's cue state, z-scored on a calibration set; novelty = 1 - mean cosine to the 5 nearest states in
         the memory bank, AVERAGED over the middle blocks (no layer is picked, so nothing is chosen on test data).
FACT     where that novelty sits among ordinary training cases (a reference set read the same way, not in the bank):
         fact = 1 - 2 * (share of reference cases less novel than this one). +1 = as familiar as the most ordinary case,
         -1 = newer than anything ordinary. Counted once from the reference set, then frozen.
"""
import os as _os
ROOT = _os.environ.get("TRIAGE_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # the repo folder

import json
import os
import random
import sys

import numpy as np
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "train"))
import common                                  # noqa: E402
from novelty import read_states                # noqa: E402

R = ROOT + ""


def _sha1(path):
    import hashlib
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class NoveltySensor:
    def __init__(self, router_dir, device=None, n_calib=1000, n_bank=4000, n_ref=1000, seed=17, bank_file=None, pick_rate=0.015,
                 cache_file=None):
        """pick_rate: the share of the training file drawn (seeded). A file already written by pick_bank.py holds exactly the
        cases this draw takes, in order, so it is read with pick_rate=1 and gives the same sensor (checked: identical readings).
        cache_file: the finished arrays (calibration, bank, reference) saved once, keyed by the sha1 of the router weights, the
        bank file and these settings. A matching cache is loaded instead of reading 6,000 cases through the router (about
        2 minutes and 1 GB of working memory on a CPU); anything that does not match is rebuilt. Same arrays, same readings."""
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        common.TASK = "router"
        self.tok = AutoTokenizer.from_pretrained(router_dir)
        self.model = GPT2LMHeadModel.from_pretrained(router_dir).to(self.dev).eval()
        bank_file = bank_file or f"{R}/data_built/messy_train.jsonl"
        key = "|".join([_sha1(os.path.join(router_dir, "model.safetensors")), _sha1(bank_file),
                        f"{n_calib},{n_bank},{n_ref},{seed},{pick_rate}"])
        if cache_file and os.path.exists(cache_file):
            z = np.load(cache_file)
            if str(z["key"]) == key:
                self.mu, self.sd, self.bank, self.ref = z["mu"], z["sd"], z["bank"], z["ref"]
                self.mid = list(range(2, self.mu.shape[0] - 1))
                return
        self._build(bank_file, n_calib, n_bank, n_ref, seed, pick_rate)
        if cache_file:
            try:
                np.savez(cache_file, key=np.array(key), mu=self.mu, sd=self.sd, bank=self.bank, ref=self.ref)
            except OSError:          # a read-only install keeps working; it just rebuilds at each start
                pass

    def _build(self, bank_file, n_calib, n_bank, n_ref, seed, pick_rate):
        rng, texts = random.Random(seed), []
        for l in open(bank_file, encoding="utf-8"):
            if rng.random() < pick_rate:
                texts.append(json.loads(l)["text"])
            if len(texts) >= n_calib + n_bank + n_ref:
                break
        calib = self._states(texts[:n_calib])
        self.mu, self.sd = calib.mean(0), calib.std(0) + 1e-6
        self.mid = list(range(2, calib.shape[1] - 1))            # blocks between the first and the last: the middle
        self.bank = self._unit(self._z(self._states(texts[n_calib:n_calib + n_bank])))
        self.ref = np.sort(self._novelty(self._unit(self._z(self._states(texts[n_calib + n_bank:])))))

    def _states(self, texts):
        common.TASK = "router"
        return read_states(self.model, self.tok, texts, self.dev)[0][:, :, :]

    def _z(self, S):
        return (S - self.mu) / self.sd

    @staticmethod
    def _unit(Z):
        return Z / (np.linalg.norm(Z, axis=-1, keepdims=True) + 1e-9)

    def _novelty(self, U):
        per = []
        for l in self.mid:
            sims = U[:, l] @ self.bank[:, l].T
            per.append(1 - np.sort(sims, -1)[:, -5:].mean(-1))
        return np.mean(per, 0)

    def read(self, texts):
        nov = self._novelty(self._unit(self._z(self._states(texts))))
        below = np.searchsorted(self.ref, nov) / len(self.ref)
        return nov, 1 - 2 * below

    def remember(self, texts):
        """learn-as-it-goes: the states of these cases join the memory bank (the bank grows; nothing else changes)"""
        self.bank = np.concatenate([self.bank, self._unit(self._z(self._states(texts)))], 0)
