"""Online RL driver for one lane of GPUs: rounds of (fresh practice problems -> n samples of the current model at the
profile's decoding and limit -> RLOO advantages -> one data-parallel gain update -> rebuild) until the time budget.

  python derive/rl_loop.py --tag V1 --gpus 0,1,2,3 --start START_DIR --case C [--lr 2e-3] [--minutes 210] [--kv]
      [--mix gsm8k_train=64 math_train=64 tuluif=16 mmlupro=16] [--n 8] [--exclude sets/a.json ...] [--port 29500]

Every round's problems are new, so its accuracy is an unbiased measure of that round's model (rl/<tag>/log.jsonl).
Reward = right and finished within the profile's 32768-token limit (cut = 0)."""
import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extra_items  # noqa: F401,E402
from wsp import case as C  # noqa: E402
from wsp import data  # noqa: E402
from wsp.common import MODELS  # noqa: E402

D = os.path.dirname(os.path.abspath(__file__))


def st(msg):
    with open(os.environ.get("RL_STATUS", "rl/STATUS"), "a") as f:
        f.write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + " " + msg + "\n")


def wait_free(gpus, limit_mib=2000, timeout=180):
    """Wait until the lane's GPUs have released their memory (engines and trainers of the previous step)."""
    t = time.time()
    while time.time() - t < timeout:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=30).stdout
            used = {l.split(",")[0].strip(): int(l.split(",")[1]) for l in out.strip().splitlines()}
            if all(used.get(g, 0) < limit_mib for g in gpus.split(",")):
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(5)
    return False


def run(cmd, env, log, timeout):
    with open(log, "w") as fh:
        try:
            return subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            return -9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--gpus", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--minutes", type=float, default=210)
    ap.add_argument("--kv", action="store_true")
    ap.add_argument("--layers", default="4-21")
    ap.add_argument("--mix", nargs="+", default=["gsm8k_train=64", "math_train=64", "tuluif=16", "mmlupro=16"])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--port", type=int, default=29500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rounds", type=int, default=999)
    ap.add_argument("--detrend", action="store_true", help="length-detrended rewards within each group")
    a = ap.parse_args()
    t0 = time.time()
    W = Path("rl") / a.tag
    W.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpus)
    ng = len(a.gpus.split(","))
    used = set(C.set_ids(a.case, "dev")) | set(C.set_ids(a.case, "holdout"))
    for f in a.exclude:
        used |= set(json.load(open(f)))
    rng = random.Random(1000 + a.seed)
    mix = [(s.split("=")[0], int(s.split("=")[1])) for s in a.mix]
    pools = {}
    for src, _ in mix:
        p = [r["id"] for r in data.load(f"practice:{src}") if r["id"] not in used]
        rng.shuffle(p)
        pools[src] = p
    pos = defaultdict(int)
    gains, optst, model = "none", "none", a.start
    r = 0
    # resume: continue after the last round that finished its update (each invocation may be a short job)
    done = [json.loads(l) for l in open(W / "log.jsonl")] if (W / "log.jsonl").exists() else []
    done = [x for x in done if isinstance(x.get("train"), dict)]
    if done:
        r = done[-1]["round"] + 1
        gains, optst = str(W / f"gains_{r}.npz"), str(W / f"opt_{r}.pt")
        mdir = MODELS / f"rl_{a.tag}_{r}"
        if not (mdir / "config.json").exists():
            run([sys.executable, f"{D}/rl_build.py", "--start", a.start, "--gains", gains, "--meta", str(W / "meta.json"),
                 "--out", str(mdir)], env, W / f"build_{r}.log", 900)
        model = str(mdir)
        for src, n in mix:
            pos[src] = r * n
        st(f"RL {a.tag} resume at round {r} from {gains}")
    while True:
        el = (time.time() - t0) / 60
        if el > a.minutes or r >= a.max_rounds:
            break
        ids = []
        for src, n in mix:
            P = pools[src]
            take = [P[(pos[src] + i) % len(P)] for i in range(n)]
            pos[src] += n
            ids += take
        json.dump(ids, open(W / f"ids_{r}.json", "w"))
        out = W / f"gen_{r}.jsonl"
        rc = -1
        for _ in range(2):
            wait_free(a.gpus)
            rc = run([sys.executable, f"{D}/dtgen.py", "--model", model, "--ids", str(W / f"ids_{r}.json"), "--n", str(a.n),
                      "--salt", f"rl{a.tag}{r}", "--out", str(out)], env, W / f"gen_{r}.log", 2400)
            if rc == 0 and out.exists():
                break
        if not out.exists():
            st(f"RL {a.tag} round {r} GEN FAILED rc={rc}")
            break
        rows = [json.loads(l) for l in open(out)]
        grp = defaultdict(list)
        for x in rows:
            x["rew"] = float(bool(x["correct"]) and not x["cut"])
            grp[x["id"]].append(x)
        mixed = 0
        for pid, xs in grp.items():
            n = len(xs)
            rw = [x["rew"] for x in xs]
            fin = [i for i in range(n) if not xs[i]["cut"]]
            if a.detrend and len(fin) > 2:
                # among the FINISHED answers, remove the part of the reward that is linear in the answer's length: right
                # answers are shorter, so the raw credit teaches "stop early", which does not make answers right (it
                # shortens all thinking); the residual credits what differs between answers of the same length.
                # Answers cut at the limit keep reward 0 and stay out of the fit (a fit through them would credit the
                # longest, looping answers).
                ln = {i: float(xs[i]["n_tokens"]) for i in fin}
                ml, mr = sum(ln.values()) / len(fin), sum(rw[i] for i in fin) / len(fin)
                vl = sum((ln[i] - ml) ** 2 for i in fin)
                b = sum((ln[i] - ml) * (rw[i] - mr) for i in fin) / vl if vl > 0 else 0.0
                for i in fin:
                    rw[i] = rw[i] - b * (ln[i] - ml)
            s = sum(rw)
            for i, x in enumerate(xs):
                x["adv"] = rw[i] - (s - rw[i]) / (n - 1) if n > 1 else 0.0
            mixed += 0 < sum(x["rew"] for x in xs) < n
        with open(W / f"roll_{r}.jsonl", "w") as f:
            for x in rows:
                f.write(json.dumps({k: x[k] for k in ("id", "k", "source", "text", "cut", "adv")}) + "\n")
        by = defaultdict(lambda: [0, 0, 0, 0])
        for x in rows:
            b = by[x["source"]]
            b[0] += 1; b[1] += x["rew"]; b[2] += x["cut"]; b[3] += x["n_tokens"]
        rec = {"round": r, "minutes": round(el, 1), "mixed_problems": mixed, "problems": len(grp),
               "acc": {s: round(100 * v[1] / v[0], 2) for s, v in by.items()},
               "cut": {s: round(100 * v[2] / v[0], 2) for s, v in by.items()},
               "tokens": {s: round(v[3] / v[0]) for s, v in by.items()}}
        wait_free(a.gpus)
        gout, oout, sout = W / f"gains_{r + 1}.npz", W / f"opt_{r + 1}.pt", W / f"train_{r + 1}.json"
        rc = run(["torchrun", "--nproc_per_node", str(ng), "--master_port", str(a.port), f"{D}/rl_train.py",
                  "--model-start", a.start, "--rollouts", str(W / f"roll_{r}.jsonl"), "--gains-in", gains, "--opt-in", optst,
                  "--gains-out", str(gout), "--opt-out", str(oout), "--stats-out", str(sout), "--lr", str(a.lr),
                  "--minibatches", str(a.minibatches), "--layers", a.layers, "--seed", str(r)] + (["--kv"] if a.kv else []),
                 env, W / f"train_{r + 1}.log", 1800)
        if rc != 0 or not gout.exists():
            rec["train"] = f"FAILED rc={rc}"
            open(W / "log.jsonl", "a").write(json.dumps(rec) + "\n")
            st(f"RL {a.tag} round {r} TRAIN FAILED rc={rc}: {open(W / f'train_{r + 1}.log').read()[-300:]}")
            break
        ts = json.load(open(sout))
        rec["train"] = {k: ts[k] for k in ("examples", "max_dev", "mean_abs_dev", "t")}
        gains, optst = str(gout), str(oout)
        mdir = MODELS / f"rl_{a.tag}_{r + 1}"
        rc = run([sys.executable, f"{D}/rl_build.py", "--start", a.start, "--gains", gains, "--meta", str(W / "meta.json"),
                  "--out", str(mdir)], env, W / f"build_{r + 1}.log", 900)
        if rc != 0:
            st(f"RL {a.tag} round {r} BUILD FAILED: {open(W / f'build_{r + 1}.log').read()[-300:]}")
            break
        old = MODELS / f"rl_{a.tag}_{r - 1}"
        if r >= 1 and old.exists():
            shutil.rmtree(old, ignore_errors=True)
        model = str(mdir)
        open(W / "log.jsonl", "a").write(json.dumps(rec) + "\n")
        st(f"RL {a.tag} r{r} {round(el)}m acc {rec['acc']} cut {rec['cut']} tok {rec['tokens']} mixed {mixed} "
           f"ex {ts['examples']} maxdev {ts['max_dev']:.3f}")
        r += 1
    st(f"RL {a.tag} DONE rounds={r} last_gains={gains}")


if __name__ == "__main__":
    main()
