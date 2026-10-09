"""Training engine with GPU safety (thermal pause, OOM back-off, per-epoch checkpoints, resume).

Run everything:   python train.py            (resumable; finished runs are skipped)
"""
import json
import os
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

import rfdata
from models import build_model

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
LOGS = os.path.join(HERE, "logs")
TEMP_PAUSE, TEMP_RESUME = 80, 70

EXPERIMENTS = [
    # name,               model,      window L, notes
    dict(name="A_cnn1d_L256", model="cnn1d", L=256, desc="Baseline small 1-D CNN on raw I/Q, 256-sample window"),
    dict(name="B_cnn1d_L1024", model="cnn1d", L=1024, desc="Same CNN, longer 1024-sample window"),
    dict(name="C_rescnn1d_L256", model="rescnn1d", L=256, desc="Deeper residual 1-D CNN (8 res blocks), 256 window"),
    dict(name="D_spec2d_L1024", model="spec2d", L=1024, desc="Spectrogram (STFT 64/16) + 2-D CNN, 1024 window"),
    # Added after diagnosing the data: LoRa here is SF7/125 kHz (1024-sample symbols) and the main
    # fingerprint (CFO, ~kHz) is only separable from the chirp position when averaged over several symbols.
    dict(name="E_cnn1d_L4096", model="cnn1d", L=4096, desc="Same small CNN, 4096-sample window (4 LoRa symbols)"),
]
COMMON = dict(batch_size=256, steps_per_epoch=300, max_epochs=60, patience=8, lr=1e-3, wd=1e-4,
              val_per_rec=50, test_per_rec=100, seed=0, min_batch=16)


def log(msg, fname="train.log"):
    line = "[%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    with open(os.path.join(LOGS, fname), "a", encoding="utf8") as f:
        f.write(line + "\n")


# ----------------------------------------------------------------------------- GPU safety
def gpu_stats():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,memory.used,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        t, p, m, u = [float(v) for v in out.strip().split("\n")[0].split(",")]
        return dict(temp=t, power=p, mem=m, util=u)
    except Exception as e:
        log("WARNING: nvidia-smi query failed: %r" % e, "thermal.log")
        return None


def thermal_guard(where):
    s = gpu_stats()
    if s is None:
        return
    with open(os.path.join(LOGS, "gpu_monitor.csv"), "a") as f:
        f.write("%s,%s,%.0f,%.1f,%.0f,%.0f\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), where, s["temp"], s["power"], s["mem"], s["util"]))
    if s["temp"] > TEMP_PAUSE:
        t0 = time.time()
        log("THERMAL PAUSE at %s: GPU %.0fC > %dC (power %.0fW). Waiting for < %dC" % (where, s["temp"], TEMP_PAUSE, s["power"], TEMP_RESUME), "thermal.log")
        torch.cuda.synchronize()
        while True:
            time.sleep(15)
            s = gpu_stats()
            if s is not None and s["temp"] < TEMP_RESUME:
                break
        log("THERMAL RESUME after %.0fs: GPU %.0fC" % (time.time() - t0, s["temp"]), "thermal.log")


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, W, y, rec, bs=1024, n_rec=None):
    model.eval()
    logps = []
    for i in range(0, len(W), bs):
        xb = rfdata.normalise(W[i:i + bs]).cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logps.append(F.log_softmax(model(xb).float(), 1).cpu())
    lp = torch.cat(logps)
    pred = lp.argmax(1)
    win_acc = (pred == y).float().mean().item()
    # recording-level decision: sum window log-probs within each recording
    R = int(rec.max()) + 1
    agg = torch.zeros(R, lp.shape[1]).index_add_(0, rec, lp)
    rec_y = torch.zeros(R, dtype=torch.long).index_copy_(0, rec, y)
    rec_pred = agg.argmax(1)
    rec_acc = (rec_pred == rec_y).float().mean().item()
    return dict(win_acc=win_acc, rec_acc=rec_acc, pred=pred.numpy(), y=y.numpy(),
                rec_pred=rec_pred.numpy(), rec_y=rec_y.numpy())


# ----------------------------------------------------------------------------- one experiment
def save_ckpt(path, **state):
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic: a crash mid-save never corrupts the previous checkpoint


def run_experiment(cfg, data):
    name, L = cfg["name"], cfg["L"]
    d = os.path.join(RUNS, name)
    os.makedirs(d, exist_ok=True)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    model = build_model(cfg["model"]).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    scaler = torch.cuda.amp.GradScaler()
    nparams = sum(p.numel() for p in model.parameters())
    state = dict(epoch=0, best_val=-1.0, best_epoch=0, bad_epochs=0, history=[], batch_size=cfg["batch_size"])

    last = os.path.join(d, "last.pt")
    if os.path.exists(last):
        ck = torch.load(last, map_location="cuda", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        state = ck["state"]
        torch.set_rng_state(ck["rng"])
        log("%s: resumed from epoch %d (best val %.4f)" % (name, state["epoch"], state["best_val"]))
    else:
        log("%s: start. %s | params=%d" % (name, cfg["desc"], nparams))

    Xtr, ytr = data["train"]
    Wv, yv, rv = rfdata.fixed_windows(*data["val"], L, cfg["val_per_rec"])
    gen = torch.Generator().manual_seed(cfg["seed"] * 1000 + state["epoch"])

    while state["epoch"] < cfg["max_epochs"] and state["bad_epochs"] < cfg["patience"]:
        ep = state["epoch"] + 1
        thermal_guard("%s ep%d start" % (name, ep))
        t0 = time.time()
        try:
            model.train()
            bs = state["batch_size"]
            n_steps = cfg["steps_per_epoch"] * cfg["batch_size"] // bs  # constant #samples per epoch
            tot_loss, tot_correct, tot_n = 0.0, 0, 0
            for step in range(n_steps):
                xb, yb, _ = rfdata.random_windows(Xtr, ytr, bs, L, gen)
                # augmentation: random carrier phase rotation (absolute phase is arbitrary per capture)
                th = torch.rand(bs, 1, generator=gen) * 2 * np.pi
                c, s = torch.cos(th), torch.sin(th)
                xb = torch.stack([c * xb[:, 0] - s * xb[:, 1], s * xb[:, 0] + c * xb[:, 1]], 1)
                xb, yb = xb.cuda(non_blocking=True), yb.cuda(non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    out = model(xb)
                    loss = F.cross_entropy(out.float(), yb, label_smoothing=0.05)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                tot_loss += loss.item() * bs
                tot_correct += (out.argmax(1) == yb).sum().item()
                tot_n += bs
                if step % 100 == 99:
                    thermal_guard("%s ep%d step%d" % (name, ep, step + 1))
            ev = evaluate(model, Wv, yv, rv)
        except torch.cuda.OutOfMemoryError:
            new_bs = state["batch_size"] // 2
            log("%s: CUDA OOM at epoch %d with batch %d -> reducing to %d and retrying epoch from last checkpoint"
                % (name, ep, state["batch_size"], new_bs))
            xb = out = loss = None
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            if new_bs < cfg["min_batch"]:
                raise RuntimeError("batch size fell below minimum after OOM")
            if os.path.exists(last):
                ck = torch.load(last, map_location="cuda", weights_only=False)
                model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
                sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
                state = ck["state"]
            state["batch_size"] = new_bs
            continue

        sched.step(ev["win_acc"])
        peak = torch.cuda.max_memory_allocated() / 2**20
        rec = dict(epoch=ep, train_loss=tot_loss / tot_n, train_acc=tot_correct / tot_n, val_win_acc=ev["win_acc"],
                   val_rec_acc=ev["rec_acc"], lr=opt.param_groups[0]["lr"], batch=bs, sec=time.time() - t0, peak_mem_mb=peak)
        state["history"].append(rec)
        state["epoch"] = ep
        improved = ev["win_acc"] > state["best_val"]
        if improved:
            state.update(best_val=ev["win_acc"], best_epoch=ep, bad_epochs=0)
            save_ckpt(os.path.join(d, "best.pt"), model=model.state_dict(), cfg=cfg, epoch=ep, val=ev["win_acc"])
        else:
            state["bad_epochs"] += 1
        save_ckpt(last, model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                  scaler=scaler.state_dict(), state=state, rng=torch.get_rng_state())
        log("%s ep%02d loss %.3f train %.4f | val win %.4f rec %.4f | lr %.1e bs %d | %.0fs peakmem %.0fMB %s"
            % (name, ep, rec["train_loss"], rec["train_acc"], ev["win_acc"], ev["rec_acc"], rec["lr"], bs,
               rec["sec"], peak, "*best*" if improved else ""))
        gen = torch.Generator().manual_seed(cfg["seed"] * 1000 + ep)

    reason = "early stopping" if state["bad_epochs"] >= cfg["patience"] else "max epochs"
    log("%s: training finished (%s) best val %.4f @ epoch %d" % (name, reason, state["best_val"], state["best_epoch"]))

    # ---- final evaluation of the BEST checkpoint (selected on validation only)
    model.load_state_dict(torch.load(os.path.join(d, "best.pt"), map_location="cuda", weights_only=False)["model"])
    res = dict(cfg=cfg, params=nparams, best_epoch=state["best_epoch"], epochs_run=state["epoch"], stop_reason=reason,
               history=state["history"])
    for split in ("val", "test", "test_crossday"):
        W, y, r = rfdata.fixed_windows(*data[split], L, cfg["val_per_rec"] if split == "val" else cfg["test_per_rec"])
        ev = evaluate(model, W, y, r)
        res[split] = dict(win_acc=ev["win_acc"], rec_acc=ev["rec_acc"], n_windows=len(y), n_recordings=len(ev["rec_y"]))
        np.savez(os.path.join(d, "preds_%s.npz" % split), pred=ev["pred"], y=ev["y"], rec_pred=ev["rec_pred"], rec_y=ev["rec_y"])
        log("%s FINAL %-13s window acc %.4f | recording acc %.4f" % (name, split, ev["win_acc"], ev["rec_acc"]))
    with open(os.path.join(d, "result.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


def main():
    os.makedirs(LOGS, exist_ok=True)
    os.makedirs(RUNS, exist_ok=True)
    assert torch.cuda.is_available(), "CUDA not available"
    log("=== train.py start | torch %s | %s" % (torch.__version__, torch.cuda.get_device_name(0)))
    torch.backends.cudnn.benchmark = True

    splits = rfdata.build_splits(seed=0, out_path=os.path.join(HERE, "splits.json"))
    data = {}
    for k, recs in splits.items():
        data[k] = rfdata.load_recordings(recs)
        X, y = data[k]
        log("split %-13s recordings=%4d  devices=%d  samples/rec=%d  (days %s)"
            % (k, X.shape[0], len(set(y.tolist())), X.shape[1], sorted({r[0] for r in recs})))

    for exp in EXPERIMENTS:
        cfg = dict(COMMON, **exp)
        if os.path.exists(os.path.join(RUNS, cfg["name"], "result.json")):
            log("%s: already complete, skipping" % cfg["name"])
            continue
        try:
            run_experiment(cfg, data)
        except Exception:
            log("ERROR in %s (continuing with next experiment):\n%s" % (cfg["name"], traceback.format_exc()))
        finally:
            torch.cuda.empty_cache()
    log("=== all experiments attempted")


if __name__ == "__main__":
    main()
