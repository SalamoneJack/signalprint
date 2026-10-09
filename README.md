# signalprint — LoRa RF fingerprinting (OSU dataset)

Classify which of 25 LoRa transmitters produced a burst of raw I/Q samples. See `REPORT.md` for results.

## Reproduce

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\python -m pip install numpy scipy scikit-learn matplotlib
.venv\Scripts\python download_data.py      # ~4.7 GB, resumable (byte-range subset, see below)
.venv\Scripts\python train.py              # all experiments; resumable, finished runs skipped
.venv\Scripts\python make_report.py        # figures/ + results.md
```
Or unattended: `powershell -ExecutionPolicy Bypass -File run_overnight.ps1`. Exact versions are in `requirements.txt`.

## Files
| File | Purpose |
|---|---|
| `download_data.py` | Fetches Setup 1 (Diff Days Indoor): 5 days × 25 devices × 10 recordings. Takes 0.5 s (500k complex samples) from each recording, starting 1 s in, via HTTP Range. |
| `rfdata.py` | Loader, leakage-safe split (`build_splits`), windowing, per-window RMS normalization. |
| `models.py` | `CNN1D` (baseline, 88k params), `ResCNN1D` (deeper, 794k), `SpecCNN` (STFT → 2-D CNN). |
| `train.py` | Training with GPU safety: thermal pause, fp16 autocast, OOM back-off, atomic per-epoch checkpoints, resume, early stopping. |
| `make_report.py` | Figures and the results table. |
| `splits.json` | Exact recording IDs in each split (plus the 10 recordings the server refuses to serve). |
| `runs/<exp>/` | `best.pt` (best validation), `last.pt` (resume state), `result.json`, `preds_*.npz`. |
| `logs/` | `train.log`, `thermal.log`, `gpu_monitor.csv` (temp/power/VRAM per check), `download.log`, `power_limit.log`. |

## Split protocol (no leakage)
- Unit of splitting = one recording (one `IQ_k.dat`, an independent capture). Splits are assigned to recording IDs **before** windowing, and every window is cut from inside one recording.
- Days 1–4: for each (day, device) pair, 10 recordings are shuffled with seed 0 → 6 train / 2 val / 2 test.
- Day 5: held out entirely as a **cross-day** test set (never used for training or model selection).
- Normalization is per window, so no statistics cross split boundaries. Model selection and early stopping use **validation only**.
- `build_splits` asserts pairwise-disjoint recording sets every time it runs.
