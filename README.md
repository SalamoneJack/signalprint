# signalprint: identifying radio transmitters from raw RF signals

Every radio transmitter has tiny manufacturing imperfections that leave a unique "fingerprint" in the signal it emits. This project trains neural networks to tell **25 physically identical LoRa IoT devices** apart using only their raw I/Q samples. It uses a real over-the-air dataset from Oregon State University, with an evaluation designed to avoid the data leakage that inflates many results in this field.

**Result:** the selected model identifies the transmitter from a single 4 ms signal slice **67% of the time on a day it never saw during training** (chance: 4%). Combining a recording's slices raises that to **84% per recording**.

![Test accuracy per experiment](https://raw.githubusercontent.com/SalamoneJack/signalprint/main/figures/test_accuracy.png)

| Experiment | What changed | Same-day test (per slice) | Unseen day (per slice) | Unseen day (per recording) |
|---|---|---|---|---|
| A | Baseline small 1-D CNN, 256-sample input | 16.1% | 16.1% | 56.4% |
| B | Same CNN, 1,024-sample input | 55.3% | 55.7% | 85.6% |
| C | Deeper residual CNN (9× parameters), 256-sample input | 21.3% | 21.4% | 83.6% |
| D | Spectrogram + 2-D CNN, 1,024-sample input | **70.5%** | 53.7% | 72.8% |
| **E (selected)** | Same small CNN as A, 4,096-sample input | 67.5% | **66.7%** | 83.6% |

The selected model was chosen on validation accuracy alone; test sets were never used for selection.

## Improvement over baseline

The selected model (E) and the baseline (A) are **the same small network**: 88k parameters, same training settings, same data. The only change is the length of signal in each input slice: 4,096 samples instead of 256. On a day neither model saw during training:

| Measure (unseen day) | Baseline A | Selected E | Improvement |
|---|---|---|---|
| Single-slice accuracy | 16.1% | 66.7% | **+50.6 pts (4.1×)** |
| Single-slice error rate | 83.9% | 33.3% | 60% fewer errors |
| Recordings identified correctly | 141 of 250 | 209 of 250 | +68 recordings |
| Recording error rate | 43.6% | 16.4% | 62% fewer errors |

**What this represents.**
- **Single slice:** the model sees one short piece of a transmission and must name which of 25 identical devices sent it. Random guessing is right 4% of the time; the baseline is about 4× better than chance, and the selected model about 17×.
- **Recording:** the model combines its predictions across a transmission, as a deployed system listening for a few seconds would.
- **Why it improved:** almost the entire gain comes from one finding about the signal, not from model capacity. Each device's fingerprint is a small frequency offset that only becomes visible across several LoRa symbols, and 256 samples is a quarter of one symbol. A network with 9× more parameters on the short input (experiment C) gained only 5 points.

**Caveat:** at the recording level, E sees more signal than A (100 slices of 4,096 samples versus 100 of 256), so part of the recording-level gain comes from seeing more data. The single-slice comparison is the headline result for that reason.

## Key findings

- **Understanding the signal beat adding model capacity.** Analysis of the raw data showed the strongest fingerprint is each device's carrier frequency offset: a stable ~2–4 kHz shift, tiny next to the ±62.5 kHz LoRa chirp. It only becomes separable when averaged across several LoRa symbols, each 1,024 samples here. Widening the input from 256 to 4,096 samples raised accuracy from 16% to 67%. A model with 9× more parameters on the short input reached only 21%.
- **Same-session accuracy can mislead.** The spectrogram model scored highest on held-out recordings from training days (70.5%) but dropped to 53.7% on a new day. The raw-I/Q CNNs lost almost nothing. Evaluating on an unseen capture day exposed this; a random split would have hidden it.
- **Errors trace back to physics and data gaps.** Remaining confusions are between devices with near-identical frequency offsets, plus one device missing a day of training data because of a server-side outage. Details are in [REPORT.md](REPORT.md).

## Evaluation design (no data leakage)

Splitting windows randomly lets slices of the same recording land in both training and test sets, which inflates accuracy. Here:

- **The unit of splitting is the recording**, an independent capture. Recordings are assigned to splits *before* being cut into windows, so no recording contributes to more than one split. This is checked by assertions every run.
- **Days 1–4:** each device's recordings are split 6 train / 2 validation / 2 test per day.
- **Day 5:** held out entirely as an unseen-day test set.
- **Normalization is per window**, so no statistics are shared across splits. Early stopping and model selection use validation only.

## Engineering

Built to run unattended overnight on a single 8 GB consumer GPU (RTX 3070 Ti):

- **Thermal and OOM safety:** GPU temperature is polled during training, with a pause above 80°C and resume below 70°C. Mixed-precision (fp16) training keeps memory low. On out-of-memory errors, the batch size is halved automatically and the epoch is retried from the last checkpoint.
- **Crash-safe checkpoints:** written atomically every epoch, so runs resume where they stopped, and each experiment is isolated so one failure doesn't stop the rest.
- **Efficient data access:** the full dataset is 1.2 TB. The downloader uses HTTP Range requests to fetch a representative 4.7 GB slice covering every device, day and recording.
- All five experiments trained in about 27 minutes. GPU peak was 61°C with no thermal pauses. Telemetry is in `logs/gpu_monitor.csv`.

## Reproduce

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\python -m pip install numpy scipy scikit-learn matplotlib
.venv\Scripts\python download_data.py      # ~4.7 GB, resumable
.venv\Scripts\python train.py              # all experiments; resumable, finished runs skipped
.venv\Scripts\python make_report.py        # figures/ + results.md
```
Or unattended: `powershell -ExecutionPolicy Bypass -File run_overnight.ps1`. Exact versions are in `requirements.txt`.

| File | Purpose |
|---|---|
| `download_data.py` | Fetches OSU Setup 1 (Different Days, Indoor): 0.5 s from each of 5 days × 25 devices × 10 recordings, via HTTP Range. |
| `rfdata.py` | I/Q loader, leakage-safe split (`build_splits`), windowing, normalization. |
| `models.py` | `CNN1D` (88k params), `ResCNN1D` (794k), `SpecCNN` (STFT → 2-D CNN). |
| `train.py` | Training loop with thermal guard, fp16, OOM back-off, checkpoints, resume, early stopping. |
| `make_report.py` | Figures and results table. |
| `splits.json` | Exact recording IDs in each split. |
| `runs/<experiment>/` | `best.pt` weights, `result.json` (metrics and training history), `preds_*.npz`. |
| `logs/` | Training log, GPU telemetry, download and environment logs. |

## Limitations

- Each configuration was trained once (single seed); differences under ~5 points between top models are not conclusive.
- The unseen-day test shares the room, receiver and settings with training. The dataset's other setups (different distances, locations and receivers) would be a harder robustness test.

## Dataset and citation
This project uses the Oregon State University LoRa RF fingerprinting dataset (NetSTAR Lab), available at
<http://research.engr.oregonstate.edu/hamdaoui/datasets>. The raw data is **not** included in this repository;
`download_data.py` fetches it from OSU. The dataset authors ask that any publication using it cite:

```bibtex
@article{elmaghbub2021lora,
  title={{LoRa} Device Fingerprinting in the Wild: Disclosing {RF} Data-Driven Fingerprint Sensitivity to Deployment Variability},
  author={Elmaghbub, Abdurrahman and Hamdaoui, Bechir},
  journal={IEEE Access},
  volume={9},
  pages={142893--142909},
  year={2021},
  publisher={IEEE}
}
```

## License
Code in this repository is released under the MIT License (see `LICENSE`). The dataset is subject to its authors' own terms.
