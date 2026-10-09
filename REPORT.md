# Overnight report: LoRa RF fingerprinting

Run: 2026-10-08 23:03 → 2026-10-09 00:01 (training itself took ~27 min). Everything ran to completion: no crashes, no OOMs, no thermal pauses.

## Results

25-way device classification (chance = 4%). "Window" = one I/Q slice classified alone. "Recording" = all of a recording's test windows combined (summed log-probabilities). Day-5 = a day never seen in training or model selection.

| Run | Params | Epochs (best) | Val win | Test win | Test rec | Day-5 win | Day-5 rec |
|---|---|---|---|---|---|---|---|
| A_cnn1d_L256 (baseline) | 88,345 | 48 (40) | 16.5% | 16.1% | 60.1% | 16.1% | 56.4% |
| B_cnn1d_L1024 | 88,345 | 60 (58) | 57.4% | 55.3% | 84.3% | 55.7% | 85.6% |
| C_rescnn1d_L256 (deeper) | 793,881 | 55 (47) | 21.5% | 21.3% | 83.8% | 21.4% | 83.6% |
| D_spec2d_L1024 (spectrogram) | 243,833 | 60 (57) | 68.7% | 70.5% | 92.4% | 53.7% | 72.8% |
| **E_cnn1d_L4096 (selected)** | 88,345 | 60 (56) | **70.0%** | 67.5% | 83.3% | **66.7%** | 83.6% |

**Kept model: `runs/E_cnn1d_L4096/best.pt`**, chosen by validation accuracy only (the test sets were never used to pick it). It also has the best per-window accuracy on the unseen day.

![test accuracy](https://raw.githubusercontent.com/SalamoneJack/signalprint/main/figures/test_accuracy.png)
![validation curves](https://raw.githubusercontent.com/SalamoneJack/signalprint/main/figures/val_curves.png)

## What I learned

1. **Window length matters most.** The LoRa signal here is SF7 / 125 kHz, so one symbol is 1,024 samples. I measured this from the chirp slope, about 130 Hz per sample. The baseline's 256-sample window sees a quarter of a symbol. Going 256 → 1,024 → 4,096 samples took per-window accuracy from 16% → 55% → 67%. Depth helped much less: C (9× the parameters, same 256 window) reached 21%.
2. **Why: the main fingerprint is carrier frequency offset (CFO).** Each device has a stable offset of roughly −1.6 to −3.8 kHz, consistent across days (I checked this directly on the raw data). That offset is tiny next to the ±62.5 kHz chirp sweep, so it only separates from the chirp's position when averaged over several symbols. Devices with near-identical CFO (for example devices 1–4) are the ones confused most.
3. **The spectrogram model looks best in-session but generalizes worse.** D tops the same-day test (70.5% per window, 92.4% per recording) but falls to 53.7% / 72.8% on Day 5. The raw-I/Q CNNs (B, E) lose almost nothing across days. Same-day results are optimistic, and the cross-day test is the honest number.
4. **Combining windows helps a lot.** Even the weak 256-window models reach 56–84% per recording.
5. **A data gap causes a clear failure.** Device 9 has no Day 2 training data (server returns 403, see below). On Day 5, 9 of its 10 recordings are classified as device 20, making device 9 the single largest source of error (9 of E's 41 wrong recordings on Day 5). The next largest are device 1 (6, confused with devices 2–4, which have similar CFO) and device 14 (5).

![confusion, unseen day](https://raw.githubusercontent.com/SalamoneJack/signalprint/main/figures/confusion_test_crossday.png)

## Data

- **Dataset used: Oregon State University LoRa RF fingerprinting dataset** (Elmaghbub & Hamdaoui, IEEE Access 2021), **Setup 1 "Different Days, Indoor"**: 25 Pycom LoPy4 transmitters, USRP B210 receiver, 1 MS/s, 5 days × 10 recordings per device. No fallback dataset was needed.
- **Subset:** the full release is 1.2 TB (Setup 1 alone is ~200 GB), and you have 50 GB free with the server capped at ~2–4 MB/s. The transmitters emit continuously (constant envelope; I verified this), so I took **0.5 s (500k complex samples) from every recording**, starting 1 s in, using HTTP Range requests. That's 4.7 GB in total. Every device, day and recording is represented.
- **Missing:** `Day2/Device9` (10 recordings) returns HTTP 403 Forbidden from the OSU server. This is a server-side permissions issue. It's logged in `splits.json` under `missing_not_downloadable`. 1,240 of 1,250 recordings are used.
- Please cite the paper if you publish (BibTeX is in the dataset's release note).

## How leakage is prevented

- **The unit of splitting is the recording:** one `IQ_k.dat`, an independent 20 s capture. Split membership is decided on recording IDs **before** any windowing, and every window is cut from inside one recording. So no recording, and no sample, appears in two splits.
- **Days 1–4:** each (day, device)'s 10 recordings are shuffled (seed 0) into 6 train / 2 val / 2 test.
- **Day 5:** held out entirely as a cross-day test set.
- **Assertions:** `rfdata.build_splits` asserts pairwise-disjoint recording sets and that Day 5 appears only in the cross-day set. The exact IDs are saved in `splits.json`.
- **Normalization is per window** (unit RMS), so no statistics are fitted across splits.
- **Model selection and early stopping use validation only.** Test numbers were computed once, from each run's best checkpoint.

| Split | Recordings | Devices | Days | Non-overlapping 256-sample windows |
|---|---|---|---|---|
| train | 594 | 25 | 1–4 | ~1.16 M (sampled randomly, 76.8k per epoch) |
| val | 198 | 25 | 1–4 | 50 per recording used |
| test (same-day) | 198 | 25 | 1–4 | 100 per recording used |
| test_crossday | 250 | 25 | 5 | 100 per recording used |

## GPU safety log

- **Power limit: not applied.** `nvidia-smi -pl 220` and `-pl 250` both failed with "Insufficient Permissions", because the session wasn't elevated and no one was awake to approve the UAC prompt. Logged in `logs/power_limit.log`. To set it yourself, run `nvidia-smi -pl 220` from an Administrator terminal (it resets on reboot).
- **Thermal guard:** ran at every epoch start and every 100 training steps (pause above 80°C, resume below 70°C). **Peak temperature 61°C, zero pauses.** Peak power 208 W of 310 W. Full trace in `logs/gpu_monitor.csv`.
- **VRAM:** fp16 autocast, batch 256. Peak training memory was 107 MB (A) up to 1.1 GB (E). The whole GPU, including your desktop's ~1.6 GB, peaked at 2.5 GB of 8 GB.
- **OOM back-off** (halve batch, roll back to the last checkpoint, retry the epoch) was tested deliberately before the real run and works. It never fired during the real run.
- **Checkpoints:** written every epoch, atomically, to `runs/*/last.pt` (resume) and `runs/*/best.pt`. Each experiment runs in its own try/except, and `train.py` is resumable and skips finished runs.

## Caveats and next steps

- **Single seed per run.** Validation curves are noisy (±5 points epoch to epoch), so treat gaps under ~5% (for example D vs E on validation) as unresolved.
- **B, D and E hit the 60-epoch cap** with learning rate already decayed to ~3e-5, so they're essentially but not formally converged. A longer schedule might add a few points.
- **Day 5 is a harder test, but not a hard one:** it's the same room, receiver and setup. The dataset's Setups 4–7 (different distances, locations, configurations, receivers) would test real robustness.
- **Promising next steps:**
  - Combine E's long window with the spectrogram front-end and check whether it keeps D's in-session gains without D's cross-day drop.
  - Download Day 2 data for device 9 if the server permission is fixed.
  - Use more than 0.5 s per recording (8,192–16,384-sample windows look worth trying).
  - Run 3 seeds per configuration.

## Reproduce

See `README.md`. In short: `download_data.py` → `train.py` → `make_report.py`, or `run_overnight.ps1`.
