"""Download a byte-range subset of the OSU LoRa RF fingerprinting dataset (Setup 1: Diff Days Indoor).

Each full recording is 20 s @ 1 MS/s complex64 (160 MB). The LoRa transmitters emit continuously
(constant envelope across the whole capture), so we fetch only a window of each recording via
HTTP Range requests. The download is resumable: completed files are skipped, partial ones retried.

Usage: python download_data.py [--offset-bytes N] [--nbytes N] [--workers K]
"""
import argparse
import concurrent.futures as cf
import json
import os
import time
import urllib.request

BASE = "https://research.engr.oregonstate.edu/hamdaoui/RFFP-dataset/LoRa-Dataset/Diff_Days_Indoor_Setup"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "osu_lora_setup1")


def fetch(url, dst, byte_range=None, retries=5):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url)
            if byte_range:
                req.add_header("Range", "bytes=%d-%d" % byte_range)
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if byte_range and len(data) != byte_range[1] - byte_range[0] + 1:
                raise IOError("short read %d" % len(data))
            tmp = dst + ".part"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, dst)
            return True
        except Exception as e:  # network errors: back off and retry
            print("  retry %d %s: %s" % (attempt + 1, url, e), flush=True)
            time.sleep(5 * (attempt + 1))
    return False


def job(day, dev, tx, offset, nbytes):
    d = os.path.join(ROOT, "Day%d" % day, "Device%d" % dev)
    os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, "IQ_%d.dat" % tx)
    meta = os.path.join(d, "IQ_%d.sigmf-meta" % tx)
    url = "%s/Day%d/Device%d/IQ_%d" % (BASE, day, dev, tx)
    ok = True
    if not (os.path.exists(dst) and os.path.getsize(dst) == nbytes):
        ok &= fetch(url + ".dat", dst, (offset, offset + nbytes - 1))
    if not os.path.exists(meta):
        ok &= fetch(url + ".sigmf-meta", meta)
    return (day, dev, tx, ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offset-bytes", type=int, default=8_000_000)  # skip first 1 s (1e6 samples * 8 B)
    ap.add_argument("--nbytes", type=int, default=4_000_000)        # 0.5 s = 500k complex samples
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(ROOT, exist_ok=True)
    with open(os.path.join(ROOT, "subset_info.json"), "w") as f:
        json.dump({"source": BASE, "offset_bytes": a.offset_bytes, "nbytes": a.nbytes,
                   "note": "Each .dat is a contiguous byte-range slice of the original 160 MB recording "
                           "(interleaved float32 I,Q)."}, f, indent=2)
    jobs = [(day, dev, tx) for day in range(1, 6) for dev in range(1, 26) for tx in range(1, 11)]
    t0, done, failed = time.time(), 0, []
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(job, *j, a.offset_bytes, a.nbytes) for j in jobs]
        for fu in cf.as_completed(futs):
            day, dev, tx, ok = fu.result()
            done += 1
            if not ok:
                failed.append((day, dev, tx))
            if done % 25 == 0:
                print("%d/%d done, %d failed, %.0fs elapsed" % (done, len(jobs), len(failed), time.time() - t0), flush=True)
    print("FINISHED: %d ok, %d failed: %s" % (len(jobs) - len(failed), len(failed), failed), flush=True)


if __name__ == "__main__":
    main()
