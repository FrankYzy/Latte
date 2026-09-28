"""
Verify that `read_sampled_frames_only` in datasets/ffs_datasets.py is exact, and time it.

For every training video:
1. Frame check: all frames read by decord equal the frames read by the official
   torchvision.io.read_video path, byte for byte, and the frame counts match.
2. Sample check: with the same Python/torch seeds, FaceForensics.__getitem__ returns
   identical tensors with the option off (official) and on.
The per-sample time of __getitem__ is recorded for both settings.

CPU only. Example:
python bench/verify_ffs_frame_reading.py --config configs/ffs/ffs_train.yaml --processes 32
"""

import os
import sys
import json
import random
import argparse
import statistics
import multiprocessing as mp
from time import time, strftime

import torch
import torchvision
from omegaconf import OmegaConf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from datasets import get_dataset

_datasets = {}


def _init_worker(config_path):
    torch.set_num_threads(1)
    for flag in (False, True):
        args = OmegaConf.load(config_path)
        args.read_sampled_frames_only = flag
        _datasets[flag] = get_dataset(args)


def _check_video(index):
    official, fast = _datasets[False], _datasets[True]
    path = official.video_lists[index]
    result = {"index": index, "path": os.path.basename(path)}

    # 1. Frame check over the whole video.
    vframes, _, _ = torchvision.io.read_video(filename=path, pts_unit='sec', output_format='TCHW')
    v_reader = fast.v_decoder(path)
    result["frames_read_video"] = len(vframes)
    result["frames_decord"] = len(v_reader)
    if len(vframes) == len(v_reader):
        all_frames = torch.from_numpy(v_reader.get_batch(list(range(len(v_reader)))).asnumpy()).permute(0, 3, 1, 2)
        result["all_frames_equal"] = bool(torch.equal(vframes, all_frames))
    else:
        result["all_frames_equal"] = False
    del vframes, v_reader

    # 2. Sample check with identical seeds; also times each __getitem__.
    seed = 1000 + index
    samples = {}
    for flag, dataset in ((False, official), (True, fast)):
        random.seed(seed)
        torch.manual_seed(seed)
        start = time()
        samples[flag] = dataset[index]["video"]
        result["getitem_s_" + ("fast" if flag else "official")] = time() - start
    result["sample_equal"] = bool(torch.equal(samples[False], samples[True]))
    return result


def main(cli):
    args = OmegaConf.load(cli.config)
    num_videos = len(get_dataset(args))
    indices = list(range(num_videos)) if cli.limit <= 0 else list(range(min(cli.limit, num_videos)))

    output_dir = os.path.join(cli.output_dir, strftime("%Y%m%d-%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)

    start = time()
    ctx = mp.get_context("fork")
    with ctx.Pool(cli.processes, initializer=_init_worker, initargs=(cli.config,)) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(_check_video, indices), 1):
            results.append(r)
            if i % 50 == 0 or i == len(indices):
                print(f"checked {i}/{len(indices)}", flush=True)
    results.sort(key=lambda r: r["index"])

    with open(os.path.join(output_dir, "per_video.jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    t_official = [r["getitem_s_official"] for r in results]
    t_fast = [r["getitem_s_fast"] for r in results]
    summary = {
        "videos": len(results),
        "processes": cli.processes,
        "frame_count_mismatch": [r["path"] for r in results if r["frames_read_video"] != r["frames_decord"]],
        "all_frames_unequal": [r["path"] for r in results if not r["all_frames_equal"]],
        "sample_unequal": [r["path"] for r in results if not r["sample_equal"]],
        "getitem_official_median_s": statistics.median(t_official),
        "getitem_official_mean_s": statistics.mean(t_official),
        "getitem_fast_median_s": statistics.median(t_fast),
        "getitem_fast_mean_s": statistics.mean(t_fast),
        "speedup_mean": statistics.mean(t_official) / statistics.mean(t_fast),
        "wall_time_s": time() - start,
        "note": "getitem times are measured with all processes reading concurrently from the same storage.",
    }
    summary["passed"] = not (summary["frame_count_mismatch"] or summary["all_frames_unequal"] or summary["sample_unequal"])
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("SUMMARY " + json.dumps(summary), flush=True)
    print(f"Results written to {output_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/ffs/ffs_train.yaml")
    parser.add_argument("--output-dir", type=str, default="./results_bench/ffs_frame_reading")
    parser.add_argument("--processes", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0, help="Check only the first N videos (0 = all).")
    main(parser.parse_args())
