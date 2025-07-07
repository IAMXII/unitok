# build_waypoint_samples.py
import os, json, argparse
import numpy as np
from tqdm import tqdm


def build_sample_from_cache(seq_path, t, cache_root):
    seq_name = os.path.basename(seq_path)
    t = int(t)
    wp_file = os.path.join(seq_path, 'waypoints', f"{t:05d}.json")
    if not os.path.exists(wp_file):
        return None
    with open(wp_file) as f:
        wp_data = json.load(f)
    if len(wp_data.get("waypoints", [])) < 8:
        return None

    waypoints = [[0.0, 0.0]] + [[p["x"], p["y"]] for p in wp_data["waypoints"][:8]]
    vqcodes = []
    for i in range(9):
        npy_path = os.path.join(cache_root, seq_name, f"{t + i:05d}.npy")
        if not os.path.exists(npy_path): return None
        code = np.load(npy_path)
        vqcodes.append(json.dumps(code.tolist()))

    return {
        "data_type": "waypoint_vqa",
        "known_waypoints": waypoints[:3],
        "future_waypoints": waypoints[3:],
        "known_vqcodes": vqcodes[:3],
        "future_vqcodes": vqcodes[3:]
    }


def main(args):
    all_samples = []
    all_tasks = []
    for seq in sorted(os.listdir(args.input_pairs)):
        seq_path = os.path.join(args.input_pairs, seq)
        wp_dir = os.path.join(seq_path, 'waypoints')
        if not os.path.exists(wp_dir): continue
        total_frames = len([f for f in os.listdir(wp_dir) if f.endswith('.json')])
        for t in range(total_frames - 8):
            all_tasks.append((seq_path, t))

    chunks = np.array_split(all_tasks, args.num_chunks)[args.chunk_idx]
    pbar = tqdm(total=len(chunks))

    for seq_path, t in chunks:
        sample = build_sample_from_cache(seq_path, t, args.cache_root)
        if sample: all_samples.append(sample)
        pbar.update()

    os.makedirs(args.temp_path, exist_ok=True)
    out_file = os.path.join(args.temp_path, f"{args.chunk_idx:06d}.jsonl")
    with open(out_file, 'w') as f:
        for s in all_samples:
            f.write(json.dumps(s) + '\n')
    print(f'Saved {len(all_samples)} samples to {out_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_pairs', type=str, required=True)
    parser.add_argument('--temp_path', type=str, required=True)
    parser.add_argument('--cache_root', type=str, default='/data/vqcache')
    parser.add_argument('--chunk_idx', type=int, default=0)
    parser.add_argument('--num_chunks', type=int, default=1)
    args = parser.parse_args()
    main(args)
