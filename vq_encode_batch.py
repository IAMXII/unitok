# vq_encode_batch.py
import os, json, argparse
from PIL import Image, ImageFile
from tqdm import tqdm
import torch
import numpy as np
from utils.config import Args
from models.unitok import UniTok
import torchvision.transforms as T
from multiprocessing.pool import ThreadPool

ImageFile.LOAD_TRUNCATED_IMAGES = False
torch.set_grad_enabled(False)

def center_crop(image, size=512):
    w, h = image.size
    scale = size / min(w, h)
    image = image.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    w, h = image.size
    left = (w - size) // 2
    top = (h - size) // 2
    return image.crop((left, top, left+size, top+size))

def collect_images(root):
    all_paths = []
    for seq in sorted(os.listdir(root)):
        img_dir = os.path.join(root, seq, 'camera/rgb_front')
        if not os.path.exists(img_dir):
            continue
        for fname in sorted(os.listdir(img_dir)):
            if fname.endswith('.jpg'):
                all_paths.append((seq, os.path.join(img_dir, fname)))
    return all_paths

def process_batch(batch_paths, vq_model, transform, cache_root, seq_name):
    images = []
    output_paths = []
    for _, img_path in batch_paths:
        try:
            img = Image.open(img_path).convert("RGB")
            if max(img.size)/min(img.size) > 2:
                continue
            img = center_crop(img)
            images.append(transform(img))
            frame_name = os.path.basename(img_path).replace('.jpg', '.npy')
            output_paths.append(os.path.join(cache_root, seq_name, frame_name))
        except:
            continue

    if not images:
        return
    batch = torch.stack(images).cuda()
    with torch.no_grad():
        codes = vq_model.img_to_idx(batch).cpu().numpy()
    for code, path in zip(codes, output_paths):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, code)

def main(args):
    device = torch.device(f'cuda:{args.gpu_id}')
    print(f"Loading VQ model on {device}")
    ckpt = torch.load(args.unitok_path, map_location='cpu')
    cfg = Args()
    cfg.load_state_dict(ckpt['args'])
    model = UniTok(cfg)
    model.load_state_dict(ckpt['trainer']['unitok'])
    model.to(device).eval()

    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3)
    ])

    all_images = collect_images(args.input_pairs)
    chunks = np.array_split(all_images, args.num_chunks)[args.chunk_idx]
    pool = ThreadPool(args.num_processes)

    print(f"Processing {len(chunks)} images in chunk {args.chunk_idx}")
    total_batches = (len(chunks) + args.batch_size - 1) // args.batch_size
    pbar = tqdm(total=total_batches, desc=f"Chunk {args.chunk_idx}")

    for i in range(0, len(chunks), args.batch_size):
        batch = chunks[i:i+args.batch_size]
        if len(batch) == 0:
            continue
        seq_name = batch[0][0]
        pool.apply_async(
            process_batch,
            args=(batch, model, transform, args.cache_root, seq_name),
            callback=lambda _: pbar.update(1)
        )

    pool.close()
    pool.join()
    pbar.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_pairs', type=str, required=True)
    parser.add_argument('--unitok_path', type=str, required=True)
    parser.add_argument('--cache_root', type=str, default='/data/vqcache')
    parser.add_argument('--chunk_idx', type=int, default=0)
    parser.add_argument('--num_chunks', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_processes', type=int, default=4)
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()
    main(args)
