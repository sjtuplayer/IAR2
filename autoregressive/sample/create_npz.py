# Modified from:
#   DiT:  https://github.com/facebookresearch/DiT/blob/main/sample_ddp.py
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn.functional as F
import torch.distributed as dist

from tqdm import tqdm
import os
from PIL import Image
import numpy as np



def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

'/fuxi_team14/users/huteng/codes/LlamaGen/samples/GPT-B-0950000-size-384-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.0-seed-0'
for name in ['GPT-B-0950000','ht-GPT-B2-0940000']:
    for cfg in ['1.0','1.25','1.5','1.75']:
        sample_folder_dir=f'samples/{name}-size-384-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-{cfg}-seed-0'
        create_npz_from_sample_folder(sample_folder_dir, 50000)
        print(sample_folder_dir,"Done.")