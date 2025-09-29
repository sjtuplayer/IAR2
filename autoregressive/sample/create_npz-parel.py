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

import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def process_image(file_path):
    """
    Process a single image file and return its numpy array representation.
    """
    sample_pil = Image.open(file_path)
    sample_np = np.asarray(sample_pil).astype(np.uint8)
    return sample_np


def create_npz_from_sample_folder(sample_dir, num=50_000, max_workers=64):
    """
    Builds a single .npz file from a folder of .png samples using multithreading.
    """
    samples = []
    file_paths = [f"{sample_dir}/{i:06d}.png" for i in range(num)]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_image, file_path): file_path for file_path in file_paths}
        for future in tqdm(as_completed(futures), total=num, desc="Building .npz file from samples"):
            samples.append(future.result())

    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


base_dir = '/fuxi_team14/users/huteng/codes/LlamaGen/samples'
for name in ['GPT-B-0250000','ht-GPT-B2-0260000']:
    for cfg in [ '1.0','1.25','1.5', '1.75','2.0','2.25','2.5']:
        sample_folder_dir = os.path.join(base_dir,
                                         f'{name}-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-{cfg}-seed-0')
        create_npz_from_sample_folder(sample_folder_dir, 50000)
        print(sample_folder_dir, "Done.")