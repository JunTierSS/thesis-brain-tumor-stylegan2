# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images or videos using a pretrained StyleGAN network pickle.
This version fixes:
- Grayscale saving (1 channel) vs RGB.
- Duplicate Click options (--network, --seeds, --trunc).
- Python warnings ("is not" for ints; buggy conditionals with "or 'string'").
- Robust slerp that returns numpy arrays (so image() can torch.from_numpy).
- Minor video name logic and small cleanups.
"""

import os
import subprocess
import re
from typing import List, Optional

import click
import dnnlib
import numpy as np
from numpy import linalg
import PIL.Image
import torch

import legacy
from opensimplex import OpenSimplex

# ---------------------------------------------------------------------------

class OSN:
    min = -1
    max = 1

    def __init__(self, seed: int, diameter: float):
        self.tmp = OpenSimplex(seed)
        self.d = diameter
        self.x = 0
        self.y = 0

    def get_val(self, angle: float):
        xoff = valmap(np.cos(angle), -1, 1, self.x, self.x + self.d)
        yoff = valmap(np.sin(angle), -1, 1, self.y, self.y + self.d)
        return self.tmp.noise2(xoff, yoff)


def circularloop(nf: int, d: float, seed: Optional[int], seeds: Optional[List[int]]):
    r = d / 2

    if seeds is None:
        rnd = np.random.RandomState(seed) if seed is not None else np.random
        latents_a = rnd.randn(1, 512)
        latents_b = rnd.randn(1, 512)
        latents_c = rnd.randn(1, 512)
    else:
        if len(seeds) != 3:
            raise AssertionError('Must choose exactly 3 seeds!')
        latents_a = np.random.RandomState(int(seeds[0])).randn(1, 512)
        latents_b = np.random.RandomState(int(seeds[1])).randn(1, 512)
        latents_c = np.random.RandomState(int(seeds[2])).randn(1, 512)

    latents = (latents_a, latents_b, latents_c)

    zs = []
    current_pos = 0.0
    step = 1.0 / max(1, nf)
    while current_pos < 1.0:
        zs.append(circular_interpolation(r, latents, current_pos))
        current_pos += step
    return zs


def circular_interpolation(radius: float, latents_persistent, latents_interpolate: float):
    latents_a, latents_b, latents_c = latents_persistent

    latents_axis_x = (latents_a - latents_b).flatten() / linalg.norm(latents_a - latents_b)
    latents_axis_y = (latents_a - latents_c).flatten() / linalg.norm(latents_a - latents_c)

    latents_x = np.sin(np.pi * 2.0 * latents_interpolate) * radius
    latents_y = np.cos(np.pi * 2.0 * latents_interpolate) * radius

    latents = latents_a + latents_x * latents_axis_x + latents_y * latents_axis_y
    return latents


def num_range(s: str) -> List[int]:
    """Accept either a comma separated list 'a,b,c' or a range 'a-c' and return as list of ints."""
    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    vals = s.split(',')
    return [int(x) for x in vals if x != '']


def size_range(s: str) -> List[int]:
    """Accept a range 'a-b' and return as a list [H, W]."""
    return [int(v) for v in s.split('-')][::-1]


def line_interpolate(zs, steps, easing):
    out = []
    for i in range(len(zs) - 1):
        for index in range(steps):
            t = index / float(max(1, steps))
            if easing == 'linear':
                fr = t
            elif easing == 'easeInOutQuad':
                fr = 2 * t * t if t < 0.5 else (-2 * t * t) + (4 * t) - 1
            elif easing == 'bounceEaseOut':
                if t < 4 / 11:
                    fr = 121 * t * t / 16
                elif t < 8 / 11:
                    fr = (363 / 40.0 * t * t) - (99 / 10.0 * t) + 17 / 5.0
                elif t < 9 / 10:
                    fr = (4356 / 361.0 * t * t) - (35442 / 1805.0 * t) + 16061 / 1805.0
                else:
                    fr = (54 / 5.0 * t * t) - (513 / 25.0 * t) + 268 / 25.0
            elif easing == 'circularEaseOut':
                fr = np.sqrt((2 - t) * t)
            elif easing == 'circularEaseOut2':
                fr = np.sqrt(np.sqrt((2 - t) * t))
            elif easing == 'backEaseOut':
                p = 1 - t
                fr = 1 - (p * p * p - p * np.sin(p * np.pi))
            else:
                fr = t
            out.append(zs[i + 1] * fr + zs[i] * (1 - fr))
    return out


def noiseloop(nf: int, d: float, seed: int):
    if seed:
        np.random.RandomState(seed)
    features = [OSN(i + seed, d) for i in range(512)]
    inc = (np.pi * 2) / max(1, nf)

    zs = []
    for f in range(nf):
        z = np.random.randn(1, 512)
        for i in range(512):
            z[0, i] = features[i].get_val(inc * f)
        zs.append(z)
    return zs


def save_image_from_tensor(img_tensor: torch.Tensor, out_path: str):
    """Save [N,H,W,C] uint8 tensor as PNG, handling 1 or 3 channels."""
    arr = img_tensor[0].cpu().numpy()  # HWC
    if arr.ndim == 3 and arr.shape[2] == 1:
        # grayscale
        PIL.Image.fromarray(arr[:, :, 0], mode='L').save(out_path)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        PIL.Image.fromarray(arr, mode='RGB').save(out_path)
    elif arr.ndim == 2:
        PIL.Image.fromarray(arr, mode='L').save(out_path)
    else:
        # Fallback: if single-channel CHW slipped through
        if arr.ndim == 3 and arr.shape[0] == 1:
            PIL.Image.fromarray(arr[0], mode='L').save(out_path)
        else:
            raise ValueError(f"Unsupported image shape for saving: {arr.shape}")


def images(G, device, inputs, space, truncation_psi, label, noise_mode, outdir, start=None, stop=None):
    tp = start
    tp_i = None
    if start is not None and stop is not None:
        tp_i = (stop - start) / max(1, len(inputs))

    for idx, i in enumerate(inputs):
        print(f'Generating image for frame {idx}/{len(inputs)} ...')
        if space == 'z':
            if isinstance(i, torch.Tensor):
                z = i.to(device)
            else:
                z = torch.from_numpy(i).to(device)
            if tp is not None and tp_i is not None:
                img = G(z, label, truncation_psi=tp, noise_mode=noise_mode)
                tp += tp_i
            else:
                img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        else:
            # space == 'w'
            if not isinstance(i, torch.Tensor):
                i = torch.from_numpy(i).to(device)
            if i.ndim == 2:
                i = i.unsqueeze(0)
            img = G.synthesis(i, noise_mode=noise_mode, force_fp32=True)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        save_image_from_tensor(img, f'{outdir}/frame{idx:04d}.png')


def interpolate(G, device, projected_w, seeds, random_seed, space, truncation_psi, label, frames, noise_mode, outdir, interpolation, easing, diameter, start=None, stop=None):
    if interpolation in ('noiseloop', 'circularloop'):
        if seeds is not None:
            print(f'Warning: interpolation type "{interpolation}" ignores --seeds and uses random/circular seeds.')
        if interpolation == 'noiseloop':
            points = noiseloop(frames, diameter, random_seed or 0)
        else:
            points = circularloop(frames, diameter, random_seed or 0, seeds)
    else:
        if projected_w is not None:
            points = np.load(projected_w)['w']
        else:
            points = seeds_to_zs(G, seeds)
            if space == 'w':
                points = zs_to_ws(G, device, label, truncation_psi, points)
        if interpolation == 'linear':
            points = line_interpolate(points, frames, easing)
        elif interpolation == 'slerp':
            points = slerp_interpolate(points, frames)

    images(G, device, points, space, truncation_psi, label, noise_mode, outdir, start, stop)


def seeds_to_zs(G, seeds):
    zs = []
    for seed in seeds or []:
        z = np.random.RandomState(int(seed)).randn(1, G.z_dim)
        zs.append(z)
    return zs


# Spherical linear interpolation returning NUMPY arrays (so images() can torch.from_numpy)
# Accepts numpy or torch inputs; converts to numpy internally.

def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    def to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    v0 = to_np(v0)
    v1 = to_np(v1)

    v0_copy = np.copy(v0)
    v1_copy = np.copy(v1)

    v0n = v0 / np.linalg.norm(v0)
    v1n = v1 / np.linalg.norm(v1)
    dot = np.sum(v0n * v1n)

    if np.abs(dot) > DOT_THRESHOLD:
        return (1.0 - t) * v0_copy + t * v1_copy

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta_t = theta_0 * t
    s0 = np.sin(theta_0 - theta_t) / sin_theta_0
    s1 = np.sin(theta_t) / sin_theta_0
    v2 = s0 * v0_copy + s1 * v1_copy
    return v2


def slerp_interpolate(zs, steps):
    out = []
    for i in range(len(zs) - 1):
        for index in range(steps):
            fraction = index / float(max(1, steps))
            out.append(slerp(fraction, zs[i], zs[i + 1]))
    return out


def truncation_traversal(G, device, z_seeds, label, start, stop, increment, noise_mode, outdir):
    count = 1
    trunc = start

    z = seeds_to_zs(G, z_seeds)[0]
    z = torch.from_numpy(np.asarray(z)).to(device)

    while trunc <= stop:
        print(f'Generating truncation {trunc:0.2f}')
        img = G(z, label, truncation_psi=trunc, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        save_image_from_tensor(img, f'{outdir}/frame{count:04d}.png')
        trunc += increment
        count += 1


def valmap(value, istart, istop, ostart, ostop):
    return ostart + (ostop - ostart) * ((value - istart) / (istop - istart))


def zs_to_ws(G, device, label, truncation_psi, zs):
    ws = []
    for z in zs:
        zt = torch.from_numpy(z).to(device)
        w = G.mapping(zt, label, truncation_psi=truncation_psi, truncation_cutoff=8)
        ws.append(w)
    return ws

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=num_range, help='List of random seeds')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1.0, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--diameter', type=float, help='Diameter for circular/noise loops', default=100.0, show_default=True)
@click.option('--frames', type=int, help='Frames between keypoints / total for loops', default=240, show_default=True)
@click.option('--fps', type=int, help='Framerate for video', default=24, show_default=True)
@click.option('--increment', type=float, help='Truncation increment value (traversal)', default=0.01, show_default=True)
@click.option('--interpolation', type=click.Choice(['linear', 'slerp', 'noiseloop', 'circularloop']), default='linear', help='Interpolation type', required=True)
@click.option('--easing', type=click.Choice(['linear', 'easeInOutQuad', 'bounceEaseOut','circularEaseOut','circularEaseOut2','backEaseOut']), default='linear', help='Easing', required=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--outdir', help='Where to save the output images/video', type=str, required=True, metavar='DIR')
@click.option('--process', type=click.Choice(['image', 'interpolation','truncation','interpolation-truncation']), default='image', help='Generation method', required=True)
@click.option('--projected-w', help='Projection result file (npz with w)', type=str, metavar='FILE')
@click.option('--random_seed', type=int, help='Random seed for loops', default=0, show_default=True)
@click.option('--scale-type', type=click.Choice(['pad', 'padside', 'symm','symmside']), default='pad', help='Scaling method for --size')
@click.option('--size', type=size_range, help='Output size (format x-y)')
@click.option('--space', type=click.Choice(['z', 'w']), default='z', help='Latent space', required=True)
@click.option('--start', type=float, help='Starting truncation value (traversal)', default=0.0, show_default=True)
@click.option('--stop', type=float, help='Stopping truncation value (traversal)', default=1.0, show_default=True)

def generate_images(
    ctx: click.Context,
    easing: str,
    interpolation: str,
    increment: Optional[float],
    network_pkl: str,
    process: str,
    random_seed: Optional[int],
    diameter: Optional[float],
    scale_type: Optional[str],
    size: Optional[List[int]],
    seeds: Optional[List[int]],
    space: str,
    fps: Optional[int],
    frames: Optional[int],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    class_idx: Optional[int],
    projected_w: Optional[str],
    start: Optional[float],
    stop: Optional[float],
):
    """Generate images/videos using a pretrained network pickle.

    Examples:

    \b
    # Generate images without truncation
    python generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 \
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate with truncation psi
    python generate.py --outdir=out --trunc=0.7 --seeds=600-605 \
        --network=metfaces.pkl

    \b
    # Class-conditional (e.g., class 1)
    python generate.py --outdir=out --seeds=0-35 --class=1 \
        --network=cifar10.pkl
    """

    # Custom size support
    if size:
        print('Render custom size:', size)
        print('Padding method:', scale_type)
        custom = True
    else:
        custom = False

    G_kwargs = dnnlib.EasyDict()
    G_kwargs.size = size
    G_kwargs.scale_type = scale_type

    print(f'Loading networks from "{network_pkl}"...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f, custom=custom, **G_kwargs)['G_ema'].to(device)  # type: ignore

    os.makedirs(outdir, exist_ok=True)

    # Projected W branch
    if process == 'image' and projected_w is not None:
        if seeds is not None:
            print('Warning: --seeds is ignored when using --projected-w')
        print(f'Generating images from projected W "{projected_w}"')
        ws = np.load(projected_w)['w']
        ws = torch.tensor(ws, device=device)
        assert ws.shape[1:] == (G.num_ws, G.w_dim)
        for idx, w in enumerate(ws):
            img = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            save_image_from_tensor(img, f'{outdir}/proj{idx:02d}.png')
        return

    # Labels for conditional G
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            ctx.fail('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    elif class_idx is not None:
        print('warn: --class ignored on unconditional network')

    if process == 'image':
        if not seeds:
            ctx.fail('--seeds is required when not using --projected-w')
        for si, seed in enumerate(seeds):
            print(f'Generating image for seed {seed} ({si+1}/{len(seeds)}) ...')
            z = torch.from_numpy(np.random.RandomState(int(seed)).randn(1, G.z_dim)).to(device)
            img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            save_image_from_tensor(img, f'{outdir}/seed{int(seed):04d}.png')

    elif process in ('interpolation', 'interpolation-truncation'):
        dirpath = os.path.join(outdir, 'frames')
        os.makedirs(dirpath, exist_ok=True)

        if seeds is not None:
            seedstr = '_'.join([str(int(s)) for s in seeds])
            vidname = f'{process}-{interpolation}-seeds_{seedstr}-{fps}fps'
        elif interpolation in ('noiseloop', 'circularloop'):
            vidname = f'{process}-{interpolation}-{diameter}dia-seed_{random_seed}-{fps}fps'
        else:
            vidname = f'{process}-{interpolation}-{fps}fps'

        if process == 'interpolation-truncation':
            interpolate(G, device, projected_w, seeds, random_seed, space, truncation_psi, label, frames, noise_mode, dirpath, interpolation, easing, diameter, start, stop)
        else:
            interpolate(G, device, projected_w, seeds, random_seed, space, truncation_psi, label, frames, noise_mode, dirpath, interpolation, easing, diameter)

        cmd = f'ffmpeg -y -r {fps} -i {dirpath}/frame%04d.png -vcodec libx264 -pix_fmt yuv420p {outdir}/{vidname}.mp4'
        subprocess.call(cmd, shell=True)

    elif process == 'truncation':
        if not seeds or len(seeds) != 1:
            ctx.fail('truncation requires a single seed value (use --seeds=123)')
        dirpath = os.path.join(outdir, 'frames')
        os.makedirs(dirpath, exist_ok=True)
        seed = seeds[0]
        vidname = f'{process}-seed_{seed}-start_{start}-stop_{stop}-inc_{increment}-{fps}fps'
        truncation_traversal(G, device, seeds, label, start, stop, increment, noise_mode, dirpath)
        cmd = f'ffmpeg -y -r {fps} -i {dirpath}/frame%04d.png -vcodec libx264 -pix_fmt yuv420p {outdir}/{vidname}.mp4'
        subprocess.call(cmd, shell=True)

#----------------------------------------------------------------------------

if __name__ == '__main__':
    generate_images()  # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
