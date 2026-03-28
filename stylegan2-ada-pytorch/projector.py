# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""

import copy
import os
from time import perf_counter

import click
import imageio
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

import dnnlib
import legacy

def project(
    G,
    target: torch.Tensor,  # [C,H,W] en rango [0,255] uint8 o float
    *,
    num_steps: int = 1000,
    w_avg_samples: int = 10000,
    initial_learning_rate: float = 0.1,
    initial_noise_factor: float = 0.05,
    lr_rampdown_length: float = 0.25,
    lr_rampup_length: float = 0.05,
    noise_ramp_length: float = 0.75,
    regularize_noise_weight: float = 1e5,
    verbose: bool = False,
    device: torch.device,
):
    """
    Projecta una imagen 'target' al espacio W/W+ de un StyleGAN2-ADA.

    Ajustes clave para tu caso:
      - Acepta target en [0,255] (uint8 o float) y lo normaliza a [-1,1].
      - Soporta G.img_channels = 1 o 3.
      - Para VGG (LPIPS), siempre usa imágenes con 3 canales:
          * Si C=1 -> repite a 3 canales.
          * Si C=3 -> usa tal cual.
      - Para redes condicionales (c_dim > 0), llama a G.mapping(z, c)
        con labels en cero (clase 0) al calcular w_avg y w_std.
      - Devuelve w_out con shape [num_steps, G.mapping.num_ws, w_dim]
        para ser compatible con tu pipeline (generate.py en space='w').
    """
    import dnnlib
    import torch.nn.functional as F
    import numpy as np
    import copy

    # Copia de la red en modo eval, sin gradientes
    G = copy.deepcopy(G).eval().requires_grad_(False).to(device)

    def logprint(*args):
        if verbose:
            print(*args)

    # ------------------------------------------------------------------
    # 1) Preparar target: [C,H,W] -> [1,C,H,W] en [-1,1]
    # ------------------------------------------------------------------
    assert target.ndim == 3, f"target debe ser [C,H,W], recibido {target.shape}"
    if target.dtype != torch.float32:
        target = target.to(torch.float32)
    # [0,255] -> [-1,1]
    target = target / 127.5 - 1.0               # [C,H,W] en [-1,1]
    target = target.unsqueeze(0).to(device)     # [1,C,H,W]

    # ------------------------------------------------------------------
    # 2) Estadísticos en W (w_avg, w_std) respetando c_dim
    # ------------------------------------------------------------------
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')

    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim).astype(np.float32)
    z_samples_t = torch.from_numpy(z_samples).to(device)  # [N, z_dim]

    if getattr(G, "c_dim", 0) > 0:
        # red condicional -> labels en cero (clase 0 por defecto)
        c_samples = torch.zeros([w_avg_samples, G.c_dim], device=device)
    else:
        c_samples = None

    # mapping(z, c) -> [N, num_ws, w_dim]
    w_samples = G.mapping(z_samples_t, c_samples)         # [N, num_ws, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]

    w_avg = np.mean(w_samples, axis=0, keepdims=True)     # [1,1,C]
    w_std = (np.mean((w_samples - w_avg) ** 2) ** 0.5).astype(np.float32)

    # ------------------------------------------------------------------
    # 3) Buffers de ruido de síntesis
    # ------------------------------------------------------------------
    noise_bufs = {name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name}

    # ------------------------------------------------------------------
    # 4) Cargar VGG16 para LPIPS
    # ------------------------------------------------------------------
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Helper: preparar tensores para VGG (si C=1 -> repetir a 3 canales)
    def to_vgg_images(x: torch.Tensor) -> torch.Tensor:
        """
        x: [N,C,H,W] en rango [-1,1]
        Devuelve [N,3,H',W'] en [0,255], con H',W' <= 256
        """
        # [-1,1] -> [0,255]
        x = (x + 1.0) * (255.0 / 2.0)
        # Si C=1, repetir canal
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        # Reescalar si es más grande de 256
        if x.shape[2] > 256:
            x = F.interpolate(x, size=(256, 256), mode='area')
        return x

    # ------------------------------------------------------------------
    # 5) Features del target
    # ------------------------------------------------------------------
    target_images = to_vgg_images(target)  # [1,3,H',W']
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)

    # ------------------------------------------------------------------
    # 6) Setup de optimización en W
    # ------------------------------------------------------------------
    w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True)  # [1,1,C]
    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)  # [T,1,C]
    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()),
                                 betas=(0.9, 0.999), lr=initial_learning_rate)

    # Inicializar ruido
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    # ------------------------------------------------------------------
    # 7) Bucle principal de optimización
    # ------------------------------------------------------------------
    for step in range(num_steps):
        t = step / num_steps

        # schedule de LR
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # w con ruido
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        # Expandir a W+ (num_ws)
        ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])  # [1,num_ws,C]

        # Imagen sintética
        synth_images = G.synthesis(ws, noise_mode='const')  # [1,Cg,H,W]d

        # Features VGG de la sintética (repitiendo canal si Cg=1)
        synth_vgg = to_vgg_images(synth_images)  # [1,3,H',W']
        synth_features = vgg16(synth_vgg, resize_images=False, return_lpips=True)

        # Distancia LPIPS
        dist = (target_features - synth_features).square().sum()

        # Regularización de ruido
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None, None, :, :]  # [1,1,H,W]
            while True:
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss = dist + reg_loss * regularize_noise_weight

        # Paso de optimización
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if verbose:
            logprint(f'step {step+1:>4d}/{num_steps}: dist {float(dist):.4f} loss {float(loss):.4f}')

        # Guardar W (solo la primera posición [0])
        w_out[step] = w_opt.detach()[0]  # [1,C]

        # Normalizar ruido
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    # ------------------------------------------------------------------
    # 8) Devolver W+ para todos los steps: [T, num_ws, C]
    # ------------------------------------------------------------------
    return w_out.repeat([1, G.mapping.num_ws, 1])


#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--target', 'target_fname', help='Target image file to project to', required=True, metavar='FILE')
@click.option('--num-steps',              help='Number of optimization steps', type=int, default=1000, show_default=True)
@click.option('--seed',                   help='Random seed', type=int, default=303, show_default=True)
@click.option('--save-video',             help='Save an mp4 video of optimization progress', type=bool, default=True, show_default=True)
@click.option('--outdir',                 help='Where to save the output images', required=True, metavar='DIR')
@click.option('--device',                 type=click.Choice(['cuda', 'cpu']), default='cuda', show_default=True,
              help='Device to use for projection')
def run_projection(
    network_pkl: str,
    target_fname: str,
    outdir: str,
    save_video: bool,
    seed: int,
    num_steps: int,
    device: str,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: --device=cuda requested but CUDA is not available. Falling back to CPU.")
        device = 'cpu'

    device = torch.device(device)
    print(f'Using device: {device}')

    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device)  # type: ignore

    target_pil = PIL.Image.open(target_fname).convert('RGB')
    w, h = target_pil.size
    s = min(w, h)
    target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
    target_uint8 = np.array(target_pil, dtype=np.uint8)

    start_time = perf_counter()
    projected_w_steps = project(
        G,
        target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device),
        num_steps=num_steps,
        device=device,
        verbose=True
    )
    print(f'Elapsed: {(perf_counter()-start_time):.1f} s')

    os.makedirs(outdir, exist_ok=True)

    target_pil.save(f'{outdir}/target.png')

    projected_w = projected_w_steps[-1]

    synth = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
    synth = (synth + 1) * (255.0 / 2.0)
    synth = synth.clamp(0, 255)[0].detach().cpu().numpy()

    if synth.shape[0] == 1:
        synth_img = synth[0].astype(np.uint8)
        PIL.Image.fromarray(synth_img, mode='L').save(f'{outdir}/proj.png')
        synth_rgb_for_video = np.stack([synth_img] * 3, axis=-1)
    elif synth.shape[0] == 3:
        synth_img = np.transpose(synth, (1, 2, 0)).astype(np.uint8)
        PIL.Image.fromarray(synth_img, mode='RGB').save(f'{outdir}/proj.png')
        synth_rgb_for_video = synth_img
    else:
        raise ValueError(f"Unexpected channel count in synth image: {synth.shape}")

    np.savez(f'{outdir}/projected_w.npz', w=projected_w.unsqueeze(0).cpu().numpy())

    if save_video:
        video = imageio.get_writer(
            f'{outdir}/proj.mp4', mode='I', fps=10, codec='libx264', bitrate='16M'
        )
        print(f'Saving optimization progress video "{outdir}/proj.mp4"')

        if target_uint8.ndim == 2:
            target_rgb = np.stack([target_uint8] * 3, axis=-1)
        elif target_uint8.ndim == 3 and target_uint8.shape[2] == 3:
            target_rgb = target_uint8
        else:
            raise ValueError(f"Unexpected target_uint8 shape: {target_uint8.shape}")

        for w_step in projected_w_steps:
            synth_step = G.synthesis(w_step.unsqueeze(0), noise_mode='const')
            synth_step = (synth_step + 1) * (255.0 / 2.0)
            synth_step = synth_step.clamp(0, 255)[0].detach().cpu().numpy()

            if synth_step.shape[0] == 1:
                frame_gray = synth_step[0].astype(np.uint8)
                frame_rgb = np.stack([frame_gray] * 3, axis=-1)
            elif synth_step.shape[0] == 3:
                frame_rgb = np.transpose(synth_step, (1, 2, 0)).astype(np.uint8)
            else:
                raise ValueError(f"Unexpected channel count in synth_step: {synth_step.shape}")

            video.append_data(np.concatenate([target_rgb, frame_rgb], axis=1))

        video.close()



#----------------------------------------------------------------------------

if __name__ == "__main__":
    run_projection() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
