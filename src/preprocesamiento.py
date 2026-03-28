#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocesador MRI (layout por TUMOR) **SIN USAR MÁSCARAS**:
  <root>/<tumor>/**/<img>

— Recorta SOLO con bbox por contenido de la IMAGEN (no busca ni usa máscaras/borders).
— Pad a cuadrado y resize a --size.
— Normaliza por percentiles (1–99) SOLO a la imagen; (opcional) equalize.
— Ignora archivos de máscara/border presentes en el input como candidatos.
— Guarda las imágenes en <output>/<tumor>/<stem>.png (conserva estructura por clase).
— Reporte JSON con conteos básicos.

Uso típico:
  python preprocess_image_bbox_only.py ^
    --input  "C:/.../datasetlisto" ^
    --output "C:/.../datasetclean" ^
    --size 256 --equalize --verbose
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageFile, ImageOps
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ----------------- Configuración básica -----------------
ALLOWED_TUMORS = {'glioma', 'meningioma', 'pituitary', 'no_tumor'}
EXTS       = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
OS_TRASH   = {'.DS_Store', 'Thumbs.db', 'desktop.ini'}

TUMOR_ALIASES = {
    'glioma': 'glioma', 'glio': 'glioma',
    'meningioma': 'meningioma', 'meningiomas': 'meningioma', 'meningiom': 'meningioma',
    'pituitary': 'pituitary', 'pit': 'pituitary', 'pituitaria': 'pituitary',
    'no_tumor': 'no_tumor', 'no-tumor': 'no_tumor', 'notumor': 'no_tumor', 'none': 'no_tumor', 'healthy': 'no_tumor'
}

def try_import_tqdm():
    try:
        from tqdm import tqdm
        return tqdm
    except Exception:
        def dummy(iterable, **kwargs): return iterable
        return dummy

tqdm = try_import_tqdm()

# ----------------- Utilidades de nombres -----------------
def normalize_tumor(name: str) -> str:
    key = name.strip().lower().replace(' ', '').replace('-', '_')
    key = key.replace('__', '_')
    return TUMOR_ALIASES.get(key, name.strip().lower())

def find_tumor_for(img_path: Path, root: Path) -> str:
    """Busca el tumor por los ancestros; si no lo encuentra, devuelve 'unknown'."""
    root = root.resolve()
    cur = img_path.parent.resolve()
    while True:
        if cur == cur.parent:
            break
        if str(cur).startswith(str(root)):
            tnorm = normalize_tumor(cur.name)
            if tnorm in ALLOWED_TUMORS:
                return tnorm
        if cur == root:
            break
        cur = cur.parent
    return "unknown"

def looks_like_mask_file(p: Path) -> bool:
    """Heurística para detectar máscaras/borders en el INPUT y NO contarlas como candidatos."""
    stem = p.stem.lower()
    parent = p.parent.name.lower()
    if stem.endswith("_mask") or stem.endswith("-mask") or stem.endswith(" mask"):
        return True
    if stem.endswith("_border") or stem.endswith("-border") or stem.endswith(" border"):
        return True
    if stem.endswith("_seg") or stem.endswith("-seg") or " seg " in stem:
        return True
    if stem.endswith("_label") or stem.endswith("-label") or " label " in stem:
        return True
    if stem.endswith("_msk") or stem.endswith("-msk"):
        return True
    if parent in ("mask", "masks", "seg", "segment", "segmentation", "labels", "label", "border", "borders"):
        return True
    return False

# ----------------- I/O & normalización -----------------
def load_grayscale_safe(p: Path) -> np.ndarray:
    with Image.open(p) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode != 'L':
            im = im.convert('L')
        arr = np.array(im, dtype=np.uint8)
    return arr

def normalize_by_percentiles(arr_u8: np.ndarray, p_low=1.0, p_high=99.0) -> np.ndarray:
    lo = float(np.percentile(arr_u8, p_low))
    hi = float(np.percentile(arr_u8, p_high))
    if hi <= lo:
        hi = lo + 1.0
    arr = np.clip((arr_u8.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    arr = (arr * 255.0 + 0.5).astype(np.uint8)
    return arr

# ----------------- BBox por contenido (SOLO IMAGEN) -----------------
def crop_content_bbox(arr: np.ndarray, pct_threshold: float = 5.0, min_frac: float = 0.0005) -> Tuple[int,int,int,int]:
    """BBox por contenido (umbral dinámico). Fallback = imagen completa si no hay señal."""
    assert arr.ndim == 2, "Se espera imagen en escala de grises"
    try:
        thr = float(np.percentile(arr, pct_threshold))
    except Exception:
        thr = 0.0
    mask = arr > thr
    if mask.mean() < float(min_frac):
        try:
            thr2 = max(1.0, float(np.percentile(arr, 1.0)))
        except Exception:
            thr2 = 0.0
        mask = arr > thr2
        if mask.mean() < float(min_frac):
            return (0, 0, int(arr.shape[1]), int(arr.shape[0]))  # sin crop
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return (0, 0, int(arr.shape[1]), int(arr.shape[0]))
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    y0 = max(0, y0 - 2); x0 = max(0, x0 - 2)
    y1 = min(arr.shape[0] - 1, y1 + 2); x1 = min(arr.shape[1] - 1, x1 + 2)
    return (x0, y0, x1 + 1, y1 + 1)

def expand_bbox_to_min_side(x0, y0, x1, y1, min_side, H, W):
    """Expande bbox al menos a min_side por lado (si cabe)."""
    w = x1 - x0; h = y1 - y0
    if min(w, h) >= min_side:
        return x0, y0, x1, y1
    cx = (x0 + x1) // 2; cy = (y0 + y1) // 2
    half = int(round(min_side / 2.0))
    nx0 = max(0, cx - half); ny0 = max(0, cy - half)
    nx1 = min(W, cx + half); ny1 = min(H, cy + half)
    if (nx1 - nx0) < min_side:
        if nx0 == 0: nx1 = min(W, nx0 + min_side)
        elif nx1 == W: nx0 = max(0, nx1 - min_side)
    if (ny1 - ny0) < min_side:
        if ny0 == 0: ny1 = min(H, ny0 + min_side)
        elif ny1 == H: ny0 = max(0, ny1 - min_side)
    return nx0, ny0, nx1, ny1

# ----------------- Prepro de IMAGEN (sin máscara) -----------------
def preprocess_image_bbox_only(
    arr_img_u8: np.ndarray,
    size: int = 256,
    do_equalize: bool = False,
    replicate_to_rgb: bool = False,
    crop_pct_threshold: float = 5.0,
    min_side_after_crop: int = 16
) -> Image.Image:
    """
    — Bbox SOLO desde IMAGEN con percentil de contenido.
    — Pad a cuadrado + resize (BICUBIC).
    — Normalización por percentiles (1–99) y (opcional) equalize.
    """
    H, W = arr_img_u8.shape[:2]

    # 1) BBox por imagen
    x0, y0, x1, y1 = crop_content_bbox(arr_img_u8, pct_threshold=crop_pct_threshold)
    if (x1 - x0) <= 0 or (y1 - y0) <= 0:
        x0, y0, x1, y1 = 0, 0, W, H

    x0, y0, x1, y1 = expand_bbox_to_min_side(x0, y0, x1, y1, int(min_side_after_crop), H, W)

    # 2) Recorte y pad a cuadrado
    crop_img = arr_img_u8[y0:y1, x0:x1]
    h, w = crop_img.shape
    side = max(h, w)
    y_off = (side - h) // 2
    x_off = (side - w) // 2

    canvas_img = np.zeros((side, side), dtype=np.uint8)
    canvas_img[y_off:y_off + h, x_off:x_off + w] = crop_img

    # 3) Resize + normalización + opcional equalize
    if side != int(size):
        pil_img = Image.fromarray(canvas_img, mode='L').resize((size, size), resample=Image.BICUBIC)
    else:
        pil_img = Image.fromarray(canvas_img, mode='L')

    arr2 = normalize_by_percentiles(np.array(pil_img, dtype=np.uint8), 1.0, 99.0)
    pil_img = Image.fromarray(arr2, mode='L')
    if do_equalize:
        pil_img = ImageOps.equalize(pil_img)

    if replicate_to_rgb:
        pil_img = Image.merge('RGB', (pil_img, pil_img, pil_img))

    return pil_img

# ----------------- Auditoría + Procesamiento (sin máscara) -----------------
def audit_and_process_no_mask(
    root: Path,
    outdir: Path,
    size: int = 256,
    do_equalize: bool = False,
    replicate_to_rgb: bool = False,
    audit_only: bool = False,
    min_side_after_crop: int = 16,
    crop_pct_threshold: float = 5.0,
    verbose: bool = False
):
    """
    Recorre el dataset, recorta por IMAGEN, pad+resize y guarda (size x size).
    Guarda la imagen como <stem>.png bajo <output>/<tumor>/.
    """
    if not root.exists():
        print("[ERROR] Input no existe: {}".format(root), file=sys.stderr)
        sys.exit(1)

    reasons = Counter()
    samples = {}
    totals = {'candidates': 0, 'processed': 0, 'failed_open': 0, 'unknown_tumor': 0, 'ignored_mask_like': 0}

    if not audit_only:
        outdir.mkdir(parents=True, exist_ok=True)

    # Candidatos: imágenes reales (no máscaras)
    all_files = [p for p in root.rglob('*') if p.is_file()]
    candidate_imgs: List[Path] = []
    for p in all_files:
        if p.name in OS_TRASH:
            continue
        if p.suffix.lower() not in EXTS:
            continue
        if looks_like_mask_file(p):
            totals['ignored_mask_like'] += 1
            continue
        candidate_imgs.append(p)
    totals['candidates'] = len(candidate_imgs)

    for img in tqdm(candidate_imgs, desc="Procesando", unit="img"):
        tumor = find_tumor_for(img, root)
        if tumor not in ALLOWED_TUMORS:
            totals['unknown_tumor'] += 1
            if verbose:
                print(f"[SKIP] Clase/tumor no reconocido para {img}")
            continue

        try:
            arr_img = load_grayscale_safe(img)
        except Exception:
            totals['failed_open'] += 1
            samples.setdefault('failed_open_example', str(img))
            if verbose:
                print(f"[OPEN] No se pudo abrir imagen: {img}")
            continue

        pil_img = preprocess_image_bbox_only(
            arr_img_u8=arr_img,
            size=int(size),
            do_equalize=bool(do_equalize),
            replicate_to_rgb=bool(replicate_to_rgb),
            crop_pct_threshold=float(crop_pct_threshold),
            min_side_after_crop=int(min_side_after_crop)
        )

        reasons['ok'] += 1
        totals['processed'] += 1

        if not audit_only:
            out_parent = outdir / tumor
            out_parent.mkdir(parents=True, exist_ok=True)
            out_img_path = out_parent / (img.stem + ".png")
            try:
                pil_img.save(out_img_path, format='PNG', optimize=False)
                if verbose:
                    print(f"[SAVE] img -> {out_img_path}")
            except Exception:
                reasons['cannot_save'] += 1
                samples.setdefault('cannot_save_example', str(out_img_path))

    # Resumen + Reporte
    print("\n=== RESUMEN (SIN MÁSCARAS) ===")
    print("Candidatos (imágenes reales): {}".format(totals['candidates']))
    print("Procesados (OK): {}".format(totals['processed']))
    print("Ignorados por parecer máscara/border: {}".format(totals['ignored_mask_like']))
    if totals['failed_open'] > 0:
        print("No se pudieron abrir: {}   ej: {}".format(totals['failed_open'], samples.get('failed_open_example')))
    if totals['unknown_tumor'] > 0:
        print("Saltadas por tumor desconocido: {}".format(totals['unknown_tumor']))

    report = {
        'layout': 'tumor_only',
        'totals': totals,
        'reasons': dict(reasons),
        'samples': samples,
    }
    try:
        outdir.mkdir(parents=True, exist_ok=True)
        report_path = outdir / "preprocess_audit_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print("\nReporte JSON: {}".format(report_path))
    except Exception as e:
        print("[WARN] No se pudo escribir reporte JSON: {}".format(e))

    return reasons, samples, totals

# ----------------- CLI -----------------
def build_argparser():
    p = argparse.ArgumentParser(description="Recorta SOLO con bbox por IMAGEN; salida size×size (ignora máscaras).")
    p.add_argument('--input', type=str,  default="C:/Users/Junwei/Downloads/dataset/datasetlisto")
    p.add_argument('--output', type=str, default="C:/Users/Junwei/Downloads/dataset/datasetclean")
    p.add_argument('--size', type=int, default=256, help="Tamaño final (lado) de imagen")
    p.add_argument('--equalize', action='store_true', help="Equalize (ImageOps) solo a la imagen")
    p.add_argument('--replicate-to-rgb', action='store_true', help="Replicar canal L a RGB para backbones preentrenados")
    p.add_argument('--audit-only', action='store_true', help="Solo auditoría (no guarda archivos)")
    p.add_argument('--min-side', type=int, default=16, help="Mínimo lado del bbox (se expande si cabe)")
    p.add_argument('--crop-pct-thr', type=float, default=5.0, help="Percentil para bbox por contenido en IMAGEN")
    p.add_argument('--verbose', action='store_true', help='Imprimir rutas guardadas y skips.')
    return p

def main():
    args = build_argparser().parse_args()
    root = Path(args.input)
    outdir = Path(args.output) if args.output else Path("./processed_out")

    audit_and_process_no_mask(
        root=root,
        outdir=outdir,
        size=int(args.size),
        do_equalize=bool(args.equalize),
        replicate_to_rgb=bool(args.replicate_to_rgb),
        audit_only=bool(args.audit_only),
        min_side_after_crop=int(args.min_side),
        crop_pct_threshold=float(args.crop_pct_thr),
        verbose=bool(args.verbose),
    )

if __name__ == "__main__":
    main()
