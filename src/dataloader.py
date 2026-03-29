# convert_figshare_mat_to_images.py
# -*- coding: utf-8 -*-
"""
Convierte .mat (Figshare Jun MRI) a imágenes separadas por clase,
nombradas como <PID>_<tumor>_<idx>.<ext>, y además exporta:
  - máscara binaria: <PID>_<tumor>_<idx>_mask.png
  - borde rasterizado: <PID>_<tumor>_<idx>_border.png

Clases:
  1 -> meningioma
  2 -> glioma
  3 -> pituitary

Requisitos:
  pip install mat73  (recomendado para .mat v7.3)
  (fallback) scipy
"""

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional, Iterable

import numpy as np
from PIL import Image, ImageDraw

LABEL_TO_NAME = {1: "meningioma", 2: "glioma", 3: "pituitary"}
IMG_EXTS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}

# -------------------------
# Carga robusta del .mat
# -------------------------
def try_load_mat(p: Path) -> Tuple[Dict[str, Any], str]:
    """Intenta cargar .mat con mat73 (v7.3) y si falla, con scipy (v7 o menos)."""
    # 1) mat73
    try:
        import mat73  # requiere h5py internamente
        d = mat73.loadmat(str(p))
        cj = d.get("cjdata", d)  # algunas distros envuelven directo a dict
        return cj, "mat73"
    except Exception as e_mat73:
        err1 = f"mat73_fail:{e_mat73}"

    # 2) scipy
    try:
        import scipy.io as sio
        d = sio.loadmat(str(p), struct_as_record=False, squeeze_me=True)
        cj = d.get("cjdata", d)
        # si vino como objeto estilo struct, lo pasamos a dict
        if not isinstance(cj, dict):
            cj = {k: getattr(cj, k) for k in dir(cj) if not k.startswith("_")}
        return cj, "scipy"
    except Exception as e_scipy:
        err2 = f"scipy_fail:{e_scipy}"

    raise RuntimeError(f"No pude leer {p}. Intentos: ['{err1}', '{err2}']")

# -------------------------
# Helpers
# -------------------------
def to_scalar(x):
    """Convierte arrays/listas escalarizadas a un escalar nativo si es posible."""
    if isinstance(x, (list, tuple)) and len(x) == 1:
        return to_scalar(x[0])
    if isinstance(x, np.ndarray):
        if x.size == 1:
            return to_scalar(x.item())
        return x
    return x

def get_any(cj: Dict[str, Any], *keys: str):
    """Obtiene cj[k] probando varias variantes de clave (insensible a mayúsculas/minúsculas)."""
    lk = {k.lower(): k for k in cj.keys()}
    for k in keys:
        if k.lower() in lk:
            return cj[lk[k.lower()]]
    return None

def normalize_pid(pid_raw) -> str:
    """Normaliza PID a string legible y estable."""
    if pid_raw is None:
        return "unknown"
    pid = to_scalar(pid_raw)
    if isinstance(pid, (int, np.integer)):
        return str(int(pid))
    if isinstance(pid, (float, np.floating)):
        if math.isfinite(pid) and abs(pid - int(pid)) < 1e-6:
            return str(int(pid))
        return str(pid).replace(".", "_")
    if isinstance(pid, str):
        pid = pid.strip()
        return pid if pid else "unknown"
    return str(pid)

def normalize_label(label_raw) -> Tuple[int, str]:
    """Devuelve (label_int, nombre_clase). Lanza si es inválido."""
    lab = to_scalar(label_raw)
    if isinstance(lab, (float, np.floating)) and abs(lab - int(lab)) < 1e-6:
        lab = int(lab)
    if not isinstance(lab, (int, np.integer)):
        raise ValueError(f"Etiqueta no numérica: {lab}")
    if lab not in LABEL_TO_NAME:
        raise ValueError(f"Etiqueta fuera de {list(LABEL_TO_NAME.keys())}: {lab}")
    return int(lab), LABEL_TO_NAME[int(lab)]

def im_minmax_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Escala min-max a [0,255] uint8."""
    arr = np.asarray(arr)
    # asegurar 2D
    if arr.ndim > 2:
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            arr = arr.reshape(arr.shape[0], arr.shape[1])
    arr = arr.astype(np.float32)
    min1 = np.min(arr)
    max1 = np.max(arr)
    if max1 <= min1:
        return np.zeros_like(arr, dtype=np.uint8)
    out = (255.0 / (max1 - min1)) * (arr - min1)
    return np.clip(out, 0, 255).astype(np.uint8)

def mask_to_uint8(mask: np.ndarray, H: int, W: int) -> Optional[np.ndarray]:
    """Convierte máscara a uint8 0/255, validando tamaño."""
    if mask is None:
        return None
    m = np.asarray(mask)
    # exprime si viene (H,W,1)
    if m.ndim == 3 and m.shape[-1] == 1:
        m = m[..., 0]
    if m.ndim != 2:
        return None
    if m.shape != (H, W):
        # tamaños inconsistentes -> rechazamos para evitar deformaciones
        return None
    # binariza robusto
    m = (m > 0).astype(np.uint8) * 255
    return m

def _coords_from_array(a: np.ndarray) -> np.ndarray:
    """Devuelve array Nx2 de coords desde (N,2) o (2,N)."""
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError("tumorBorder no es 2D")
    if a.shape[1] == 2:
        return a.astype(np.float64)
    if a.shape[0] == 2:
        return a.T.astype(np.float64)
    raise ValueError("tumorBorder con forma desconocida")

def _score_bounds(coords_xy: np.ndarray, H: int, W: int) -> int:
    """Cuenta cuántos puntos quedan dentro [0,W-1]x[0,H-1] interpretando coords como (x,y)."""
    x = coords_xy[:, 0]
    y = coords_xy[:, 1]
    ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    return int(np.sum(ok))

def border_to_uint8(border_obj, H: int, W: int, close: bool = True, width: int = 1) -> Optional[np.ndarray]:
    """
    Intenta rasterizar tumorBorder a imagen 0/255.
    Acepta:
      - np.ndarray (N,2) o (2,N)
      - lista de arrays/listas (varios contornos)
      - dict con claves 'x'/'y' o similares
    Asume coords 1-based (MATLAB); se convierten a 0-based y se recortan al borde.
    """
    def _normalize_one(b) -> Optional[np.ndarray]:
        if b is None:
            return None
        # dict con x/y
        if isinstance(b, dict):
            # claves posibles
            xs = None
            ys = None
            for k in b.keys():
                lk = k.lower()
                if lk in ("x", "xs", "col", "cols"):
                    xs = np.asarray(b[k]).reshape(-1)
                if lk in ("y", "ys", "row", "rows"):
                    ys = np.asarray(b[k]).reshape(-1)
            if xs is not None and ys is not None and xs.size == ys.size and xs.size > 0:
                coords = np.stack([xs, ys], axis=1)
            else:
                # dict raro -> abortar
                return None
        else:
            a = np.asarray(b)
            if a.ndim != 2:
                return None
            # (N,2) o (2,N)
            coords = _coords_from_array(a)

        # probar dos interpretaciones: (x,y) y (y,x)
        # convertir de 1-based a 0-based (aprox): restamos 1 y luego recortamos
        c_xy = coords.copy()
        c_xy -= 1.0
        c_yx = coords[:, ::-1].copy()
        c_yx -= 1.0

        # elegir la que mejor cae dentro de la imagen
        s_xy = _score_bounds(c_xy, H, W)
        s_yx = _score_bounds(c_yx, H, W)
        chosen = c_xy if s_xy >= s_yx else c_yx

        # clamp a límites
        chosen[:, 0] = np.clip(np.round(chosen[:, 0]), 0, W - 1)
        chosen[:, 1] = np.clip(np.round(chosen[:, 1]), 0, H - 1)
        return chosen.astype(np.int32)

    # border puede ser un contorno o una lista de contornos
    contours: List[np.ndarray] = []
    if border_obj is None:
        return None
    if isinstance(border_obj, (list, tuple)):
        for b in border_obj:
            c = _normalize_one(b)
            if c is not None and len(c) >= 2:
                contours.append(c)
    else:
        c = _normalize_one(border_obj)
        if c is not None and len(c) >= 2:
            contours.append(c)

    if not contours:
        return None

    canvas = Image.new("L", (W, H), color=0)
    draw = ImageDraw.Draw(canvas)

    for c in contours:
        pts = [(int(x), int(y)) for x, y in c]
        if close and pts[0] != pts[-1]:
            pts = pts + [pts[0]]
        # dibujar como polilínea
        draw.line(pts, fill=255, width=width)

    return np.array(canvas, dtype=np.uint8)

def save_image_uint8(img_u8: np.ndarray, out_path: Path, ext: str):
    """Guarda una imagen uint8 en escala de grises."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.fromarray(img_u8)  # 'L' automático para uint8 2D
    if ext.lower() in {"jpg", "jpeg"}:
        im = im.convert("L")
        im.save(str(out_path), format="JPEG", quality=95)
    else:
        im.save(str(out_path))

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser("Convert .mat (Figshare Jun MRI) -> imágenes + máscara + borde por clase con nombres <PID>_<tumor>_<idx>.<ext>")
    ap.add_argument("--in",  dest="in_dir",   required=True, help="Path to input .mat files directory")
    ap.add_argument("--out", dest="out_dir", required=True, help="Path to output directory")
    ap.add_argument("--ext", type=str, default="png", help="Extensión de salida: png|jpg|tif|... (default: png)")
    ap.add_argument("--skip_mask", action="store_true", help="No guardar la máscara aunque exista.")
    ap.add_argument("--skip_border", action="store_true", help="No guardar el borde aunque exista.")
    ap.add_argument("--border_width", type=int, default=1, help="Grosor del trazo del borde (px).")
    args = ap.parse_args()

    in_dir  = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    ext = args.ext.lower().strip(".")
    if ext not in IMG_EXTS:
        print(f"[WARN] Extensión '{ext}' no usual. Se guardará igualmente.")
    out_dir.mkdir(parents=True, exist_ok=True)

    # contador por (pid, tumor) para evitar choques de nombre
    counters: Dict[Tuple[str, str], int] = {}
    # CSV de trazabilidad
    csv_path = out_dir / "index_conversion.csv"
    _csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_w = csv.writer(_csv_file)
    csv_w.writerow([
        "orig_mat", "loader", "pid", "label_id", "tumor",
        "out_image", "out_mask", "out_border"
    ])

    mat_files = sorted(in_dir.glob("*.mat"))
    if not mat_files:
        print(f"[ERROR] No encontré .mat en: {in_dir}")
        sys.exit(1)

    ok, fail = 0, 0
    for i, p in enumerate(mat_files, 1):
        try:
            cj, how = try_load_mat(p)
            label_id, tumor_name = normalize_label(get_any(cj, "label"))
            pid = normalize_pid(get_any(cj, "PID", "pid"))

            img = get_any(cj, "image")
            if img is None:
                raise ValueError("Campo 'image' no encontrado en cjdata.")

            img_u8 = im_minmax_to_uint8(np.array(img))
            H, W = img_u8.shape[0], img_u8.shape[1]

            key = (pid, tumor_name)
            counters[key] = counters.get(key, 0) + 1
            idx = counters[key]

            # rutas de salida
            out_rel_img = Path(tumor_name) / f"{pid}_{tumor_name}_{idx:04d}.{ext}"
            out_rel_msk = Path(tumor_name) / f"{pid}_{tumor_name}_{idx:04d}_mask.png"
            out_rel_bor = Path(tumor_name) / f"{pid}_{tumor_name}_{idx:04d}_border.png"

            out_path_img = out_dir / out_rel_img
            out_path_msk = out_dir / out_rel_msk
            out_path_bor = out_dir / out_rel_bor

            # guardar imagen
            save_image_uint8(img_u8, out_path_img, ext)

            # máscara (opcional)
            mask_rel = ""
            if not args.skip_mask:
                mask_raw = get_any(cj, "tumorMask", "tumormask", "mask")
                m_u8 = mask_to_uint8(mask_raw, H, W)
                if m_u8 is not None:
                    save_image_uint8(m_u8, out_path_msk, "png")
                    mask_rel = str(out_rel_msk)
                else:
                    print(f"[MASK] Omitida (ausente o tamaño inválido) -> {p.name}")

            # borde (opcional)
            border_rel = ""
            if not args.skip_border:
                border_raw = get_any(cj, "tumorBorder", "tumorborder", "border")
                b_u8 = border_to_uint8(border_raw, H, W, close=True, width=args.border_width)
                if b_u8 is not None:
                    save_image_uint8(b_u8, out_path_bor, "png")
                    border_rel = str(out_rel_bor)
                else:
                    print(f"[BORDER] Omitido (ausente o no interpretable) -> {p.name}")

            # CSV
            csv_w.writerow([
                str(p), how, pid, label_id, tumor_name,
                str(out_rel_img), mask_rel, border_rel
            ])
            ok += 1

            if i % 100 == 0:
                print(f"[PROG] {i}/{len(mat_files)} | OK={ok} FAIL={fail}")

        except Exception as e:
            fail += 1
            print(f"[ERROR] {p}: {e}")

    _csv_file.close()
    print(f"\n[FIN] Convertidos OK={ok} | Fallidos={fail} | Salida={out_dir}")
    print(f"[FIN] Índice CSV: {csv_path}")

if __name__ == "__main__":
    main()
