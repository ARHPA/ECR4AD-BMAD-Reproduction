from PIL import Image
import cv2
import numpy as np
import os
import gc
import math
import ctypes
import glob
from pathlib import Path
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, Literal, Tuple, Dict, Any, List, Optional
import openslide

def get_level_by_magnification(slide_path, target_mag):
    slide = openslide.OpenSlide(slide_path)
    mpp_x = float(slide.properties["openslide.mpp-x"])
    mpp_y = float(slide.properties["openslide.mpp-y"])
    base_pixel_size = (mpp_x + mpp_y) / 2

    base_mag = 10.0 / base_pixel_size

    downsamples = [float(d) for d in slide.level_downsamples]
    level_pixel_sizes = [base_pixel_size * ds for ds in downsamples]

    target_pixel_size = 10.0 / target_mag

    level = min(range(len(level_pixel_sizes)), key=lambda i: abs(level_pixel_sizes[i] - target_pixel_size))
    slide.close()

    return level, base_mag, level_pixel_sizes[level]

def in_hsv_range(image, min_coverage = 60):
    if image is None:
        return False

    if isinstance(image, Image.Image):
        # Convert PIL Image to numpy array and BGR format
        image = image.convert("RGB")
        image = np.array(image)
        # Convert RGB to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    else:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define HSV ranges
    lower_bound = np.array([90, 8, 103], dtype=np.uint8)
    upper_bound = np.array([180, 255, 255], dtype=np.uint8)

    # Create mask for pixels within the specified ranges
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Calculate tissue coverage percentage
    coverage = cv2.countNonZero(mask)  / mask.size * 100
    # If coverage is less than min_coverage, return False
    if coverage < min_coverage:
        return False

    return True

def use_local_vips_linux(
    vips_prefix: str,
    openslide_prefix: Optional[str] = None,
    dicom_prefix: Optional[str] = None,
) -> Dict[str, str]:
    """
    在 CentOS/Linux 中预加载 libdicom → OpenSlide → libvips（按此顺序）。
    - vips_prefix:       libvips 安装前缀，如 /home/you/local/vips
    - openslide_prefix:  OpenSlide 安装前缀；不传则尝试用 vips 的 lib 目录
    - dicom_prefix:      libdicom 安装前缀；不传则尝试在 openslide/vips 的 lib* 目录发现

    返回已加载库的实际路径，便于调试。
    """
    def pick_libdir(prefix: str) -> Path:
        p = Path(prefix)
        for name in ("lib64", "lib"):
            d = p / name
            if d.is_dir():
                return d
        raise FileNotFoundError(f"未在 {prefix} 下找到 lib64 或 lib 目录")

    def dlopen_one(patterns, libdirs):
        for libdir in libdirs:
            for pat in patterns:
                for path in sorted(glob.glob(str(Path(libdir) / pat))):
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                    return path
        return None

    vips_libdir = pick_libdir(vips_prefix)
    # 让当前进程解析依赖时能找到库（仅对本进程生效）
    os.environ["LD_LIBRARY_PATH"] = str(vips_libdir) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")

    if openslide_prefix:
        oslide_libdir = pick_libdir(openslide_prefix)
    else:
        oslide_libdir = vips_libdir

    dicom_libdirs = []
    if dicom_prefix:
        dicom_libdirs.append(pick_libdir(dicom_prefix))
    dicom_libdirs.extend([oslide_libdir, vips_libdir])

    loaded = {}

    # 1) libdicom（如需要）
    dicm = dlopen_one(["libdicom.so.1*", "libdicom.so*"], dicom_libdirs)
    if dicm:
        loaded["libdicom"] = dicm

    # 2) OpenSlide
    oslide = dlopen_one(["libopenslide.so.0*", "libopenslide.so*"], [oslide_libdir])
    if not oslide:
        raise FileNotFoundError(f"未在 {oslide_libdir} 找到 libopenslide.so（请检查 OpenSlide 安装路径）")
    loaded["openslide"] = oslide

    # 3) libvips
    vips = dlopen_one(["libvips.so.*", "libvips.so*"], [vips_libdir])
    if not vips:
        raise FileNotFoundError(f"未在 {vips_libdir} 找到 libvips.so（请检查 libvips 安装路径）")
    loaded["vips"] = vips
    loaded["vips_libdir"] = str(vips_libdir)
    return loaded

# 这些设置在 import pyvips 之后才能生效
def tune_vips_cache(concurrency: int = 8,
                    max_nodes: int = 2000,
                    max_mem_bytes: int = 512 * 1024**2,
                    max_files: int = 800):
    """
    调整 vips 的并发和缓存（建议：根据 ulimit -n 设置 max_files，留安全余量）
    """
    import pyvips
    os.environ.setdefault("VIPS_CONCURRENCY", str(concurrency))
    pyvips.cache_set_max(max_nodes)
    pyvips.cache_set_max_mem(max_mem_bytes)
    pyvips.cache_set_max_files(max_files)

def read_region(path: str,
        location: tuple[int, int],  # (x0, y0) in level-0 coords
        level: int,
        size: tuple[int, int],      # (w0, h0) desired output size
        *,
        access: str = "random") -> Image.Image:
    import pyvips

    x0, y0 = location
    w_out, h_out = size

    imgL = pyvips.Image.openslideload(path, level=level, access=access)

    key = f"openslide.level[{level}].downsample"
    if imgL.get_typeof(key) != 0:
        ds = float(imgL.get(key))
    else:
        w0_full = int(imgL.get("openslide.level[0].width"))
        ds = w0_full / imgL.width

    rx = math.floor(x0 / ds)
    ry = math.floor(y0 / ds)

    W, H = imgL.width, imgL.height
    ix1 = max(0, min(rx, W))
    iy1 = max(0, min(ry, H))
    ix2 = max(0, min(rx + w_out, W))
    iy2 = max(0, min(ry + h_out, H))
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    if iw == 0 or ih == 0:
        blank = pyvips.Image.black(w_out, h_out)            # 1 band zeros
        rgba = blank.bandjoin([blank, blank, blank])        # 4 bands (0,0,0,0)
        mem = rgba.write_to_memory()
        arr = np.frombuffer(mem, dtype=np.uint8).reshape(h_out, w_out, 4)
        return Image.fromarray(arr, mode="RGBA")

    crop = imgL.extract_area(ix1, iy1, iw, ih)

    if crop.bands == 3:
        crop = crop.bandjoin_const([255])          # RGB + A=255
    elif crop.bands != 4:
        crop = crop.colourspace("srgb")
        if crop.bands == 3:
            crop = crop.bandjoin_const([255])
        else:
            crop = crop.bandjoin([crop, crop]).bandjoin_const([255])

    dx = ix1 - rx 
    dy = iy1 - ry
    out = crop.embed(dx, dy, w_out, h_out, extend="background", background=[0, 0, 0, 0])

    mem = out.write_to_memory()
    arr = np.frombuffer(mem, dtype=np.uint8).reshape(h_out, w_out, 4)
    return Image.fromarray(arr, mode="RGBA")
