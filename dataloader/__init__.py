VIPS_PREFIX = "libvips/bin"
OPENSLIDE_PREFIX = "openslide/bin"
DICOM_PREFIX = "libdicom/bin"

# The whole-slide-image helpers (openslide / pyvips) are only required by the
# slide-level pipelines (Camelyon16, LNM_Prostate). The patch-level pipelines
# (Camelyon16-BMAD, GleasonArvaniti) do not need them, so importing this package
# must not hard-fail when the native WSI libraries are unavailable (e.g. on a
# minimal Colab runtime).
try:
    from .wsi_utils import use_local_vips_linux, tune_vips_cache  # noqa: F401
    use_local_vips_linux(VIPS_PREFIX, OPENSLIDE_PREFIX, DICOM_PREFIX)
except Exception as _wsi_err:  # noqa: BLE001
    import warnings
    warnings.warn(
        "Whole-slide utilities (openslide/pyvips) are unavailable: "
        f"{_wsi_err}. Patch-level datasets (e.g. Camelyon16BMAD) will still work."
    )
