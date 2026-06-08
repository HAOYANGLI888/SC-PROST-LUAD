# OpenSlide On Windows For Stage 6A

The Stage 6A synthetic smoke test works without OpenSlide. Real TCGA `.svs`
reading should use OpenSlide.

## Recommended Conda Environment

```powershell
conda activate gpu_py310
conda install -c conda-forge openslide
python -m pip install openslide-python
```

Verify:

```powershell
python -c "import openslide; print(openslide.__version__)"
```

## Windows DLL Fallback

If importing `openslide` reports a missing DLL:

1. Download a Windows OpenSlide binary distribution.
2. Extract it to a stable path, for example `C:\tools\openslide-win64`.
3. Add its `bin` directory to `PATH` before running Python:

```powershell
$env:PATH = "C:\tools\openslide-win64\bin;$env:PATH"
python -c "import openslide; print(openslide.__version__)"
```

The project can use Pillow for synthetic TIFF, PNG, and JPEG smoke fixtures.
That fallback is single-level and is not a full replacement for OpenSlide when
reading pyramidal SVS files. `tifffile` or `pyvips` may be explored for limited
fallback access, but formal TCGA extraction should use OpenSlide.

