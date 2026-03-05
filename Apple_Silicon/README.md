# Stage 1 Inference for Apple Silicon

This folder contains a simplified script for running Stage 1 (quality restoration) inference on Apple Silicon Macs (M1/M2/M3/M4).

## What is Stage 1?

The original project has 4 stages:
1. **Stage 1**: Overall quality restoration (denoising, color correction)
2. **Stage 2**: Face detection
3. **Stage 3**: Face enhancement
4. **Stage 4**: Blending

This script only runs Stage 1 - the overall quality restoration of old photos.

## Requirements

- Apple Silicon Mac (M1, M2, M3, or M4) or Intel Mac
- Python 3.9+
- PyTorch with MPS support (Apple Silicon) or CPU

## Setup

1. **Create a virtual environment** (optional but recommended):

```bash
python3 -m venv venv
source venv/bin/activate
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Download pretrained models**:

```bash
bash download_models.sh
```

This will download:
- Restoration models (VAE_A_quality, VAE_B_quality, mapping_quality, etc.)
- Scratch detection model (for automatic mask generation)

## Usage

### Option 1: Photos without scratches (Quality Restoration)

```bash
python stage1_inference.py \
    --test_input /path/to/input_folder \
    --outputs_dir /path/to/output_folder \
    --Quality_restore
```

### Option 2: Photos with scratches (Automatic mask detection)

The script will automatically detect scratches and generate masks:

```bash
python stage1_inference.py \
    --test_input /path/to/input_folder \
    --outputs_dir /path/to/output_folder \
    --Scratch_and_Quality_restore
```

### Option 3: Photos with scratches (Manual mask)

If you already have masks, provide them directly:

```bash
python stage1_inference.py \
    --test_input /path/to/input_folder \
    --test_mask /path/to/mask_folder \
    --outputs_dir /path/to/output_folder \
    --Scratch_and_Quality_restore
```

### High Resolution Mode

For high-resolution images with scratches:

```bash
python stage1_inference.py \
    --test_input /path/to/input_folder \
    --outputs_dir /path/to/output_folder \
    --Scratch_and_Quality_restore \
    --HR
```

## Command-line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--test_input` | Input image directory (required) | - |
| `--outputs_dir` | Output directory (required) | - |
| `--test_mask` | Mask directory for scratched images | Auto-detect if not provided |
| `--checkpoints_dir` | Path to pretrained models | `./checkpoints/restoration` |
| `--gpu_ids` | GPU IDs (e.g., 0,1,2 or -1 for CPU/MPS) | `-1` |

### Processing Modes

| Option | Description | Default |
|--------|-------------|---------|
| `--test_mode` | Image processing mode: Full, Scale, or Crop | `Crop` |
| `--Quality_restore` | For RGB images without scratches | - |
| `--Scratch_and_Quality_restore` | For scratched images (auto-detect masks) | - |
| `--HR` | High resolution mode (with scratches) | - |

### Model Parameters

| Option | Description | Default |
|--------|-------------|---------|
| `--ngf` | Number of gen filters in first conv layer | 64 |
| `--n_downsample_global` | Number of downsampling layers | 3 |
| `--mc` | Max channels | 64 |
| `--k_size` | Kernel size | 4 |
| `--start_r` | Start layer for resblock | 1 |
| `--mapping_n_block` | Number of resblocks in mapping | 6 |
| `--map_mc` | Max channels in mapping | 512 |
| `--norm` | Normalization type | `instance` |
| `--spatio_size` | Spatial size | 32 |

### Other Options

| Option | Description | Default |
|--------|-------------|---------|
| `--device` | Device: auto, mps, or cpu | `auto` |
| `--batchSize` | Batch size | 1 |
| `--mask_dilation` | Mask dilation | 0 |
| `--which_epoch` | Which epoch to load | `latest` |

## How Scratch Detection Works

When you use `--Scratch_and_Quality_restore` without providing `--test_mask`:

1. The script automatically loads the scratch detection model (UNet)
2. It processes each input image to detect scratch regions
3. Generates binary masks (threshold at 0.4 probability)
4. Saves masks to `outputs_dir/masks/mask/`
5. Preprocessed inputs go to `outputs_dir/masks/input/`
6. Then runs the restoration model with these auto-generated masks

## Output Structure

The output directory will contain:
- `input_image/` - Preprocessed input images
- `restored_image/` - Restored output images
- `origin/` - Original input images
- `masks/` - (When using scratch detection)
  - `input/` - Preprocessed inputs used for restoration
  - `mask/` - Generated scratch masks

## Examples

```bash
# Quality restoration (no scratches)
python stage1_inference.py \
    --test_input ~/Pictures/old_photos \
    --outputs_dir ~/Pictures/restored \
    --Quality_restore

# With scratches - automatic mask detection
python stage1_inference.py \
    --test_input ~/Pictures/old_photos \
    --outputs_dir ~/Pictures/restored \
    --Scratch_and_Quality_restore

# High resolution with scratches
python stage1_inference.py \
    --test_input ~/Pictures/old_photos \
    --outputs_dir ~/Pictures/restored \
    --Scratch_and_Quality_restore \
    --HR

# Use CPU explicitly
python stage1_inference.py \
    --test_input ~/Pictures/old_photos \
    --outputs_dir ~/Pictures/restored \
    --device cpu
```

## Notes

- The script automatically uses MPS (Metal Performance Shaders) on Apple Silicon for GPU acceleration
- If MPS is not available, it falls back to CPU
- Output images are saved as PNG files
- The model files should be located at: `../Global/checkpoints/`
- The detection model should be at: `../Global/checkpoints/detection/FT_Epoch_latest.pt`
