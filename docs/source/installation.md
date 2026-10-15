# Installation

Simulusion supports multiple installation methods depending on your use case.

## From PyPI

For stable releases:

```bash
# Core dependencies (torch, torchvision, torchaudio, gym, gymnasium, etc.)
pip install simulusion

# With specific extras
pip install simulusion[gym]       # Additional gym environments (Box2D and classic-control rendering, ALE, huggingface-hub, autorom)
pip install simulusion[ml-agents] # Unity ML-Agents support
pip install simulusion[ml]        # TensorBoard, Weights & Biases, logging tools
pip install simulusion[viz]       # FastAPI, Uvicorn, documentation tools
pip install simulusion[docs]      # Sphinx and documentation tools
pip install simulusion[dev]       # Testing and development tools (pytest, mypy, pre-commit)

# Install multiple extras
pip install simulusion[gym,ml-agents,dev]
```

### Available Extras

| Extra | Description |
|-------|-------------|
| `gym` | Additional Gym environment dependencies (Box2D and classic-control rendering, ALE, huggingface-hub, autorom) |
| `ml-agents` | Unity ML-Agents support |
| `ml` | TensorBoard, Weights & Biases, and logging tools |
| `viz` | Latent-space visualization (opencv-python, umap-learn, scikit-learn, plotly) |
| `docs` | Sphinx and documentation building tools |
| `dev` | pytest, mypy, pre-commit for development |

### Core Dependencies

The minimal installation includes only: torch, torchvision, click, einops, tqdm, and pyyaml. Everything else — environment backends, logging, and visualization — comes from the extras above.

## From Source

For the latest development version:

```bash
git clone https://github.com/ParamThakkar123/simulusion.git
cd simulusion

# Core dependencies
pip install -e .

# With extras
pip install -e ".[gym,ml-agents,ml,viz,dev,docs]"
```

## PyTorch Build Selection

Simulusion does not pin a single PyTorch wheel index. Install the PyTorch build that matches your platform (CPU, macOS, CUDA, ROCm, etc.) using the index recommended by the [PyTorch installation selector](https://pytorch.org/get-started/locally/).

```bash
# Example: CUDA 12.1 wheels. Replace the index for CPU, ROCm, or other CUDA versions.
uv add torch torchvision torchaudio --index https://download.pytorch.org/whl/cu121

# Or using pip.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Docker

Build the default CPU image and run the Simulusion CLI:

```bash
# Build
docker build -t simulusion .

# Show the CLI help
docker run --rm simulusion

# Run a specific command
docker run --rm simulusion models list
```

The Dockerfile installs PyTorch explicitly before installing Simulusion so the wheel
source is controlled by the `PYTORCH_INDEX_URL` build argument. The default uses
CPU wheels. To build against a CUDA wheel index, pass the matching PyTorch index
and run the container with the NVIDIA runtime:

```bash
docker build \
  --build-arg PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 \
  -t simulusion:cu121 .

docker run --rm --gpus all simulusion:cu121 models list
```

Additional optional dependency groups can be installed at build time with
`SIMULUSION_EXTRAS`, for example `--build-arg SIMULUSION_EXTRAS=viz,ml`. Runtime data is
stored under `/data/simulusion`, which you can persist with a bind mount or volume.

## Verification

Verify your installation:

```python
import torch
import simulusion

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print("Simulusion imported successfully!")
```