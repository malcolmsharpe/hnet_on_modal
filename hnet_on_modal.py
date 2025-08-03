import modal

### Image definitions

WHL_REQS = 'cu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64'

FLASH_ATTN_WHL = f'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+{WHL_REQS}.whl'
MAMBA_SSM_WHL = f'https://github.com/state-spaces/mamba/releases/download/v2.2.5/mamba_ssm-2.2.5+{WHL_REQS}.whl'
CAUSAL_CONV1D_WHL = f'https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.2/causal_conv1d-1.5.2+{WHL_REQS}.whl'

image_flash_attn = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11",
    )
    .apt_install(
        # Required for installs from github.
        'git',
        'build-essential',
    )
    .pip_install(
        "ninja",
        "packaging",
        "wheel",
        "setuptools>=61.0.0",
    )
    .pip_install(
        'torch==2.7.1',
        extra_options="--index-url https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        FLASH_ATTN_WHL, extra_options='--only-binary=:all:',
    )
)

image = (
    image_flash_attn
    .pip_install(
        'einops',
        'optree',
        'regex',
        'omegaconf',
    )
    .pip_install(
        MAMBA_SSM_WHL, extra_options='--only-binary=:all:',
    )
    .pip_install(
        CAUSAL_CONV1D_WHL, extra_options='--only-binary=:all:',
    )
    .run_commands(
        'git clone https://github.com/goombalab/hnet',
    )
    .run_commands(
        'cd hnet && pip install -e . --no-deps && pip check',
    )
)

### Volume setup

VOLUME_NAME = 'hnet_vol'
VOLUME_PATH = '/' + VOLUME_NAME
MODELS_DIR = VOLUME_PATH + '/models'
volume = modal.Volume.from_name(VOLUME_NAME)
VOLUME_CFG = {VOLUME_PATH: volume}

### App and functions

GPU_CONFIG = 'A10G'

app = modal.App('hnet')

@app.function(
    gpu=GPU_CONFIG,
    image=image_flash_attn
)
def print_versions():
    import torch, sys
    print("torch:", torch.__version__)
    print("cuda from torch:", torch.version.cuda)
    print("python:", sys.version.split()[0])
    try:
        print("cxx11 abi:", torch._C._GLIBCXX_USE_CXX11_ABI)
    except Exception as e:
        print("cxx11 abi: unknown", e)
    from flash_attn.ops.triton.layer_norm import RMSNorm
    print("flash-attn OK")

@app.function(
    gpu=GPU_CONFIG,
    image=image
)
def print_help():
    import subprocess

    print(subprocess.check_output(["python", "/hnet/generate.py", "--help"], text=True))

@app.function(
    image=image,
    volumes=VOLUME_CFG,
)
def volume_test():
    import subprocess

    print(subprocess.check_output(['du', '-h', VOLUME_PATH]))
    print(subprocess.check_output(['du', '-h', MODELS_DIR]))

@app.function(
    gpu=GPU_CONFIG,
    image=image,
    volumes=VOLUME_CFG,
)
def example_generate(
    model_name: str = "hnet_2stage_XL",
):
    import subprocess

    print(subprocess.check_output([
        'python', 'generate.py',
        '--model-path', MODELS_DIR + f'/cartesia-ai/{model_name}/{model_name}.pt',
        '--config-path', f'configs/{model_name}.json',
        '--max-tokens', '1024',
         '--temperature', '1.0',
         '--top-p', '1.0',
         ],
         cwd='/hnet',
         text=True,
         ))
