# Adapted from
# https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/llm-serving/download_llama.py

import modal

VOLUME_NAME = 'hnet_vol'
VOLUME_PATH = '/' + VOLUME_NAME
MODELS_DIR = VOLUME_PATH + '/models'
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
VOLUME_CFG = {VOLUME_PATH: volume}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        [
            "huggingface_hub",  # download models from the Hugging Face Hub
            "hf-transfer",  # download models faster
        ]
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


MINUTES = 60
HOURS = 60 * MINUTES


app = modal.App(
    image=image, secrets=[modal.Secret.from_name("huggingface-secret")]
)


@app.function(volumes=VOLUME_CFG, timeout=4 * HOURS)
def download_model(model_name, model_revision, force_download=False):
    from huggingface_hub import snapshot_download

    print(f'Downloading {model_name}:{model_revision}')

    volume.reload()

    snapshot_download(
        model_name,
        local_dir=MODELS_DIR + "/" + model_name,
        ignore_patterns=[
            # H-Net is only provided in PyTorch format.
            #"*.pt",
            "*.bin",
            "*.pth",
            "original/*",
        ],
        revision=model_revision,
        force_download=force_download,
    )

    volume.commit()


DEFAULT_NAME = ""
DEFAULT_REVISION = ''

@app.local_entrypoint()
def main(
    model_name: str = DEFAULT_NAME,
    model_revision: str = DEFAULT_REVISION,
    force_download: bool = False,
):
    download_model.remote(model_name, model_revision, force_download)
