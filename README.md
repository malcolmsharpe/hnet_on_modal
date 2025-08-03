# H-Net on Modal

Quick-and-dirty scripts for H-Net inference on Modal.

## Usage

```
./download_hnet_1stage_L.sh
modal shell hnet_on_modal.py::example_generate
cd /hnet
python generate.py --model-path /hnet_vol/models/cartesia-ai/hnet_1stage_L/hnet_1stage_L.pt --config-path configs/hnet_1stage_L.json --max-tokens 1024 --temperature 1.0 --top-p 1.0
```
