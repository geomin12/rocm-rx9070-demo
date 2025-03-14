# ROCm Demo on RX9070

## ROCM Community Edition Setup using TheRock 🪨

```shell
mkdir rocm
cd rocm
# Check GitHub releases for other distributions.
wget https://therock-artifacts.s3.us-east-2.amazonaws.com/therock-dist-gfx1201-20250305.tar.gz
tar -xzf therock-dist-gfx1201-20250305.tar.gz
export PATH="$PWD/bin:$PATH"
export LD_LIBRARY_PATH="$PWD/lib:$LD_LIBRARY_PATH"
cd ..

```

## Pytorch / torchvision / torchaudio installation

Install torch+rocm6.3 nightly:

```shell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.3
```

If any application setup overrides this torch version, install it again.

## DeepSeek demo

```shell
pip install gradio transformers

python deepseek/demo.py
```

## SDXL demo

```shell
python sdxl/demo.py
```

## SDXL-Turbo + ComfyUI demo

```shell
git clone https://github.com/comfyanonymous/ComfyUI

cd ComfyUI

pip install -r requirements.txt

huggingface-cli download stabilityai/sdxl-turbo sd_xl_turbo_1.0_fp16.safetensors --local-dir ./models/checkpoints

HIP_VISIBLE_DEVICES=0 python main.py --fp32-vae --fp8_e4m3fn-unet
```

Feel free to use any ComfyUI JSON workflow file in the `comfyui_json_demo` directory, and please move any JSON workflow file to the `ComfyUI/user/default/workflows` directory.

Please move any model checkpoints (ex: `sd_xl_turbo_1.0_fp16.safetensors`) to `ComfyUI/models/checkpoints` directory.
