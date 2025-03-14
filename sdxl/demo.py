import gradio as gr
import numpy as np
import random
import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16

repo = "stabilityai/sdxl-turbo"
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLPipeline.from_pretrained(repo, vae=vae, torch_dtype=torch.float16).to(device)

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1344

def infer(prompt, negative_prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps, progress=gr.Progress(track_tqdm=True)):

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    generator = torch.Generator().manual_seed(seed)

    image = pipe(
        prompt = prompt,
        negative_prompt = negative_prompt,
        guidance_scale = guidance_scale,
        num_inference_steps = num_inference_steps,
        width = width,
        height = height,
        generator = generator
    ).images[0]

    return image, seed

examples = [
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    "An astronaut riding a green horse",
    "A delicious ceviche cheesecake slice",
]

css="""
#col-container {
    margin: 0 auto;
    max-width: 580px;
}
"""

with gr.Blocks(css=css) as demo:

    with gr.Column(elem_id="col-container"):
        gr.Markdown(f"""
        # Demo [SDXL-Turbo](https://huggingface.co/stabilityai/stable-diffusion-3-medium)
        Learn more about the [Stable Diffusion XL series](https://stability.ai/news/stability-ai-sdxl-turbo). Try on [Stability AI API](https://platform.stability.ai/docs/api-reference#tag/Generate/paths/~1v2beta~1stable-image~1generate~1sd3/post), [Stable Assistant](https://stability.ai/stable-assistant), or on Discord via [Stable Artisan](https://stability.ai/stable-artisan). Run locally with [ComfyUI](https://github.com/comfyanonymous/ComfyUI) or [diffusers](https://github.com/huggingface/diffusers)
        """)

        with gr.Row():

            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
                container=False,
            )

            run_button = gr.Button("Run", scale=0)

        result = gr.Image(label="Result", show_label=False)

        with gr.Accordion("Advanced Settings", open=False):

            negative_prompt = gr.Text(
                label="Negative prompt",
                max_lines=1,
                placeholder="Enter a negative prompt",
            )

            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )

            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

            with gr.Row():

                width = gr.Slider(
                    label="Width",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=64,
                    value=1024,
                )

                height = gr.Slider(
                    label="Height",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=64,
                    value=1024,
                )

            with gr.Row():

                guidance_scale = gr.Slider(
                    label="Guidance scale",
                    minimum=0.0,
                    maximum=10.0,
                    step=0.1,
                    value=0,
                )

                num_inference_steps = gr.Slider(
                    label="Number of inference steps",
                    minimum=1,
                    maximum=15,
                    step=1,
                    value=4,
                )

        gr.Examples(
            examples = examples,
            inputs = [prompt]
        )
    gr.on(
        triggers=[run_button.click, prompt.submit, negative_prompt.submit],
        fn = infer,
        inputs = [prompt, negative_prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps],
        outputs = [result, seed]
    )

demo.launch()
