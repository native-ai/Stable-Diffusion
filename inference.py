import argparse
from stable_diffusion.sd_15 import loader
from stable_diffusion import generator
from PIL import Image
from transformers import CLIPTokenizer
import torch

def generate_image(args):
    if torch.cuda.is_available() and args.allow_cuda:
        device = "cuda"
    elif torch.backends.mps.is_available() and args.allow_mps:
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    tokenizer = CLIPTokenizer(args.tokenizer_vocab, merges_file=args.tokenizer_merges)
    models = loader.preload_models_from_standard_weights(args.model_file, device)

    input_image = None
    if args.input_image_path:
        input_image = Image.open(args.input_image_path)
        input_image = input_image.resize((512, 512))

    output_image = generator.generate(
        prompt=args.prompt,
        uncond_prompt=args.uncond_prompt,
        input_image=input_image,
        strength=args.strength,
        do_cfg=args.do_cfg,
        cfg_scale=args.cfg_scale,
        sampler_name=args.sampler,
        n_inference_steps=args.num_inference_steps,
        seed=args.seed,
        models=models,
        device=device,
        idle_device="cpu",
        tokenizer=tokenizer,
    )

    output_image = Image.fromarray(output_image)
    output_image.save(args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using Stable Diffusion.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation.")
    parser.add_argument("--uncond_prompt", type=str, default="", help="Negative prompt for image generation.")
    parser.add_argument("--do_cfg", type=bool, default=True, help="Whether to use classifier-free guidance.")
    parser.add_argument("--cfg_scale", type=float, default=8, help="CFG scale value.")
    parser.add_argument("--input_image_path", type=str, help="Path to the input image for image-to-image generation.")
    parser.add_argument("--strength", type=float, default=0.9, help="Strength of the image-to-image generation.")
    parser.add_argument("--sampler", type=str, default="ddpm", help="Sampler name.")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--model_file", type=str, required=True, help="Path to the model weights file.")
    parser.add_argument("--output_path", type=str, default="image1.png", help="Path to save the generated image.")
    parser.add_argument("--tokenizer_vocab", type=str, default="model_weight/vocab.json",
                        help="Path to the tokenizer vocab file.")
    parser.add_argument("--tokenizer_merges", type=str, default="model_weight/merges.txt",
                        help="Path to the tokenizer merges file.")
    parser.add_argument("--allow_cuda", type=bool, default=False, help="Allow usage of CUDA if available.")
    parser.add_argument("--allow_mps", type=bool, default=False, help="Allow usage of MPS if available.")

    args = parser.parse_args()
    generate_image(args)
