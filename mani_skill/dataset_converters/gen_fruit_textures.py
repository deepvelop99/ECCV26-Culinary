"""Generate photorealistic seamless fruit-skin textures via Stable Diffusion.

Outputs <fruit>_diffuse.jpg (512x512, tileable-ish) for each requested fruit
into /data/Cutting_rebuttal/assets/fruits/. Existing OBJ files already have
spherical UV coords so the texture wraps around the mesh in Blender.
"""
from __future__ import annotations
import argparse
from pathlib import Path

# Per-fruit prompts — tuned for fruit-skin texture (not whole fruit photo).
PROMPTS = {
    "avocado":          ("dark green pebbly avocado skin closeup, photorealistic, "
                         "seamless texture, no background, no shadows, top-down, 8k"),
    "grape":            ("purple grape skin closeup, glossy smooth, photorealistic, "
                         "seamless tileable texture, no shadows, 8k"),
    "kiwi":             ("brown fuzzy kiwifruit skin closeup, photorealistic detail, "
                         "seamless texture, no shadows, top-down, 8k"),
    "lemon":            ("bright yellow lemon peel closeup, dimpled bumpy skin, "
                         "photorealistic, seamless tileable, no shadows, 8k"),
    "mango":            ("ripe mango skin closeup, yellow-orange-red gradient, "
                         "smooth glossy, photorealistic seamless, 8k"),
    "pear":             ("yellow-green pear skin closeup, subtle freckles, "
                         "smooth matte, photorealistic seamless tileable, 8k"),
    "persimmon":        ("orange persimmon skin closeup, smooth glossy, "
                         "photorealistic seamless, no shadows, 8k"),
    "pineapple_slice":  ("yellow pineapple flesh closeup cut surface, fibrous, "
                         "photorealistic, seamless texture, top-down, 8k"),
    "tomato":           ("red tomato skin closeup, glossy smooth, "
                         "photorealistic seamless tileable, no shadows, 8k"),
    "watermelon_slice": ("red watermelon flesh with black seeds, juicy texture, "
                         "photorealistic, top-down closeup, 8k"),
}

NEG = ("background, shadows, text, watermark, signature, blurry, low quality, "
       "deformed, cartoon, illustration, drawing, hand")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/data/Cutting_rebuttal/assets/fruits")
    ap.add_argument("--model-id", default="stabilityai/sd-turbo",
                    help="SD-Turbo: 1-4 step fast, photorealistic")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--guidance", type=float, default=0.0,
                    help="SD-Turbo expects guidance=0")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fruits", nargs="+", default=list(PROMPTS.keys()))
    args = ap.parse_args()

    import torch
    from diffusers import AutoPipelineForText2Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"[gen] loading {args.model_id} on {device} dtype={dtype}")
    pipe = AutoPipelineForText2Image.from_pretrained(args.model_id, torch_dtype=dtype)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    g = torch.Generator(device=device).manual_seed(args.seed)

    for fruit in args.fruits:
        prompt = PROMPTS[fruit]
        print(f"[gen] {fruit}: {prompt[:80]}...")
        img = pipe(
            prompt=prompt, negative_prompt=NEG,
            num_inference_steps=args.steps, guidance_scale=args.guidance,
            width=args.size, height=args.size,
            generator=g,
        ).images[0]
        out = out_dir / f"{fruit}_diffuse.jpg"
        img.save(out, quality=92)
        print(f"  → {out.name}")

    print(f"[gen] done — {len(args.fruits)} textures")


if __name__ == "__main__":
    main()
