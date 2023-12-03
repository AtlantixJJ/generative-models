import math
import os
import sys
sys.path.insert(0, '.')
import torch
import numpy as np
from glob import glob
from pathlib import Path
from typing import Optional
from moviepy.editor import ImageSequenceClip
from tqdm import tqdm
from einops import rearrange, repeat
from fire import Fire
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import ToTensor

from sgm.util import default, instantiate_from_config


def imwrite(fpath, arr):
    with open(fpath, "wb") as f:
        Image.fromarray(arr).save(f)


def read_images(input_path: str):
    """Read all images in folder `input_path` or a single image `input_path`."""
    path = Path(input_path)
    all_img_paths = []
    if path.is_file():
        if any([input_path.endswith(x) for x in ["jpg", "jpeg", "png"]]):
            all_img_paths = [input_path]
        else:
            raise ValueError("Path is not valid image file.")
    elif path.is_dir():
        all_img_paths = sorted(
            [
                f
                for f in path.iterdir()
                if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
        )
        if len(all_img_paths) == 0:
            raise ValueError("Folder does not contain any images.")
    else:
        raise ValueError

    images = []
    for input_img_path in all_img_paths:
        with Image.open(input_img_path) as image:
            if image.mode == "RGBA":
                image = image.convert("RGB")
            w, h = image.size

            if h % 64 != 0 or w % 64 != 0:
                width, height = 1024, 576
                image = image.resize((width, height))
                print(
                    f"WARNING: Your image is of size {h}x{w} which is not divisible by 64. We are resizing to {height}x{width}!"
                )

            image = ToTensor()(image)
            image = image * 2.0 - 1.0
        image = image.unsqueeze(0)
        images.append(image)
    return images


def write_video(output_path, frames, fps=24):
    """Convert a list of frames to an MP4 video using MoviePy.
    Args:
        output_path: Path to the output video file.
        frames: List of image frames (PIL images or NumPy arrays).
        fps: Frames per second for the output video.
    """
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(output_path,
        codec='libx264', fps=fps,
        preset='ultrafast', threads=1)


def sample(
    input_path: str = "../../expr/dcn/video_input",
    num_steps: Optional[int] = None,
    version: str = "svd_cond",
    fps_id: int = 24,
    motion_bucket_id: int = 127,
    cond_aug: float = 0.02,
    sigma_cond: float = 100.0,
    seed: int = 23,
    decoding_t: int = 8,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    device: str = "cuda",
    output_folder: Optional[str] = None,
):
    """
    Simple script to generate a single sample conditioned on an image `input_path` or multiple images, one for each
    image file in folder `input_path`. If you run out of VRAM, try decreasing `decoding_t`.
    """
    init_frames = read_images(input_path)
    num_frames = len(init_frames)

    if version == "svd":
        num_frames = default(num_frames, 14)
        num_steps = default(num_steps, 25)
        output_folder = default(output_folder, "outputs/simple_video_sample/svd/")
        model_config = "scripts/sampling/configs/svd.yaml"
    elif version == "svd_cond":
        num_frames = default(num_frames, 14)
        num_steps = default(num_steps, 25)
        output_folder = default(output_folder, "outputs/simple_video_sample/svd/")
        model_config = "scripts/sampling/configs/svd_cond.yaml"
    elif version == "svd_xt":
        num_frames = default(num_frames, 25)
        num_steps = default(num_steps, 30)
        output_folder = default(output_folder, "outputs/simple_video_sample/svd_xt/")
        model_config = "scripts/sampling/configs/svd_xt.yaml"
    elif version == "svd_image_decoder":
        num_frames = default(num_frames, 14)
        num_steps = default(num_steps, 25)
        output_folder = default(
            output_folder, "outputs/simple_video_sample/svd_image_decoder/"
        )
        model_config = "scripts/sampling/configs/svd_image_decoder.yaml"
    elif version == "svd_xt_image_decoder":
        num_frames = default(num_frames, 25)
        num_steps = default(num_steps, 30)
        output_folder = default(
            output_folder, "outputs/simple_video_sample/svd_xt_image_decoder/"
        )
        model_config = "scripts/sampling/configs/svd_xt_image_decoder.yaml"
    else:
        raise ValueError(f"Version {version} does not exist.")

    model = load_model(model_config, device, decoding_t, num_steps, sigma_cond)
    torch.manual_seed(seed)

    init_frames = torch.cat(init_frames).to(device)
    first_frame = init_frames[:1]
    H, W = first_frame.shape[2:]
    assert first_frame.shape[1] == 3

    if (H, W) != (576, 1024):
        print(
            "WARNING: The conditioning frame you provided is not 576x1024. This leads to suboptimal performance as model was only trained on 576x1024. Consider increasing `cond_aug`."
        )
    if motion_bucket_id > 255:
        print(
            "WARNING: High motion bucket! This may lead to suboptimal performance."
        )

    if fps_id < 5:
        print("WARNING: Small fps value! This may lead to suboptimal performance.")

    if fps_id > 30:
        print("WARNING: Large fps value! This may lead to suboptimal performance.")

    value_dict = {}
    value_dict["motion_bucket_id"] = motion_bucket_id
    value_dict["fps_id"] = fps_id
    value_dict["cond_aug"] = cond_aug

    os.makedirs(output_folder, exist_ok=True)
    base_count = len(glob(os.path.join(output_folder, "*.mp4")))
    video_path = os.path.join(output_folder, f"{base_count:06d}.mp4")
    n_half = decoding_t // 2
    n_batch = math.floor(num_frames / n_half) # discard last batch for convenience
    x0_init = init_frames[:2 * n_half]
    frames = []
    with torch.no_grad():
        with torch.autocast(device):
            for i in tqdm(range(n_batch - 1)):
                cur_cond_frames = init_frames[i * n_half : (i + 2) * n_half]
                value_dict["cond_frames_without_noise"] = cur_cond_frames
                value_dict["cond_frames"] = cur_cond_frames + cond_aug * torch.randn_like(cur_cond_frames)
                samples = infer_video(model, value_dict, x0_init, sigma_cond, decoding_t, device)
                x0_init = torch.cat([
                    samples[n_half:],
                    init_frames[(i + 1) * n_half : (i + 2) * n_half]])
                frames.append(samples[:n_half].cpu())
    frames = (torch.cat(frames, dim=0) * 0.5 + 0.5).clamp(0, 1)
    vid = (rearrange(frames, "t c h w -> t h w c") * 255).cpu().numpy().astype(np.uint8)
    #for i, frame in enumerate(vid):
    #    imwrite(os.path.join(output_folder, f"{base_count:06d}_{i:03d}.png"), frame)
    write_video(video_path, list(vid))


def infer_video(model, value_dict, x0_init, sigma_cond, decoding_t, device):
    cond_frames = value_dict['cond_frames']
    num_frames = cond_frames.shape[0]
    batch, batch_uc = get_batch(
        get_unique_embedder_keys_from_conditioner(model.conditioner),
        value_dict,
        [1, num_frames],
        T=num_frames,
        device=device,
    )
    c, uc = model.conditioner.get_unconditional_conditioning(
        batch,
        batch_uc=batch_uc,
        force_uc_zero_embeddings=[
            "cond_frames",
            "cond_frames_without_noise",
        ],
    )

    for k in ["crossattn", "concat"]:
        if uc[k].shape[0] < num_frames:
            # 1 in, num_frames out
            uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames)
            c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames)
            uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=num_frames)
            c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=num_frames)

    #randn = torch.randn(shape, device=device)
    z0_init = model.encode_first_stage(x0_init)
    noisy_z0 = z0_init + sigma_cond * torch.randn_like(z0_init)
    noisy_z0 /= (1 + sigma_cond ** 2) ** 0.5

    additional_model_inputs = {}
    additional_model_inputs["image_only_indicator"] = torch.zeros(
        2, num_frames
    ).to(device)
    additional_model_inputs["num_video_frames"] = batch["num_video_frames"]

    def denoiser(input, sigma, c):
        return model.denoiser(
            model.model, input, sigma, c, **additional_model_inputs
        )

    samples_z = model.sampler(denoiser, noisy_z0, cond=c, uc=uc)
    model.en_and_decode_n_samples_a_time = decoding_t
    samples_x = model.decode_first_stage(samples_z)
    return samples_x


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_batch(keys, value_dict, N, T, device):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to(device),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames":
            batch[key] = value_dict["cond_frames"]
            #batch[key] = repeat(value_dict["cond_frames"], "1 ... -> b ...", b=N[0])
        elif key == "cond_frames_without_noise":
            batch[key] = value_dict["cond_frames_without_noise"]
            #batch[key] = repeat(value_dict["cond_frames_without_noise"], "1 ... -> b ...", b=N[0])
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def load_model(
    config: str,
    device: str,
    num_frames: int,
    num_steps: int,
    sigma_cond: float
):
    config = OmegaConf.load(config)
    if device == "cuda":
        config.model.params.conditioner_config.params.emb_models[
            0
        ].params.open_clip_embedding_config.params.init_device = device

    config.model.params.sampler_config.params.num_steps = num_steps
    config.model.params.sampler_config.params.discretization_config.params.sigma_max = sigma_cond
    config.model.params.sampler_config.params.guider_config.params.num_frames = (
        num_frames
    )
    if device == "cuda":
        with torch.device(device):
            model = instantiate_from_config(config.model).to(device).eval()
    else:
        model = instantiate_from_config(config.model).to(device).eval()

    #filter = DeepFloydDataFiltering(verbose=False, device=device)
    return model#, filter


if __name__ == "__main__":
    Fire(sample)
