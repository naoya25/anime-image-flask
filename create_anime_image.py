from diffusers.utils import load_image
from controlnet_aux import CannyDetector
import torch
import random
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler


def create_anime_image(file_name: str) -> bool:
    try:
        init_image = load_image(f"./static/images/{file_name}").resize((512, 512))
        # canny画像の生成
        canny_detector = CannyDetector()
        canny_image = canny_detector(init_image)

        # ControlNetモデルの準備
        controlnet_pose = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float16
        ).to("mps")

        # パイプラインの準備
        pipe = StableDiffusionControlNetPipeline.from_single_file(
            "./models/Counterfeit-V3.0_fix_fp16.safetensors",
            controlnet=[controlnet_pose],
            torch_dtype=torch.float16,
        ).to("mps")
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

        if pipe.safety_checker is not None:
            pipe.safety_checker = lambda images, **kwargs: (images, None)
        pipe.enable_attention_slicing()

        # 画像生成
        seed = random.randrange(0, 4294967295, 1)
        generator = torch.Generator("mps").manual_seed(seed)
        image = pipe(
            prompt="anime style",
            negative_prompt="(worst quality:1.4), (low quality:1.4), (monochrome:1.3)",
            num_inference_steps=20,
            generator=generator,
            eta=1.0,
            image=[canny_image],
        ).images[0]

        image.save("./static/images/controlnet-result.png")
    except Exception as e:
        return False
    return True


if __name__ == "__main__":
    file_name = "target.jpg"
    print(create_anime_image(file_name=file_name))
