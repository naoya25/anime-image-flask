from controlnet_aux import CannyDetector
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler
import torch
import random
from PIL import Image


def create_anime_image(file_name: str) -> bool:
    try:
        original_image = Image.open(f"./static/images/{file_name}")

        # 入力画像の整形
        aspect_ratio = original_image.width / original_image.height
        if aspect_ratio > 1:
            new_width = 512
            new_height = int(new_width / aspect_ratio) - (int(new_width / aspect_ratio) % 8)
        else:
            new_height = 512
            new_width = int(new_height * aspect_ratio) - (int(new_height * aspect_ratio) % 8)
        init_image = original_image.resize((new_width, new_height))

        # Canny画像作成
        canny_detector = CannyDetector()
        canny_image = canny_detector(init_image)

        # Canny画像の保存
        canny_image.save("./static/images/canny_image.png")

        # ControlnetのPipelineの準備
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny").to("mps")
        pipe = StableDiffusionControlNetPipeline.from_single_file(
            "./models/animePastelDream_softBakedVae.safetensors",
            controlnet=controlnet,
            safety_checker=lambda images, **kwargs: (images, None),
        ).to("mps")

        # スケジューラ、最適化、
        # pipe.safety_checker = lambda images, **kwargs: (images, None)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.enable_attention_slicing()

        # seed値の設定
        seed = random.randrange(0, 4294967295, 1)
        generator = torch.Generator("mps").manual_seed(seed)

        # 画像生成
        image = pipe(
            prompt="anime style",  # アニメ系の画像にしたい
            negative_prompt="(worst quality:1.4), (low quality:1.4), (monochrome:1.3), nsfw, NSFW",
            num_inference_steps=20,
            generator=generator,
            image=canny_image,
        ).images[0]

        # 画像保存
        image.save("./static/images/result.png")
        print(f"Saved to result.png  seed: {seed}")
    except Exception as e:
        return False
    return True


if __name__ == "__main__":
    file_name = "test2.png"
    create_anime_image(file_name=file_name)
