# 使ってない

import torch
import PIL
from PIL import Image
import numpy as np
import random
from diffusers import StableDiffusionImg2ImgPipeline


def preprocess(image):  # 入力画像のデータ整形
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def img2img(file_name: str) -> str:
    pipe = StableDiffusionImg2ImgPipeline.from_single_file(
        "./models/animePastelDream_softBakedVae.safetensors"
    )
    pipe = pipe.to("mps")  # windowsはcuda、macはmps

    init_image = Image.open(f"./static/images/{file_name}").convert("RGB")
    init_image = init_image.resize((512, 512))
    init_image = preprocess(init_image)

    prompt = "anima style,"
    if pipe.safety_checker is not None:
        pipe.safety_checker = lambda images, **kwargs: (images, None)
    pipe.enable_attention_slicing()
    seed = random.randrange(0, 4294967295, 1)
    generator = torch.Generator("mps").manual_seed(seed)
    image = pipe(prompt=prompt, image=init_image, generator=generator).images[0]
    image.save("./static/images/result.png")
    return "Success saved to result.png"


if __name__ == "__main__":
    file_name = "target.jpg"
    print(img2img(file_name=file_name))
