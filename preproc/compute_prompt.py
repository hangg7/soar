import base64
import os.path as osp
from glob import glob
from io import BytesIO

import imageio.v3 as iio
import tyro
from openai import OpenAI
from PIL import Image


def main(
    data_dir: str,
):
    img_dir = osp.join(data_dir, "images")
    prompt_path = osp.join(data_dir, "prompt.txt")

    if osp.exists(prompt_path):
        print("Prompt already computed.")
    else:
        base64_imgs = []
        for path in sorted(glob(osp.join(img_dir, "*.png"))):
            img = iio.imread(path)
            img_bytes = BytesIO()
            Image.fromarray(img).save(img_bytes, format="PNG")
            img_base64 = base64.b64encode(img_bytes.getvalue()).decode("utf-8")
            base64_imgs.append(img_base64)

        client = OpenAI()
        result = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        "Here's a video with a human. Describe concisely about the human.",
                        *map(lambda x: {"image": x, "resize": 768}, base64_imgs[:2]),
                    ],
                },
            ],
            max_tokens=200,
        )
        __import__("ipdb").set_trace()


if __name__ == "__main__":
    tyro.cli(main)
