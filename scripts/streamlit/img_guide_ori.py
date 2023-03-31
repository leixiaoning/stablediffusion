from diffusers import DiffusionPipeline
from diffusers.utils import load_image
import torch
from PIL import Image

pipe = DiffusionPipeline.from_pretrained("checkpoints/stable-diffusion-2-1-unclip", torch_dtype=torch.float16)
pipe.to("cuda")

# get image
#url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/stable_unclip/tarsila_do_amaral.png"
#image = load_image(url)
imgpath = 'assets/stable-samples/img2img/upscaling-in.png' 
image = Image.open(imgpath).convert("RGB") # PIL

# run image variation
image = pipe(image).images[0]


print("done")