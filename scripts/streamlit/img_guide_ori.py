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

import os
inputdir = "/www/simple_ssd/lxn3/karlo/datatest/317_2/reconstruct"
output = "/www/simple_ssd/lxn3/karlo/datatest/317_2/reconstruct_sd"
os.makedirs(output, exist_ok=True)
l = os.listdir(inputdir)
for i in range(len(l)):
    imgpath = os.path.join(inputdir, l[i])
    image = Image.open(imgpath).convert("RGB")
    image = pipe(image).images[0]
    image = image.resize((512,512))
    image.save(os.path.join(output, '{}.jpg'.format(i)))
print("done")