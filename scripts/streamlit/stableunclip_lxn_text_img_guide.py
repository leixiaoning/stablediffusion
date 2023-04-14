import importlib
import streamlit as st
import torch
import cv2
import numpy as np
import PIL
from omegaconf import OmegaConf
from PIL import Image
from tqdm import trange
import io, os
from torch import autocast
from einops import rearrange, repeat
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from contextlib import nullcontext

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from diffusers import DiffusionPipeline
from accelerate import Accelerator
from ft_config import FTConfig, collate_fn, DreamBoothDataset, load_image, resize_image
import bitsandbytes as bnb
import itertools
from diffusers.optimization import get_scheduler
import json as js
from tqdm import tqdm
from ldm.modules.diffusionmodules.util import noise_like


torch.set_grad_enabled(False)

PROMPTS_ROOT = "scripts/prompts/"
SAVE_PATH = "outputs/demo/stable-unclip/"

VERSION2SPECS = {
    "Stable unCLIP-L": {"H": 768, "W": 768, "C": 4, "f": 8},
    "Stable unOpenCLIP-H": {"H": 768, "W": 768, "C": 4, "f": 8},
    "Full Karlo": {}
}


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    importlib.invalidate_caches()
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_interactive_image(key=None):
    image = st.file_uploader("Input", type=["jpg", "JPEG", "png"], key=key)
    if image is not None:
        image = Image.open(image)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        return image

def load_img_local(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    #print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.

def load_img(display=True, key=None):
    image = get_interactive_image(key=key)
    if display:
        st.image(image)
    w, h = image.size
    print(f"loaded input image of size ({w}, {h})")
    w, h = map(lambda x: x - x % 64, (w, h))
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.


def get_init_img(batch_size=1, key=None):
    init_image = load_img(key=key).cuda()
    init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
    return init_image


def sample(
        model,
        prompt,#基于prompt生成的context引导，在unet里面使用，不影响效果
        n_runs=1,
        n_samples=1,
        H=768,
        W=768,
        C=4,
        f=8,
        scale=10.0,
        ddim_steps=50,
        ddim_eta=0.0,
        callback=None,
        skip_single_save=False,
        save_grid=True,
        ucg_schedule=None,
        negative_prompt="",
        adm_cond=None,
        adm_uc=None,
        use_full_precision=False,
        only_adm_cond=False
):
    batch_size = n_samples
    precision_scope = autocast if not use_full_precision else nullcontext
    # decoderscope = autocast if not use_full_precision else nullcontext
    if use_full_precision: st.warning(f"Running {model.__class__.__name__} at full precision.")
    if isinstance(prompt, str):
        prompt = [prompt]
    prompts = batch_size * prompt

    outputs = st.empty()

    with precision_scope("cuda"):
        with model.ema_scope():
            all_samples = list()
            for n in trange(n_runs, desc="Sampling"):
                shape = [C, H // f, W // f]
                if not only_adm_cond:
                    uc = None
                    if scale != 1.0:
                        uc = model.get_learned_conditioning(batch_size * [negative_prompt])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    c = model.get_learned_conditioning(prompts)

                if adm_cond is not None:
                    if adm_cond.shape[0] == 1:
                        adm_cond = repeat(adm_cond, '1 ... -> b ...', b=batch_size)
                    if adm_uc is None:
                        st.warning("Not guiding via c_adm")
                        adm_uc = adm_cond
                    else:
                        if adm_uc.shape[0] == 1:
                            adm_uc = repeat(adm_uc, '1 ... -> b ...', b=batch_size)
                    if not only_adm_cond:
                        c = {"c_crossattn": [c], "c_adm": adm_cond}
                        uc = {"c_crossattn": [uc], "c_adm": adm_uc}
                    else:
                        c = adm_cond
                        uc = adm_uc
                #上面整理完了扩散模型的引导特征, crossattn是context，这边不会有变动，adm则是这里真正使用的img clip emb   
                samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                 conditioning=c,
                                                 batch_size=batch_size,
                                                 shape=shape,
                                                 verbose=False,
                                                 unconditional_guidance_scale=scale,
                                                 unconditional_conditioning=uc,
                                                 eta=ddim_eta,
                                                 x_T=None,
                                                 callback=callback,
                                                 ucg_schedule=ucg_schedule
                                                 ) # 扩散过程
                x_samples = model.decode_first_stage(samples_ddim) # decode (1, 4, 96, 96) -> (1, 3, 768, 768)
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                
                #image = load_img_local('/www/simple_ssd/lxn3/karlo/datatest/317_2/style/starry_night.jpeg')
                #image = image.type(x_samples.dtype).to(x_samples.device)
                #samples_ddim2 = model.get_first_stage_encoding(model.encode_first_stage(image))
                #x_samples2 = model.decode_first_stage(samples_ddim2)
                

                if not skip_single_save:
                    base_count = len(os.listdir(os.path.join(SAVE_PATH, "samples")))
                    for x_sample in x_samples:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        Image.fromarray(x_sample.astype(np.uint8)).save(
                            os.path.join(SAVE_PATH, "samples", f"{base_count:09}.png"))
                        base_count += 1

                all_samples.append(x_samples)

                # get grid of all samples
                grid = torch.stack(all_samples, 0)
                grid = rearrange(grid, 'n b c h w -> (n h) (b w) c')
                outputs.image(grid.cpu().numpy())

            # additionally, save grid
            grid = Image.fromarray((255. * grid.cpu().numpy()).astype(np.uint8))
            if save_grid:
                grid_count = len(os.listdir(SAVE_PATH)) - 1
                grid.save(os.path.join(SAVE_PATH, f'grid-{grid_count:06}.png'))

    return x_samples


def make_oscillating_guidance_schedule(num_steps, max_weight=15., min_weight=1.):
    schedule = list()
    for i in range(num_steps):
        if float(i / num_steps) < 0.1:
            schedule.append(max_weight)
        elif i % 2 == 0:
            schedule.append(min_weight)
        else:
            schedule.append(max_weight)
    print(f"OSCILLATING GUIDANCE SCHEDULE: \n {schedule}")
    return schedule


def torch2np(x):
    x = ((x + 1.0) * 127.5).clamp(0, 255).to(dtype=torch.uint8)
    x = x.permute(0, 2, 3, 1).detach().cpu().numpy()
    return x


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def init(version="Stable unCLIP-L", load_karlo_prior=False):
    state = dict()
    if not "model" in state:
        if version == "Stable unCLIP-L":
            config = "configs/stable-diffusion/v2-1-stable-unclip-l-inference.yaml"
            ckpt = "checkpoints/sd21-unclip-l.ckpt"

        elif version == "Stable unOpenCLIP-H":
            config = "configs/stable-diffusion/v2-1-stable-unclip-h-inference.yaml"
            ckpt = "checkpoints/sd21-unclip-h.ckpt"

        elif version == "Full Karlo":
            from ldm.modules.karlo.kakao.sampler import T2ISampler
            st.info("Loading full KARLO..")
            karlo = T2ISampler.from_pretrained(
                root_dir="checkpoints/karlo_models",
                clip_model_path="ViT-L-14.pt",
                clip_stat_path="ViT-L-14_stats.th",
                sampling_type="default",
            )
            state["karlo_prior"] = karlo
            state["msg"] = "loaded full Karlo"
            return state
        else:
            raise ValueError(f"version {version} unknown!")

        config = OmegaConf.load(config)
        model, msg = load_model_from_config(config, ckpt, vae_sd=None)
        state["msg"] = msg

        if load_karlo_prior:
            from ldm.modules.karlo.kakao.sampler import PriorSampler
            st.info("Loading KARLO CLIP prior...")
            karlo_prior = PriorSampler.from_pretrained(
                root_dir="checkpoints/karlo_models",
                clip_model_path="ViT-L-14.pt",
                clip_stat_path="ViT-L-14_stats.th",
                sampling_type="default",
            )
            state["karlo_prior"] = karlo_prior
        state["model"] = model
        state["ckpt"] = ckpt
        state["config"] = config
    return state


def load_model_from_config(config, ckpt, verbose=False, vae_sd=None):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    msg = None
    if "global_step" in pl_sd:
        msg = f"This is global step {pl_sd['global_step']}. "
    if "model_ema.num_updates" in pl_sd["state_dict"]:
        msg += f"And we got {pl_sd['state_dict']['model_ema.num_updates']} EMA updates."
    global_step = pl_sd.get("global_step", "?")
    sd = pl_sd["state_dict"]
    if vae_sd is not None:
        for k in sd.keys():
            if "first_stage" in k:
                sd[k] = vae_sd[k[len("first_stage_model."):]]

    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    print(f"Loaded global step {global_step}")
    return model, msg

def process_data(config, request_dir = 'cache/tmp_data/'):
    instance_dir = os.path.join(request_dir, 'instance_data')
    
    n_valid_image = 0
    if os.path.exists(config.instance_dir):
        imgs = os.listdir(config.instance_dir)
        for i in range(len(imgs)):
            imgi = imgs[i]
            d = os.path.join(config.instance_dir, imgi)
            user_image, image_error_flag = load_image(d, 'jpg')

            if not image_error_flag:
                if config.face_detect:
                    face_image, box, roll = resize_image(user_image)
                else:
                    face_image = cv2.resize(user_image, (1024, 1024))
                    box = (0.0, 0.0, 1.0, 1.0)
                    roll = 0.0
                if not face_image is None:
                    cv2.imwrite(os.path.join(instance_dir, str(i) + ".jpg"), face_image)
                    with open(os.path.join(os.path.join(instance_dir, str(i) + ".jpg.json")), "w",
                                      encoding='utf-8') as f:
                        js.dump({"image": str(i) + ".jpg", "box": box, "roll": float(roll)}, f)
                    n_valid_image += 1
    if n_valid_image <= 0:
        if os.path.exists(request_dir):
            import shutil
            shutil.rmtree(request_dir)
        print("valid image num is not enough")
        exit()
    return instance_dir

if __name__ == "__main__":
    version = 'Stable unCLIP-L'
    steps = 20 #采样步数
    #初始化模型
    state = init(version=version, load_karlo_prior=True)
    pipe_img_karlo = DiffusionPipeline.from_pretrained("/www/simple_ssd/lxn3/mtimageblend/plugins/imageblend/models/karlo-v1-alpha-image-variations", \
            torch_dtype=torch.float16, custom_pipeline='src/unclip_image_interpolation_lxn.py')
    pipe_img_karlo.to('cuda')    
    sampler = DDIMSampler(state["model"])
    #这里模型都已经在cuda上,且requires_grad都是true
    config = FTConfig()
    instance_dir = process_data(config)#数据处理
    train_dataset = DreamBoothDataset(
        instance_data_root=instance_dir,
        instance_prompt=config.instance_prompt,
        class_data_root=config.class_data_dir if config.with_prior_preservation else None,
        class_prompt=config.class_prompt,
        tokenizer=None,
        size=config.resolution,
        center_crop=config.center_crop,
        text_max_length=state['karlo_prior']._prior.model.text_ctx,
        img_guide=True,
        pipe_img_karlo=pipe_img_karlo,
        sd_model=state["model"],
        karlo_prior_model=state["karlo_prior"],
    )
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batchsize, shuffle=True, collate_fn=collate_fn, num_workers=0)
    ######
    accelerator = Accelerator(mixed_precision='fp16', log_with="tensorboard", logging_dir='cache/log')
    if accelerator.is_main_process:
        print("accelerator main process !!")
    #优化器
    optimizer_class = bnb.optim.AdamW8bit
    params_to_optimize = (state['model'].model.parameters())
                                #, state['karlo_prior']._prior.parameters()))
    optimizer = optimizer_class(
        params_to_optimize,
        lr=3.5075e-06,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8,
    )
    
    
    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps * config.gradient_accumulation_steps,
        num_training_steps=config.max_train_steps * config.gradient_accumulation_steps,
    )

    accelerator.prepare(state['model'].model, optimizer, train_dataloader, lr_scheduler)#state['karlo_prior']._prior

    shape=[4, 768 // 8, 768 // 8]
    SAVE_PATH = os.path.join(SAVE_PATH, version)
    os.makedirs(os.path.join(SAVE_PATH, "samples"), exist_ok=True)
    device = state['model'].betas.device
    seed_everything(2023)
    batchsize = 2*config.batchsize
    sampler.make_schedule(ddim_num_steps=steps, ddim_eta=0.0, verbose=False)
    for epoch in range(config.num_train_epochs):
        state['model'].model.train()
        #state['karlo_prior']._prior.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(state['model'].model):

                text_emb = batch['text_emb']
                img_emb = batch['img_emb']
                pixels = batch['pixel_values']
                emb = img_emb

                
                img = torch.randn([batchsize]+shape, device=device)
                timesteps = sampler.ddim_timesteps
                time_range = np.flip(timesteps)
                total_steps = timesteps.shape[0]
                iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
                for i, step in enumerate(iterator):
                    index = total_steps - i - 1
                    ts = torch.full((batchsize,), step, device=device, dtype=torch.long)
                    
                    c = emb['c']
                    unconditional_conditioning = emb['uc']
                    x_in = torch.cat([img] * 2)
                    t_in = torch.cat([ts] * 2)
                    c_in = dict()
                    for k in c:
                        if isinstance(c[k], list):
                            c_in[k] = [torch.cat([
                                unconditional_conditioning[k][i],
                                c[k][i]]) for i in range(len(c[k]))]
                        else:
                            c_in[k] = torch.cat([
                                    unconditional_conditioning[k],
                                    c[k]])
                    model_uncond, model_t = state['model'].model(x_in, t_in, **c_in).chunk(2) # DiffusionWrapper
                    model_output = model_uncond + 10.0 * (model_t - model_uncond)

                    if state['model'].parameterization == "v":
                        e_t = state['model'].predict_eps_from_z_and_v(img, ts, model_output)
                    else:
                        e_t = model_output

                    alphas = sampler.ddim_alphas
                    alphas_prev = sampler.ddim_alphas_prev
                    sqrt_one_minus_alphas = sampler.ddim_sqrt_one_minus_alphas
                    sigmas = sampler.ddim_sigmas

                    a_t = torch.full((batchsize, 1, 1, 1), alphas[index], device=device)
                    a_prev = torch.full((batchsize, 1, 1, 1), alphas_prev[index], device=device)
                    sigma_t = torch.full((batchsize, 1, 1, 1), sigmas[index], device=device)
                    sqrt_one_minus_at = torch.full((batchsize, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

                    if state['model'].parameterization != "v":
                        pred_x0 = (img - sqrt_one_minus_at * e_t) / a_t.sqrt()
                    else:
                        pred_x0 = state['model'].predict_start_from_z_and_v(img, ts, model_output)

                    dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
                    noise = sigma_t * noise_like(img.shape, device, False) * 1.0

                    x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

                    img = x_prev
                    
                #torch.cuda.empty_cache()
                samples_ddim = img
                x_samples = state["model"].decode_first_stage(samples_ddim) # decode (1, 4, 96, 96) -> (1, 3, 768, 768)
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                
                #image = load_img_local('/www/simple_ssd/lxn3/karlo/datatest/317_2/style/starry_night.jpeg')
                #image = image.type(x_samples.dtype).to(x_samples.device)
                #samples_ddim2 = model.get_first_stage_encoding(model.encode_first_stage(image))
                #x_samples2 = model.decode_first_stage(samples_ddim2)
                

                if True: #保存图片
                    base_count = len(os.listdir(os.path.join(SAVE_PATH, "samples")))
                    for x_sample in x_samples:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        Image.fromarray(x_sample.astype(np.uint8)).save(
                            os.path.join(SAVE_PATH, "samples", f"{base_count:09}.png"))
                        base_count += 1

    

    
    
    
        
