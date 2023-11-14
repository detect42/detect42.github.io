---
title: Diffusion入门--inference
tags: diffusion
abbrlink: 2786a856
date: 2023-09-07 21:46:52
---

### 本文为diffusion库用于推理的基本介绍，以代码块功能和实现为主。


---


### 三行简化版：

```python
from diffusers import DiffusionPipeline
import os
pipeline = DiffusionPipeline.from_pretrained("/data1/sdmodels/stable-diffusion-v1-4", use_safetensors=True)

pipeline.to("cuda")

image = pipeline("An image of a squirrel in Picasso style").images[0]

output_dir = "/data1/sdtest/"
output_filename = "2.1.png"
output_path = os.path.join(output_dir, output_filename)
image.save(output_path)
print("生成的图像已保存到:", output_path)
```

可以看到核心部分只用pipeline调用一个训练好的模型即可

----

### 当然我们也可以指定model和scheduler，并且自己生成原始噪声并手动实现循环降噪


```python
import numpy as np
from PIL import Image
import torch
from diffusers import DDPMScheduler, UNet2DModel, EulerDiscreteScheduler, UNet2DConditionModel
from diffusers import DDPMScheduler, UNet2DModel
import os

#分别指定scheduler和model
scheduler = DDPMScheduler.from_pretrained("google/ddpm-cat-256")
model = UNet2DModel.from_pretrained("google/ddpm-cat-256", use_safetensors=True).to("cuda")

#设定scheduler步数
scheduler.set_timesteps(50)

#批次大小为1，表示只有一个样本。
#有3个通道，可能表示了一个具有3个颜色通道的图像或者3个特征通道的数据。
#数据的空间维度由sample_size确定，表示了图像的高度和宽度。
sample_size = model.config.sample_size
noise = torch.randn((1, 3, sample_size, sample_size)).to("cuda")


input = noise
for t in scheduler.timesteps:
    #用model预测残余噪声
    with torch.no_grad():
        noisy_residual = model(input, t).sample
    #用scheduler实现diffusion反向传播，参数为预测噪声、步数。当前图像
    previous_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
    input = previous_noisy_sample


image = (input / 2 + 0.5).clamp(0, 1).squeeze()
image = (image.permute(1, 2, 0) * 255).round().to(torch.uint8).cpu().numpy()
image = Image.fromarray(image)


# 将图像数据保存为图像文件
output_dir = "/data1/sdtest/"
output_filename = "2.1.png"
output_path = os.path.join(output_dir, output_filename)
image.save(output_path)
print("生成的图像已保存到:", output_path)
```


---

#### 关于图像转换的一点补充：
将一个PyTorch张量（`input`）转换为一个NumPy数组，然后再将其转换为PIL（Python Imaging Library）图像对象。让我来解释每一行代码的作用：

1. `image = (input / 2 + 0.5).clamp(0, 1).squeeze()`

   - `input`是一个PyTorch张量，假设其值范围在[-1, 1]之间。
   - `(input / 2 + 0.5)`：这一步将输入张量的值从范围[-1, 1]线性映射到[0, 1]范围内。
   - `.clamp(0, 1)`：这一步确保所有的值都在[0, 1]之间，即截断小于0和大于1的值。
   - `.squeeze()`：如果输入张量的形状中存在大小为1的维度，这一步将其挤压掉，使得输出张量的维度更紧凑。

2. `image = (image.permute(1, 2, 0) * 255).round().to(torch.uint8).cpu().numpy()`

   - `image.permute(1, 2, 0)`：这一步对`image`张量的维度进行重排列，将通道维度移到最后一个维度上。这通常是由于PIL图像和NumPy数组的通道顺序不同。
   - `* 255`：将所有像素值乘以255，将像素值缩放到0-255的整数范围内。
   - `.round()`：四舍五入将浮点数像素值转换为整数像素值。
   - `.to(torch.uint8)`：将张量的数据类型转换为无符号8位整数，以确保数值范围在0-255之间。
   - `.cpu().numpy()`：将PyTorch张量转换为NumPy数组。

3. `image = Image.fromarray(image)`

   - `Image.fromarray(image)`：这一步将NumPy数组转换为PIL图像对象，使得你可以使用PIL库的功能来处理和显示图像。

总之，这段代码的目的是将一个经过处理的PyTorch张量（通常代表图像数据）转换为PIL图像对象，以便进行后续的图像处理或显示。

---

### 插播sdxl的基本实现


sdxl因为base&refine机制的引入实现机制相对复杂，这里仅提供基础写法。

```python
import torch
from diffusers import DiffusionPipeline
from PIL import Image
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# load both base & refiner
base = DiffusionPipeline.from_pretrained(
    "/data1/sdmodels/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
base.to("cuda")
refiner = DiffusionPipeline.from_pretrained(
    "/data1/sdmodels/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.to("cuda")

# Define how many steps and what % of steps to be run on each experts (80/20) here
n_steps = 40
high_noise_frac = 0.8

prompt = "1girl, aqua eyes, baseball cap, blonde hair, closed mouth, earrings, green background, hat, hoop earrings, jewelry, looking at viewer, shirt, short hair, simple background, solo, upper body, yellow shirt,high quality,masterpiece"

# run both experts
image = base(
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_end=high_noise_frac,
    output_type="latent",
).images
image = refiner(
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_start=high_noise_frac,
    image=image,
).images[0]

output_dir = "/data1/sdtest/"
output_filename = "2.1.png"
output_path = os.path.join(output_dir, output_filename)
# 将图像数据保存为图像文件
image.save(output_path)
print("生成的图像已保存到:", output_path)
```

----

### 当然有时我们希望能批量出图，并且可以调整整个图模型的各种参数，比如size，step_number等等。


```python
from diffusers import DiffusionPipeline, StableDiffusionPipeline
from diffusers import EulerDiscreteScheduler
import torch
from PIL import Image
import numpy as np
import os
from transformers import CLIPImageProcessor
from diffusers import DPMSolverMultistepScheduler
from diffusers.utils import make_image_grid
from diffusers import AutoencoderKL

model_id = "/data1/sdmodels/stable-diffusion-xl-base-1.0"
prompt = "1girl, aqua eyes, baseball cap, blonde hair, closed mouth, earrings, green background, hat, hoop earrings, jewelry, looking at viewer, shirt, short hair, simple background, solo, upper body, yellow shirt,high quality,masterpiece"
num_inference_steps = 50

# 使用fp16降低精度几乎不影响出图质量且可以节省大量显存并提速
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True, variant="fp16", safety_checker=None)

pipe.to("cuda")
# pipe.enable_xformers_memory_efficient_attention()

# 为模型选定随机种子
generator = torch.Generator("cuda").manual_seed(np.random.randint(0, 114514))

# pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
# pipe.enable_xformers_memory_efficient_attention()
# image = pipe(prompt, generator=generator, height=1024, width=1024,num_inference_steps=num_inference_steps).images[0]
# vae = AutoencoderKL.from_pretrained(
#   "stabilityai/sdxl-vae", torch_dtype=torch.float16).to("cuda")
# pipe.vae = vae

# 批量出图
def get_inputs(batch_size):
    generator = [torch.Generator("cuda").manual_seed(np.random.randint(0, 114514))
                 for i in range(batch_size)]
    prompts = batch_size * [prompt]  #批量prompt
    num_inference_steps = 20  #推理步数
    height = 512
    width = 512 #size
    guidance_scale = 7.5
    return {"prompt": prompts, "generator": generator, "num_inference_steps": num_inference_steps, "height": height, "width": width, "guidance_scale": guidance_scale}
#提示词（prompt）
#负面词（negative_prompt）
#图片宽高（width, height）
#采样步数（num_inference_steps）
#引导强度（guidance_scale）
#生成张数（num_images_per_prompt）



pipe.enable_attention_slicing()

#一次生成4张图片并且以2*2的方式以网格的形式保存
images = pipe(**get_inputs(batch_size=4)).images
grid = make_image_grid(images, 2, 2)

# save the image in the specified directory
# 选择一个文件保存路径
output_dir = "/data1/sdtest/"
output_filename = "2.1.png"
output_path = os.path.join(output_dir, output_filename)
# 将图像数据保存为图像文件
grid.save(output_path)
print("生成的图像已保存到:", output_path)
```

----

###  最后我们希望sd的所有部分都可以自己实现包括encoder/decoder/latent/voe等

对各位部件以及每行代码的功能的详细解释可见stablediffusion官网。

### [较为系统的结合源码的阐释-click here](https://huggingface.co/blog/stable_diffusion#writing-your-own-inference-pipeline)

```python
from tqdm.auto import tqdm
from diffusers import UniPCMultistepScheduler
from PIL import Image
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
import os

#sd几大模块的设定
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", use_safetensors=True)
tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder", use_safetensors=True)
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet", use_safetensors=True)
scheduler = UniPCMultistepScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

torch_device = "cuda"
vae.to(torch_device)
text_encoder.to(torch_device)
unet.to(torch_device)

prompt = ["1girl, aqua eyes, baseball cap, blonde hair, closed mouth, earrings, green background, hat, hoop earrings, jewelry, looking at viewer, shirt, short hair, simple background, solo, upper body, yellow shirt,high quality,masterpiece"]
height = 512  # default height of Stable Diffusion
width = 512  # default width of Stable Diffusion
num_inference_steps = 25  # Number of denoising steps
guidance_scale = 7.5  # Scale for classifier-free guidance
# Seed generator to create the inital latent noise
generator = torch.manual_seed(0)
batch_size = len(prompt)


text_input = tokenizer(
    prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
)

with torch.no_grad():
    text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

max_length = text_input.input_ids.shape[-1]
uncond_input = tokenizer([""] * batch_size, padding="max_length",max_length=max_length, return_tensors="pt")
uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

latents = torch.randn(
    (batch_size, unet.in_channels, height // 8, width // 8),
    generator=generator,
)
latents = latents.to(torch_device)

latents = latents * scheduler.init_noise_sigma

scheduler.set_timesteps(num_inference_steps)
for t in tqdm(scheduler.timesteps):
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)

    latent_model_input = scheduler.scale_model_input(
        latent_model_input, timestep=t)

    # predict the noise residual
    with torch.no_grad():
        noise_pred = unet(latent_model_input, t,
                          encoder_hidden_states=text_embeddings).sample

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * \
        (noise_pred_text - noise_pred_uncond)

    # compute the previous noisy sample x_t -> x_t-1
    latents = scheduler.step(noise_pred, t, latents).prev_sample


# scale and decode the image latents with vae
latents = 1 / 0.18215 * latents
with torch.no_grad():
    image = vae.decode(latents).sample

image = (image / 2 + 0.5).clamp(0, 1).squeeze()
image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
images = (image * 255).round().astype("uint8")
image = Image.fromarray(image)
output_dir = "/data1/sdtest/"
output_filename = "2.1.png"
output_path = os.path.join(output_dir, output_filename)
# 将图像数据保存为图像文件
image.save(output_path)
print("生成的图像已保存到:", output_path)
```