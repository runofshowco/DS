from torchvision.utils import save_image
from werkzeug.utils import secure_filename
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDPMScheduler
from torch import autocast
import torch
import json
import os
import uuid
from PIL import Image
from natsort import natsorted
from glob import glob
import shlex
from flask import jsonify
import io
import subprocess
import shutil
from threading import Thread
import time

import random

import requests
from base64 import encodebytes

from task_queue import Task_Queue as tq
from init import app, db



PROJECT_DIR = "/".join(str(__file__).split('/')[0:-1])
UPLOAD_FOLDER = 'data/'
MODEL_NAME = "dreamlike-art/dreamlike-photoreal-2.0"
OUTPUT_DIR = os.path.join(PROJECT_DIR,"stable_diffusion_weights")
WEIGHTS_DIR ="stable_diffusion_weights"

print(WEIGHTS_DIR)

from handle_json import get_data, update_data

training_steps = 1201

import os
import re
from PIL import Image, ImageOps

def crop_image(im):
    im = ImageOps.exif_transpose(im)
    (x, y) = im.size
    min_sz = min(x, y)
    x_crop = max(0, x - min_sz) / 2;
    y_crop = max(0, y - min_sz) / 3
    left = x_crop
    right = x - x_crop
    top = y_crop
    bottom = y - y_crop * 2
    img1 = im.crop((left, top, right, bottom))
    return img1.resize((512, 512))
    
def resize_all(path):
    all_files = os.listdir(path)
    image_list = []
    idx = 1
    os.mkdir(os.path.join(path, 'cropped'))
    for x in all_files:
        if len(re.findall('.jpg|.png|.jpeg', x)) > 0:
            im = Image.open(os.path.join(path, x))
            cropped = crop_image(im)
            cropped.save(os.path.join(path,'cropped', x))
                
    return

def train_model(user_id):

    #Remove existing files
    #remove_files_in_directory(app.config['OUTPUT_DIR'])
    #remove_files_in_directory(app.config['WEIGHTS_DIR'])
    
    concepts_list = [
    {
        "instance_prompt":      f"photo of X123 person",
        "class_prompt":         "photo of a person",
        "instance_data_dir":    os.path.join(PROJECT_DIR, 'data', user_id, user_id, 'cropped'),
        "class_data_dir":       os.path.join(PROJECT_DIR, 'data', "person")
    }
    ]

    

    

    with open(os.path.join(PROJECT_DIR, "data", user_id, "concepts_list.json"), "w") as f:
        json.dump(concepts_list, f, indent=4)
        
    resize_all(os.path.join(PROJECT_DIR, 'data', user_id, user_id))

    output_dir = os.path.join(PROJECT_DIR, "data", user_id, "stable_diffusion_weights", user_id)
    
    # cmd = f'''python3 {PROJECT_DIR}/train_dreambooth.py \
    # --pretrained_model_name_or_path={MODEL_NAME} \
    # --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
    # --output_dir="{output_dir}" \
    # --with_prior_preservation --prior_loss_weight=1.0 \
    # --seed=1337 \
    # --resolution=512 \
    # --train_batch_size=1 \
    # --train_text_encoder \
    # --mixed_precision="fp16" \
    # --use_8bit_adam \
    # --gradient_accumulation_steps=1 \
    # --learning_rate=1e-6 \
    # --lr_scheduler="constant" \
    # --lr_warmup_steps=30 \
    # --num_class_images=69 \
    # --sample_batch_size=4 \
    # --max_train_steps={training_steps} \
    # --save_interval={training_steps} \
    # --save_sample_prompt="photo of X123 person" \
    # --concepts_list="{os.path.join(PROJECT_DIR, "data", user_id, "concepts_list.json")}"'''

    cmd= f'''python3 {PROJECT_DIR}/train_dreambooth.py \
  --pretrained_model_name_or_path={MODEL_NAME} \
  --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
  --output_dir={output_dir} \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --seed=1337 \
  --resolution=512 \
  --train_batch_size=1 \
  --train_text_encoder \
  --mixed_precision="fp16" \
  --use_8bit_adam \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=30 \
  --num_class_images=69 \
  --sample_batch_size=3 \
  --max_train_steps={training_steps} \
  --save_interval={training_steps} \
  --save_sample_prompt="photo of X123 person" \
  --concepts_list="{os.path.join(PROJECT_DIR, "data", user_id, "concepts_list.json")}"'''

    

    # Training script here, for example:
    print("Training model...")

    # try:
    #     result = subprocess.run(cmd, check=True, shell=True, text=True, capture_output=True)
    #     print(result.stdout)
    # except subprocess.CalledProcessError as e:
    #     print(f'Error occurred: {e}')

    args = shlex.split(cmd)
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    stdout, stderr = process.communicate()

    print(cmd)
    print(stdout, stderr)

    # # try:
    # #     process = subprocess.run(cmd, shell=True, check=True, text=True, capture_output=True)
    # # except subprocess.CalledProcessError as e:
    # #     print(f"Command '{e.cmd}' returned non-zero exit status {e.returncode}.")
    # #     print("STDOUT:")
    # #     print(e.stdout)
    # #     print("STDERR:")
    # #     print(e.stderr)
    # #     return "Model not trained successfully"

    if process.returncode != 0:
        #write_to_txt_file("model_saving_status.txt", "Model not trained successfully")
        #return {'error': stderr.decode()}, 400
        print("Model not trained successfully")
        return "Model not trained successfully"
    else:
        # Once training is done, write to txt file
        #write_to_txt_file("model_saving_status.txt", "Model trained successfully")
        #return {'message': stdout.decode()}, 200
        print("Model trained successfully")
        return "Model trained successfully"
    

def save_model(user_id):

    # Remove existing files
    #remove_files_in_directory(app.config['OUTPUT_DIR'])
    #remove_files_in_directory(app.config['WEIGHTS_DIR'])



    ckpt_path = f"{PROJECT_DIR}/data/{user_id}/stable_diffusion_weights/{user_id}/{training_steps}" + "/model.ckpt"

    half_arg = ""
    fp16 = True
    if fp16:
        half_arg = "--half"

    cmd = f'''python {PROJECT_DIR}/convert_diffusers_to_original_stable_diffusion.py \
    --model_path="{PROJECT_DIR}/data/{user_id}/stable_diffusion_weights/{user_id}/{training_steps}" \
    --checkpoint_path={ckpt_path} {half_arg}
    '''

    print(cmd)

    # Model saving script here, for example:
    print("Saving model...")

    args = shlex.split(cmd)
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    stdout, stderr = process.communicate()
    print("e----",stdout,stderr)
    # try:
    #     process = subprocess.run(cmd, shell=False, check=True, text=True, capture_output=True)
    # except subprocess.CalledProcessError as e:
    #     print(f"Command '{e.cmd}' returned non-zero exit status {e.returncode}.")
    #     print("STDOUT:")
    #     print(e.stdout)
    #     print("STDERR:")
    #     print(e.stderr)
    #     return "Model not trained successfully"

    if process.returncode != 0:
        #write_to_txt_file("model_saving_status.txt", "Model no saved successfully")
        #return {'error': stderr.decode()}, 400
        print("Model not saved successfully")
        return "Model not saved successfully"
    else:
        # Once model is saved, write to txt file
        #write_to_txt_file("model_saving_status.txt", "Model saved successfully")
        #return {'message': stdout.decode()}, 200
        print("Model saved successfully")
        return "Model saved successfully"



def generated_image_store_dir(user_id):
    # generate images using the trained model .ckpt file which is saved in the stable_diffusion_weights folder
    # and save the generated images in the person folder
    ckpt_path = f"{PROJECT_DIR}/data/{user_id}/stable_diffusion_weights/{user_id}/{training_steps}"
    data = tq.get_one_by(id=user_id)
    prompt = data["prompt"]
    negative_prompt = data["negative_prompt"]
    guidance_scale = data["guidance_scale"]
    seeds = json.loads(data.get("seeds") if data.get("seeds") else "[]" )
    pipe = StableDiffusionPipeline.from_pretrained(ckpt_path, safety_checker=None, torch_dtype=torch.float16).to("cuda")
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()

    seed_length = len(seeds)

    for i in range(max(0, 4 - seed_length)):
        seeds.append(random.randint(1e15, 1e16 -1))
    print(seeds)
    chosen_seeds = random.sample(range(0, len(seeds)), 4)
    images = []
    for x in chosen_seeds:
        g_cuda = torch.Generator(device='cuda')
        g_cuda.manual_seed(int(x))
    
        
    
        prompt = prompt
        negative_prompt = negative_prompt
        num_samples = 1
        guidance_scale = float(guidance_scale)
        num_inference_steps = 50
        height = 512
        width = 512
    
        with autocast("cuda"), torch.inference_mode():
            single = pipe(
            prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_samples,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=g_cuda
        ).images[0]
        images.append(single)    

    #save_image(images[0], 'image.png')

    # # Convert the image to bytes
    # img_byte_arr = io.BytesIO()
    # images[0].save(img_byte_arr, format='PNG')
    # img_byte_arr = img_byte_arr.getvalue()

    try:
        for i , img in enumerate(images):
            # # img_pil = Image.fromarray(img.permute(1,2,0).cpu().numpy())
            # img_pil = Image.fromarray(img.permute(1,2,0).cpu().numpy())
            img.save(f"{PROJECT_DIR}/data/{user_id}/output/{i}.png")
        return "Images generated successfully"
    except Exception as e:
        print(e)
        return "Images not generated successfully"

    # now save the images into the user_id/person folder
    # save the images in the person folder

def clear_model_files(user_id):
    ckpt_path = f"{PROJECT_DIR}/data/{user_id}/stable_diffusion_weights/"

    is_exists = os.path.isdir(ckpt_path)

    if is_exists == True:
        shutil.rmtree(ckpt_path)
    
    return True

def encode_image(image_path):
    pil_img = Image.open(image_path, mode='r') # reads the PIL image
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format='PNG') # convert the PIL image to byte array
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
    return encoded_img

def send_image(track_id):
    images = []
    for filename in natsorted(glob(f"{PROJECT_DIR}/data/{track_id}/output/*.png")):
        images.append(encode_image(filename))
    
    data = {"message":"Generation Successfull!", "images": images, "track_id": track_id}
    headers = {'x-project': "bmlja2ZhcnJlbGw6ZzFuOHY4emkyNGxxemV0ZndxdmZ6Y2J3bTg4emFsaw=="}

    # print(data)

    url = 'https://api.aieventbooth.com/v3/api/lambda/nickfarrell/webhook'
    r = requests.post(url, json= data, headers=headers)

    print(r.__dict__['_content'])
    
    return True




    
