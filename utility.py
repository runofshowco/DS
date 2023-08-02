from torchvision.utils import save_image
from werkzeug.utils import secure_filename
from diffusers import StableDiffusionPipeline, DDIMScheduler
from torch import autocast
import torch
import json
import os
import uuid
from PIL import Image
from natsort import natsorted
from glob import glob
import shlex

import io
import subprocess
import shutil
from threading import Thread

PROJECT_DIR = "/".join(str(__file__).split('/')[0:-1])
UPLOAD_FOLDER = 'data/'
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = os.path.join(PROJECT_DIR,"stable_diffusion_weights")
WEIGHTS_DIR = os.path.join(PROJECT_DIR,"stable_diffusion_weights")


def train_model(user_id):

    #Remove existing files
    #remove_files_in_directory(app.config['OUTPUT_DIR'])
    #remove_files_in_directory(app.config['WEIGHTS_DIR'])

    concepts_list = [
    {
        "instance_prompt":      f"photo of {user_id} person",
        "class_prompt":         "photo of a person",
        "instance_data_dir":    os.path.join(PROJECT_DIR, 'data', user_id, user_id),
        "class_data_dir":       os.path.join(PROJECT_DIR, 'data', user_id, "person")
    }
    ]

    

    with open(os.path.join(PROJECT_DIR, "data", user_id, "concepts_list.json"), "w") as f:
        json.dump(concepts_list, f, indent=4)

    output_dir = os.path.join(PROJECT_DIR, "data", user_id, "stable_diffusion_weights", user_id)
    
    cmd = f'''python3 {PROJECT_DIR}/train_dreambooth.py \
    --pretrained_model_name_or_path={MODEL_NAME} \
    --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
    --output_dir="{output_dir}" \
    --revision="fp16" \
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
    --lr_warmup_steps=0 \
    --num_class_images=50 \
    --sample_batch_size=4 \
    --max_train_steps=1000 \
    --save_interval=1000 \
    --save_sample_prompt="photo of {user_id} person" \
    --concepts_list="{os.path.join(PROJECT_DIR, "data", user_id, "concepts_list.json")}"'''

    # Training script here, for example:
    print("Training model...")


    args = shlex.split(cmd)
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    stdout, stderr = process.communicate()

    print(cmd)
    print(stdout, stderr)

    # try:
    #     process = subprocess.run(cmd, shell=True, check=True, text=True, capture_output=True)
    # except subprocess.CalledProcessError as e:
    #     print(f"Command '{e.cmd}' returned non-zero exit status {e.returncode}.")
    #     print("STDOUT:")
    #     print(e.stdout)
    #     print("STDERR:")
    #     print(e.stderr)
    #     return "Model not trained successfully"

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
    

def save_model(user_id,track_user):

    # Remove existing files
    #remove_files_in_directory(app.config['OUTPUT_DIR'])
    #remove_files_in_directory(app.config['WEIGHTS_DIR'])



    ckpt_path = f"{PROJECT_DIR}/data/{user_id}/stable_diffusion_weights/{user_id}/1000" + "/model.ckpt"

    half_arg = ""
    fp16 = True
    if fp16:
        half_arg = "--half"

    cmd = f'''python {PROJECT_DIR}/convert_diffusers_to_original_stable_diffusion.py \
    --model_path="{PROJECT_DIR}/data/{user_id}/stable_diffusion_weights/{user_id}/1000" \
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



def generated_image_store_dir(user_id,track_user):
    # generate images using the trained model .ckpt file which is saved in the stable_diffusion_weights folder
    # and save the generated images in the person folder
    data = get_data()
    prompt = data['track_user'][user_id]["prompt"]
    negative_prompt = data['track_user'][user_id]["negetive_prompt"]
    guidance_scale = data['track_user'][user_id]["guidance_scale"]

    pipe = StableDiffusionPipeline.from_pretrained(WEIGHTS_DIR, safety_checker=None, torch_dtype=torch.float16).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    g_cuda = torch.Generator(device='cuda')
    seed = 5345
    g_cuda.manual_seed(seed)

    prompt = prompt
    negative_prompt = negative_prompt
    num_samples = 4
    guidance_scale = guidance_scale
    num_inference_steps = 50
    height = 512
    width = 768

    with autocast("cuda"), torch.inference_mode():
        images = pipe(
        prompt,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_samples,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=g_cuda
    ).images

    #save_image(images[0], 'image.png')

    # # Convert the image to bytes
    # img_byte_arr = io.BytesIO()
    # images[0].save(img_byte_arr, format='PNG')
    # img_byte_arr = img_byte_arr.getvalue()

    try:
        for i , img in enumerate(images):
            img_pil = Image.fromarray(img.permute(1,2,0).cpu().numpy())
            img_pil.save(f"{PROJECT_DIR}/data/{user_id}/output/{i}.png")
        return "Images generated successfully"
    except Exception as e:
        print(e)
        return "Images not generated successfully"

    # now save the images into the user_id/person folder
    # save the images in the person folder

    