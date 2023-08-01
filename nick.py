#Create a Flask API for stable diffusion
from flask import Flask, request, jsonify, send_file
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
from flask_cors import CORS
import io
import subprocess
import shutil
from threading import Thread
# Folder to store uploaded images
UPLOAD_FOLDER = 'data/'
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = "stable_diffusion_weights"
WEIGHTS_DIR = "stable_diffusion_weights"
track_user = {}

app = Flask(__name__)

CORS(app) # Adds CORS to all routes

app.config['MODEL_NAME'] = MODEL_NAME
app.config['OUTPUT_DIR'] = OUTPUT_DIR
app.config['WEIGHTS_DIR'] = WEIGHTS_DIR
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def check_file_status(file_path,search_item):
    with open(file_path, 'r') as file:
        contents = file.read()
        if search_item in contents:
            return True
        else:
            return False

def remove_status_text(file_path, text_to_remove):
    with open(file_path, 'r') as file:
        contents = file.read()

    # Replace the target text
    contents = contents.replace(text_to_remove, '')

    # Write the modified contents back to the file
    with open(file_path, 'w') as file:
        file.write(contents)



def remove_files_in_directory(directory):
    # Check if directory exists
    if os.path.exists(directory):
        # Check if there are files in the directory
        if len(os.listdir(directory)) != 0:
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
        else:
            print(f'The directory {directory} is empty, no files to remove.')
    else:
        print(f'The directory {directory} does not exist.')




def write_to_txt_file(file_path, text):
    with open(file_path, "a") as f:
        f.write("\n" + text)

def train_model(user_id):
    global MODEL_NAME,OUTPUT_DIR,WEIGHTS_DIR,track_user

    #Remove existing files
    #remove_files_in_directory(app.config['OUTPUT_DIR'])
    #remove_files_in_directory(app.config['WEIGHTS_DIR'])

    concepts_list = [
    {
        "instance_prompt":      f"photo of {user_id} person",
        "class_prompt":         "photo of a person",
        "instance_data_dir":    f"data/{user_id}",
        "class_data_dir":       f"data/{user_id}/person"
    }
    ]

    with open(f"data/{user_id}/concepts_list.json", "w") as f:
        json.dump(concepts_list, f, indent=4)
    
    cmd = f'''python3 train_dreambooth.py \
    --pretrained_model_name_or_path={MODEL_NAME} \
    --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
    --output_dir="{OUTPUT_DIR}/{user_id}" \
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
    --concepts_list="data/{user_id}/concepts_list.json"'''

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


def save_model(user_id):
    global MODEL_NAME,OUTPUT_DIR,WEIGHTS_DIR,track_user

    # Remove existing files
    #remove_files_in_directory(app.config['OUTPUT_DIR'])
    #remove_files_in_directory(app.config['WEIGHTS_DIR'])



    ckpt_path = f"data/{user_id}/stable_diffusion_weights/{user_id}/1000" + "/model.ckpt"

    half_arg = ""
    fp16 = True
    if fp16:
        half_arg = "--half"
    
    cmd = f'''python convert_diffusers_to_original_stable_diffusion.py \
    --model_path="data/{user_id}/stable_diffusion_weights/{user_id}/1000" \
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
    global track_user
    # generate images using the trained model .ckpt file which is saved in the stable_diffusion_weights folder
    # and save the generated images in the person folder
    prompt = track_user[user_id]["prompt"]
    negative_prompt = track_user[user_id]["negetive_prompt"]
    guidance_scale = track_user[user_id]["guidance_scale"]

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
            img_pil.save(f"data/{user_id}/output/{i}.png")
        return "Images generated successfully"
    except Exception as e:
        print(e)
        return "Images not generated successfully"

    # now save the images into the user_id/person folder
    # save the images in the person folder

    

    

@app.route('/upload', methods=['POST','GET'])
def upload_file():
    global UPLOAD_FOLDER

    # with open('model_saving_status.txt', 'w') as file:
    #     pass

    # make the user_id directory like this
    # user_id = {"1":{"upload_image":"successfull","train_model":"successfull","save_model":"successfull","generate_image":"successfull"}
    # "2":{"upload_image":"successfull","train_model":"successfull","save_model":"successfull","generate_image":"successfull"}
    

    image_files = request.files.getlist('images')
    prompt = request.form["prompt"]
    negetive_prompt = request.form["negetive_prompt"]
    guidance_scale = float(request.form["guidance_scale"])

    

    user_id = str(uuid.uuid4())

    track_user[user_id] = {"upload_image": None, "train_model": None, "save_model": None, "generate_image": None,"prompt":None,"negetive_prompt":None,"guidance_scale":None}

    # save the prompt, negetive_prompt, guidance_scale in the track_user[user_id]
    track_user[user_id]["prompt"] = prompt
    track_user[user_id]["negetive_prompt"] = negetive_prompt
    track_user[user_id]["guidance_scale"] = guidance_scale

    # Make a folder in the data folder with the user_id with all the permission to read write and execute
    os.mkdir(os.path.join(app.config['UPLOAD_FOLDER'], user_id))
    os.mkdir(os.path.join(app.config['UPLOAD_FOLDER'], user_id, user_id))
    os.mkdir(os.path.join(app.config['UPLOAD_FOLDER'], user_id, "person"))
    os.mkdir(os.path.join(app.config['UPLOAD_FOLDER'], user_id, "stable_diffusion_weights"))
    os.mkdir(os.path.join(app.config['UPLOAD_FOLDER'], user_id, "stable_diffusion_weights", user_id))
    os.mkdir(os.path.join(app.config['UPLOAD_FOLDER'], user_id, "stable_diffusion_weights", user_id, "1000"))
    # make the output folder where the images will be stored
    os.mkdir(os.path.join(app.config['UPLOAD_FOLDER'], user_id, "output"))

    # save the images in the user_id/user_id folder
    upload_folder = f"data/{user_id}/{user_id}/"


    for file in image_files:
        filename = secure_filename(file.filename)
        file.save(os.path.join(upload_folder, filename))

    
    track_user[user_id] = {"upload_image":"successfull"}
        
    
    # Add user_id to track_user such that the "upload_image":"successfull" is added to the user_id
    # put all the values 'null' in the track_user[user_id] except the "upload_image":"successfull"
    # then add for same user_id "train_model":"successfull" apppend the data 
    #track_user[user_id]["train_model"] = "successfull"

   

    # this is my folder structure
    # data
    #  - user_id
    #     - user_id
    #          - image1.jpg
    #          - image2.jpg
    #     - person(in this directroy all the trained images will be saved)
    #     - stable_diffusion_weights
    #          - user_id
    #               - 1000
    #                    - model.ckpt
    #write_to_txt_file("model_saving_status.txt", "Files saved successfully")
    return f'Files uploaded successfully. Your user_id is {user_id}'


#----------train_dreambooth.py---------#


#---------training finished ----------#

#------------ model saving -------#

# -------------------- model saving finished ----------#

#-------- generating images -----------#
@app.route('/generate_images', methods=['POST','GET'])
def generate_image():

    global MODEL_NAME,OUTPUT_DIR,WEIGHTS_DIR,track_user

    # get the user_id from the track_user
    user_id = request.form["user_id"]


    return jsonify({"track_user":track_user}), 200

    # first check if the user_id is valid or not
    if user_id not in track_user.keys():
        return jsonify({"track_user":track_user}), 400

    if ((track_user[user_id]["train_model"] == "successfull") and (track_user[user_id]["save_model"] == "successfull") and (track_user[user_id]["generate_image"] == "successfull") and (track_user[user_id]["upload_image"] == "successfull")):
        try:
            # get the images from the folder and return the images
            # images are stored on the data/{user_id}/output folder
            # get the images from the folder and return the images
            for filename in natsorted(glob(f"data/{user_id}/output/*.png")):
                print(filename)
                return send_file(filename, mimetype='image/png')

        except Exception as e:
            print(e)
            return jsonify({"track_user":track_user}), 400




#----------generated images finished -------#

if __name__ == '__main__':
    app.run(port=5110,debug=True)

