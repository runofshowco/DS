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
user_id_list = []
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








    

@app.route('/upload', methods=['POST','GET'])
def upload_file():
    global UPLOAD_FOLDER,user_id_list

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

    #track_user[user_id] = {"upload_image": None, "train_model": None, "save_model": None, "generate_image": None,"prompt":None,"negetive_prompt":None,"guidance_scale":None}

    # save the prompt, negetive_prompt, guidance_scale in the track_user[user_id]
    # track_user[user_id]["prompt"] = prompt
    # track_user[user_id]["negetive_prompt"] = negetive_prompt
    # track_user[user_id]["guidance_scale"] = guidance_scale

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
    track_user[user_id]["train_model"] = None
    track_user[user_id]["save_model"] = None
    track_user[user_id]["generate_image"] = None
    track_user[user_id]["prompt"] = prompt
    track_user[user_id]["negetive_prompt"] = negetive_prompt
    track_user[user_id]["guidance_scale"] = guidance_scale
    track_user[user_id]["status"] = "idle"

    # add the user_id into the list
    user_id_list.append(user_id)


        
    
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
    return f'Files uploaded successfully. Your user_id is {user_id} {track_user} {user_id_list}'


#----------train_dreambooth.py---------#


#---------training finished ----------#

#------------ model saving -------#

# -------------------- model saving finished ----------#

#-------- generating images -----------#
@app.route('/generate_images', methods=['POST','GET'])
def generate_image():
    global MODEL_NAME,OUTPUT_DIR,WEIGHTS_DIR,track_user
    user_id = request.form["user_id"]

    if user_id not in track_user.keys():
        return jsonify({"track_user":track_user}), 400

    if ((track_user[user_id]["train_model"] == "successfull") and (track_user[user_id]["save_model"] == "successfull") and (track_user[user_id]["generate_image"] == "successfull") and (track_user[user_id]["upload_image"] == "successfull")):
        try:
            for filename in natsorted(glob(f"data/{user_id}/output/*.png")):
                print(filename)
                return send_file(filename, mimetype='image/png')
        except Exception as e:
            print(e)
            return jsonify({"track_user":track_user}), 400  

    return "No valid response", 404  # Add this line




#----------generated images finished -------#

if __name__ == '__main__':
    app.run(port=5110,debug=True)

