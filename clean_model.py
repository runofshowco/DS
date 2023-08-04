from handle_json import get_data, update_data
import os
PROJECT_DIR = "/".join(str(__file__).split('/')[0:-1])
UPLOAD_FOLDER = 'data/'
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = os.path.join(PROJECT_DIR,"stable_diffusion_weights")
WEIGHTS_DIR ="stable_diffusion_weights"


data = get_data()

new_user_list = []

for key, x in enumerate(data['track_user']):
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], key, key)
    l = len(os.listdir(img_path))

    if l == 0:
        del x[key]
    else:
        new_user_list.append(key)

print(new_user_list)
print(data)