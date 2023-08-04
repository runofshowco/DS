from handle_json import get_data, update_data
import os
PROJECT_DIR = "/".join(str(__file__).split('/')[0:-1])
UPLOAD_FOLDER = f'{PROJECT_DIR}/data/'
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = os.path.join(PROJECT_DIR,"stable_diffusion_weights")
WEIGHTS_DIR ="stable_diffusion_weights"


data = get_data()

new_user_list = []

for idx in data['track_user'].keys().copy():
    img_path = os.path.join(UPLOAD_FOLDER, idx, idx)
    l = len(os.listdir(img_path))

    if l == 0:
        del data['track_user'][idx]
    else:
        new_user_list.append(idx)

print(new_user_list)
print(data)