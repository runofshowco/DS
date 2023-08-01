import os
from nick import track_user,user_id_list
import json

from handle_json import get_data, update_data

# Get the absolute path of the directory the script is located in
abs_dir_path = os.path.dirname(os.path.abspath('/workspace/nickfarrell'))

# Change the working directory to the directory the script is in
os.chdir(abs_dir_path)

def main():
    #remove_status_text('model_saving_status.txt', 'Files saved successfully')
    # loop through the track_user then find if the user_id have null value in the "train_model" and "save_model" and "generate_image"
    # check which user_id not not trained and saved and generated
    # pop the first element from the list and train and save and generate
    # first check if the user_id is null or not

    data = get_data()
    

    for key,value in data['track_user'].items():
        if value["status"] == "pending":
            print(f"User {key} is in pending state")
            return 


    # check if if any user_id is in the model_status.json file
    # Assuming model_status.json is a file path
    
    # check if the user_id_list is empty or not
    if len(data['user_id_list']) == 0:
        print("User id list is empty")
        return


    
    
    user_id = data['user_id_list'].pop(0)

    print(data['track_user'])

    if data['track_user'][user_id]["train_model"] == "successfull" and data['track_user'][user_id]["save_model"] == "successfull" and data['track_user'][user_id]["generate_image"] == "successfull":
        print("This user_id have been trained and saved and generated")
    try:
        if data['track_user'][user_id]["train_model"] == "successfull" and data['track_user'][user_id]["save_model"] == "successfull" and data['track_user'][user_id]["generate_image"] == "successfull":
            raise Exception("This user_id have been trained and saved and generated")
        
        from utility import train_model, save_model , generated_image_store_dir
        
        train_status = train_model(user_id)
        save_status = save_model(user_id,track_user)
        generate_status = generated_image_store_dir(user_id,track_user)
        data['track_user'][user_id]["status"] = "pending"
        if ((train_status == "Model trained successfully") and (save_status=="Model saved successfully") and (generate_status=="Images generated successfully")):
            #remove_status_text('model_saving_status.txt', 'Files saved successfully')
            data['track_user'][user_id]["train_model"] = "successfull"
            data['track_user'][user_id]["save_model"] = "successfull"
            data['track_user'][user_id]["generate_image"] = "successfull"
            data['track_user'][user_id]["status"] = "idle"
        
        update_data(data)

    except Exception as e:
        print('--->', e)
        data['user_id_list'].append(user_id)
        data['track_user'][user_id]["status"] = "idle"
        update_data(data)
        





if __name__ == '__main__':
    main()