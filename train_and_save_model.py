import os
import json
import traceback
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

    if data['track_user'][user_id]["status"] == "completed" and data['track_user'][user_id].get("model_cleared") == "successfull":
        print("This user_id have been trained and saved and generated", user_id)
        update_data(data)
        return
    try:
        data['track_user'][user_id]["status"] = "pending"
        update_data(data)
        from utility import train_model, save_model , generated_image_store_dir, clear_model_files
        
        train_status = None
        save_status = None
        generate_status = None

        if data['track_user'][user_id]["train_model"] != "successfull":
            train_status = train_model(user_id)
            data['track_user'][user_id]["train_model"] = "successfull"
            update_data(data)
        
        if data['track_user'][user_id]["save_model"] != "successfull":
            save_status = save_model(user_id,data['track_user'])
            data['track_user'][user_id]["save_model"] = "successfull"
            update_data(data)
        
        if data['track_user'][user_id]["generate_image"] != "successfull":
            generate_status = generated_image_store_dir(user_id,data['track_user'])
            data['track_user'][user_id]["generate_image"] = "successfull"
            update_data(data)

        if data['track_user'][user_id].get("model_cleared") != "successfull":
            clear_model_files(user_id)
            data['track_user'][user_id].get["model_cleared"] = "successfull"
        
        data['track_user'][user_id]["status"] = "completed"
        # data['user_id_list'].append(user_id)
        update_data(data)

    except Exception as e:
        print('--->', e)
        traceback.print_exc()
        data['user_id_list'].append(user_id)
        data['track_user'][user_id]["status"] = "idle"
        update_data(data)
        





if __name__ == '__main__':
    main()