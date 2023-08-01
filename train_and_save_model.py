import os
from nick import track_user,user_id_list
import json

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

    

    for key,value in track_user.items():
        if value["status"] == "pending":
            print(f"User {key} is in pending state")
            return 


    # check if if any user_id is in the model_status.json file
    # Assuming model_status.json is a file path
    with open('model_status.json') as f:
        model_status = json.load(f)
    
    if not model_status:  # This is equivalent to checking if a dictionary is empty
        print("No user_id in the model_status.json file")
        return
    


    
    
    # extract the first user_id from the json file
    if isinstance(model_status["user_id"], list):
        user_id = model_status["user_id"][0]
        model_status["user_id"].pop(0)

    if track_user[user_id]["train_model"] == "successfull" and track_user[user_id]["save_model"] == "successfull" and track_user[user_id]["generate_image"] == "successfull":
        print("This user_id have been trained and saved and generated")
    try:
        if track_user[user_id]["train_model"] == "successfull" and track_user[user_id]["save_model"] == "successfull" and track_user[user_id]["generate_image"] == "successfull":
            raise Exception("This user_id have been trained and saved and generated")
        
        from utility import train_model, save_model , generated_image_store_dir
        
        train_status = train_model(user_id)
        save_status = save_model(user_id,track_user)
        generate_status = generated_image_store_dir(user_id,track_user)
        track_user[user_id]["status"] = "pending"
        if ((train_status == "Model trained successfully") and (save_status=="Model saved successfully") and (generate_status=="Images generated successfully")):
            #remove_status_text('model_saving_status.txt', 'Files saved successfully')
            track_user[user_id]["train_model"] = "successfull"
            track_user[user_id]["save_model"] = "successfull"
            track_user[user_id]["generate_image"] = "successfull"
            track_user[user_id]["status"] = "idle"
            with open("model_status.json", "w") as f:
                json.dump(model_status, f, indent=4)

    except Exception as e:
        print('--->', e)
        track_user[user_id]["status"] = "idle"
        with open("model_status.json", "r") as f:
            data = json.load(f)
            data["user_id"].append(user_id)
        with open("model_status.json", "w") as f:
            json.dump(data, f, indent=4)





if __name__ == '__main__':
    main()