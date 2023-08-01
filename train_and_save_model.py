import os
from nick import track_user,user_id_list

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
    if len(user_id_list) == 0:
        print("All the user_id have been trained and saved and generated")
        return 
    user_id = user_id_list[0]
    user_id_list.pop(0)
    if track_user[user_id]["train_model"] == "successfull" and track_user[user_id]["save_model"] == "successfull" and track_user[user_id]["generate_image"] == "successfull":
       for i in range(len(user_id_list)):
              if track_user[user_id_list[i]]["train_model"] == "successfull" and track_user[user_id_list[i]]["save_model"] == "successfull" and track_user[user_id_list[i]]["generate_image"] == "successfull":
                user_id_list.pop(i)
    try:
        if track_user[user_id]["train_model"] == "successfull" and track_user[user_id]["save_model"] == "successfull" and track_user[user_id]["generate_image"] == "successfull":
            raise Exception("This user_id have been trained and saved and generated")
        
        from utility import train_model, save_model , generated_image_store_dir
        train_status = train_model(user_id)
        save_status = save_model(user_id,track_user)
        generate_status = generated_image_store_dir(user_id,track_user)
        if ((train_status == "Model trained successfully") and (save_status=="Model saved successfully") and (generate_status=="Images generated successfully")):
            #remove_status_text('model_saving_status.txt', 'Files saved successfully')
            track_user[user_id]["train_model"] = "successfull"
            track_user[user_id]["save_model"] = "successfull"
            track_user[user_id]["generate_image"] = "successfull"
    except Exception as e:
        print('--->', e)
        user_id_list.append(user_id)





if __name__ == '__main__':
    main()