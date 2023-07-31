from nick import train_model, save_model , check_file_status,remove_status_text,write_to_txt_file
import os
from nick import track_user,generate_image

# Get the absolute path of the directory the script is located in
abs_dir_path = os.path.dirname(os.path.abspath('/workspace/datascience_learning/nick_farrell/model_saving_status.txt'))

# Change the working directory to the directory the script is in
os.chdir(abs_dir_path)

def main():
    #remove_status_text('model_saving_status.txt', 'Files saved successfully')
    # loop through the track_user then find if the user_id have null value in the "train_model" and "save_model" and "generate_image"
    for user_id,value in track_user.items():
        if value["train_model"] == None and value["save_model"] == None and value["generate_image"] == None:
            try:
                train_status = train_model()
                save_status = save_model()
                generate_status = generate_image(user_id)
                if (train_status == "Model trained successfully") and (save_status=="Model saved successfully"):
                    remove_status_text('model_saving_status.txt', 'Files saved successfully')
                    value["train_model"] = "successfull"
                    value["save_model"] = "successfull"
            except Exception as e:
                print('--->', e)
                write_to_txt_file('model_saving_status.txt', 'Files saved successfully')





if __name__ == '__main__':
    main()