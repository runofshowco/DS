import os
import json
import traceback
from handle_json import get_data, update_data
import time
from task_queue import Task_Queue as tq
from init import app, db

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

    with app.app_context():
        data = tq.get_by(export='ready')

        if(len(data) == 0):
            print("Export Queue Empty!")
            return
        
        data = data[0]

        try:
            tq.update(data['id'], export='pending')
            from utility import send_image
            send_image(data['id'])
            tq.update(data['id'], export='done')
        except Exception as e:
            print('--->', e)
            traceback.print_exc()
            tq.update(data['id'], export= 'ready')

if __name__ == '__main__':
    main()