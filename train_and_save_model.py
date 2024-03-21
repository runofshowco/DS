import os
import json
import traceback
from handle_json import get_data, update_data
import time
from task_queue import Task_Queue as tq
from init import app, db

# Get the absolute path of the directory the script is located in
abs_dir_path = os.path.dirname('/workspace/ds_nickfarrell/')

# Change the working directory to the directory the script is in
os.chdir(abs_dir_path)

def main():
    #remove_status_text('model_saving_status.txt', 'Files saved successfully')
    # loop through the track_user then find if the user_id have null value in the "train_model" and "save_model" and "generate_image"
    # check which user_id not not trained and saved and generated
    # pop the first element from the list and train and save and generate
    # first check if the user_id is null or not

    with app.app_context():
        data = tq.get_by(status='idle')
        print(data)

        if len(data) == 0:
            print("No task in queue")
            return

        threshold = 1
        total_running = 0

        pending = tq.get_by(status='pending')

        if len(pending) >= threshold:
            print(f'''Already training {threshold} training''')
            return 

        

        data = data[0]
        print(data['id'])

        tq.update(data['id'], status='pending')


        try:
            strating_time= time.time()
            tq.update(data['id'], status='pending', start= time.time()) 
            from utility import train_model, save_model , generated_image_store_dir, clear_model_files
            train_status = None
            save_status = None
            generate_status = None

            if not data['train_model']:
                start = time.time()
                train_status = train_model(data['id'])
                tq.update(data['id'], train_model= 1)
                tq.update(data['id'], training_time = time.time() - start)
            
            
            if not data['save_model']:
                start = time.time()
                save_status = save_model(data['id'])
                tq.update(data['id'], save_model= 1)
                tq.update(data['id'], model_saving_time = time.time() - start)
            
            if not data['generate_image']:
                start = time.time()
                generate_status = generated_image_store_dir(data['id'])
                tq.update(data['id'], generate_image= 1)
                tq.update(data['id'], generating_time = time.time() - start)

            if not data['model_cleared']:
                
                clear_model_files(data['id'])
                tq.update(data['id'], model_cleared= 1)
            
            tq.update(data['id'], status= 'completed', export='ready', end=time.time(), processing_time= time.time() - strating_time)
        

        except Exception as e:
            print('--->', e)
            traceback.print_exc()
            tq.update(data['id'], status= 'idle')
        



if __name__ == '__main__':
    main()