import os
import json
import traceback
from handle_json import get_data, update_data
import time
from task_queue import Task_Queue as tq
from init import app, db


def main():

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