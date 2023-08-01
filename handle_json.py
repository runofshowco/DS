import json
from pathlib import Path
import os
import string

tmp = __file__
tmp = "/".join(tmp.split('/')[0:-1])
json_path = os.path.join(tmp, 'model_status.json')

print(json_path)

def update_data(data):
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)

def get_data():
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        
        print(data)
        
        if data.get('user_id_list') is None:
            data['user_list'] = []
        
        if data.get('track_id') is None:
            data['track_user'] = {}
        
        return data
    except Exception as e:
        print(e)
        return {"user_id_list": [], "track_user": {}}


    