import json


def update_data(data):
    with open("model_status.json", "w") as f:
        json.dump(data, f, indent=4)

def get_data():
    try:
        with open("model_status.json", "r") as f:
            data = json.load(f)
        
        if data.get('user_id_list') is None:
            data['user_list'] = []
        
        if data.get('track_id') is None:
            data['track_user'] = {}
        
        return data
    except Exception as e:
        print(e)
        return {"user_id_list": [], "track_user": {}}

    