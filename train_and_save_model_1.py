from nick import train_model, save_model , check_file_status,remove_status_text,write_to_txt_file
import os

# Get the absolute path of the directory the script is located in
abs_dir_path = os.path.dirname(os.path.abspath('/workspace/datascience_learning/nick_farrell/model_saving_status.txt'))

# Change the working directory to the directory the script is in
os.chdir(abs_dir_path)

def main():
    remove_status_text('model_saving_status.txt', 'Files saved successfully')
    try:
        train_status = train_model()
        if train_status == "Model trained successfully":
            save_status = save_model()
            if save_status == "Model saved successfully":
                remove_status_text('model_saving_status.txt', 'Files saved successfully')
        else:
            print("Model training was not successful. Not saving model.")
            write_to_txt_file('model_saving_status.txt', 'Model not trained successfully')
    except Exception as e:
        print('--->', e)
        write_to_txt_file('model_saving_status.txt', 'Files saved successfully')


if __name__ == '__main__':
    main()
