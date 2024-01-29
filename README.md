
# Project Setup Guide

This guide provides step-by-step instructions on how to set up the project from the `ds_nickfarrell` Git repository. Please follow these steps to ensure a correct setup.

## 1. Clone the Git Repository

First, clone the repository from the provided URL. You can do this by opening your terminal and running the following command:

```bash
git clone http://23.29.118.76:3000/mkdlabs/ds_nickfarrell.git
```

This will create a `ds_nickfarrell` directory with the project files in your current working directory.


## 2. Run init.sh

With the project path set, you can now run the `init.sh` script. In your terminal, navigate to the project base directory and execute the following command:

```bash
sudo -s
bash scripts/init.sh
```

This script initializes the project environment and start the server. This command is only for the first time.

## 4. Run setup.sh (Optional)

Run the `setup.sh` script to run the server and cronjob. It's included in `init.sh` as well, so, if you run `init.sh`, no need to run it. However, when you restart the server or your computer, you should run this script. Ensure you're in the project base directory and execute:

```bash
sudo -s
bash scripts/setup.sh
```

This script will perform the necessary setup tasks defined for the project.

# API Documentation

This part describes the API Endpoints of the application

## Routes

### 1. Upload Route: `/upload/` [POST]

This route is used to upload pictures of a specific person, along with a prompt for generating a themed avatar, and some hyperparameters for customization.

#### Parameters:

- `images`: Array of images of the specific person.
- `prompt`: String for the specific theme-based avatar. Use "X123" as an identifier for the specific person.
- `negative_prompt`: (Optional) String to explicitly exclude certain elements from the generated avatar.
- `guidance_scale`: A number from 0 to 20. Higher values adhere more closely to the prompt, while lower values allow for more creativity.
- `seeds`: (Optional) Array of numbers indicating the initial state for inference. A random number from this array will be chosen. If left empty, a random number is used.

#### Return Parameters:

- `message`: String describing the status or message about the request.
- `track_id`: A unique ID used to track the progress of the request and to fetch the generated images.

### 2. Get Images Route: `/get_images` [POST]

This route is used to get the generated images about the avatar or the status of the image processing. 

#### Parameters:

- `track_id`: The track ID for the specific request you want to get image for.


#### Return Parameters:

- `message`: String describing the status or message about the request.
- `details`: Object with some parameters indicating the status of the request.
- `images`: The base64 image array is included and will be accessible upon completion of the request.