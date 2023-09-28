import subprocess

def train_model():

    cmd=f'''python3 train_dreambooth.py \
  --pretrained_model_name_or_path="dreamlike-art/dreamlike-photoreal-2.0" \
  --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
  --output_dir="data/a9e1bcf2-ae89-40db-bac3-f0ee248c1dda/stable_diffusion_weights/a9e1bcf2-ae89-40db-bac3-f0ee248c1dda" \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --seed=1337 \
  --resolution=512 \
  --train_batch_size=1 \
  --train_text_encoder \
  --mixed_precision="fp16" \
  --use_8bit_adam \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=30 \
  --num_class_images=69 \
  --sample_batch_size=4 \
  --max_train_steps=1207 \
  --save_interval=1207 \
  --save_sample_prompt="photo of X123 person" \
  --concepts_list="/workspace/ds_nickfarrell/data/a9e1bcf2-ae89-40db-bac3-f0ee248c1dda/concepts_list.json"'''

    # print(cmd)
    
    # Training script here, for example:
    print("Training model...")


    try:
        result = subprocess.run(cmd, check=True, shell=True, text=True, capture_output=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f'Error occurred: {e}')

if __name__ == "__main__":
    train_model()