mkdir -p /workspace/ds_nickfarrell/data/
rm -rf /workspace/ds_nickfarrell/data/person
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1SY8ZIQ2BbB-eJfz6nP74NLRXqFGy4Ehr' -O /workspace/person.zip
apt-get update
apt install zip unzip
unzip /workspace/person.zip -d /workspace/ds_nickfarrell/data/
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 /workspace/ds_nickfarrell/create_tables.py
bash /workspace/ds_nickfarrell/scripts/setup.sh
