# Set the base directory to the current working directory by default
BASE_DIR=$(pwd)


mkdir -p "${BASE_DIR}/data/"
rm -rf "${BASE_DIR}/data/person"
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1SY8ZIQ2BbB-eJfz6nP74NLRXqFGy4Ehr' -O "${BASE_DIR}/person.zip"
apt-get update
apt install zip unzip
unzip "${BASE_DIR}/person.zip" -d "${BASE_DIR}/data/"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 "${BASE_DIR}/create_tables.py"
bash "${BASE_DIR}/scripts/setup.sh"
