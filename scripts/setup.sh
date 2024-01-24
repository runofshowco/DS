BASE_DIR=$(pwd)

apt-get update;
apt-get install nano cron;
cd "${BASE_DIR}";
source venv/bin/activate;
pkill -f gunicorn; 
gunicorn --bind=0.0.0.0:5110 app:app --timeout=1000 --workers=2 &>/workspace/server.nick &
crontab -u root "${BASE_DIR}/scripts/nick.cron"
service cron start
