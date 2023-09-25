apt-get update;
apt-get install nano cron;
cd /workspace/ds_nickfarrell/;
source venv/bin/activate;
pkill -f gunicorn; 
gunicorn --bind=0.0.0.0:5110 app:app --timeout=1000 --workers=2 &>/workspace/server.nick &
crontab -u root /workspace/ds_nickfarrell/scripts/nick.cron
service cron start
