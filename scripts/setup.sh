apt-get update;
apt-get install nano cron;
cd /workspace/nickfarrell/;
source venv/bin/activate;
pkill -f gunicorn; 
gunicorn --bind=0.0.0.0:5110 app:app --timeout=1000 --workers=2 &>/workspace/server.nick &
crontab -u root /workspace/nick.cron
service cron start
