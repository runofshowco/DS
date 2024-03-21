from email.policy import default
from flask_sqlalchemy import SQLAlchemy
from flask import Flask
from datetime import datetime
from flask_migrate import Migrate

app= Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite3'
app.config['SQLALCHEMY_TRACK_NOTIFICATIONS'] = False
db= SQLAlchemy(app)

migrate = Migrate(app, db)