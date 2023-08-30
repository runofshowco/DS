from init import db

class Task_Queue(db.Model):
    id = db.Column(db.String, primary_key=True)
    status = db.Column(db.String, nullable=True)
    upload_image = db.Column(db.Integer, nullable=True, default=0)
    train_model = db.Column(db.Integer, nullable=True, default=0)
    save_model = db.Column(db.Integer, nullable=True, default=0)
    generate_image = db.Column(db.Integer, nullable=True, default=0)
    prompt = db.Column(db.String, nullable=True)
    negative_prompt = db.Column(db.String, nullable=True)
    guidance_scale = db.Column(db.Numeric(5,2))
    model_cleared = db.Column(db.Integer, nullable=True, default=0)
    start = db.Column(db.Numeric(20,4), nullable=True)
    training_time = db.Column(db.Numeric(20,4), nullable=True)
    model_saving_time = db.Column(db.Numeric(20,4), nullable=True)
    generating_time = db.Column(db.Numeric(20,4), nullable=True)
    end = db.Column(db.Numeric(20,4), nullable=True)
    processing_time = db.Column(db.Numeric(20,4), nullable=True)
    arrival_time = db.Column(db.Numeric(20,4), nullable=True)
    export = db.Column(db.String, nullable=True)
    seeds = db.Column(db.String, nullable=True)
    @property
    def serialize(self):
       """Return object data in easily serializable format"""
       return {
            "id"         : self.id,
            "status"  : self.status,
            "upload_image": self.upload_image,
            "train_model": self.train_model,
            "save_model": self.save_model,
            "generate_image": self.generate_image,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "guidance_scale": 13.0,
            "status": self.status,
            "model_cleared": self.model_cleared,
            "start": self.start,
            "training_time": self.training_time,
            "model_saving_time": self.model_saving_time,
            "generating_time": self.generating_time,
            "end": self.end,
            "processing_time": self.processing_time,
            "arrival_time": self.arrival_time,
            "seeds": self.seeds,
            "export": self.export
       }

    @classmethod
    def get_single(cls, id):
        return db.session.execute(db.select(Task_Queue).filter_by(id=id)).scalars().first().serialize
    
    @classmethod
    def get_by(cls, **kwargs):
        try:
            return [x.serialize for x in db.session.execute(db.select(Task_Queue).filter_by(**kwargs).order_by(Task_Queue.arrival_time)).scalars().all()]
        except Exception as e:
            print(e)
            return []
    
    @classmethod
    def get_one_by(cls, **kwargs):
        try:
            return db.session.execute(db.select(Task_Queue).filter_by(**kwargs).order_by(Task_Queue.arrival_time)).scalars().first().serialize
        except:
            return None

    @classmethod
    def update(cls, id, **kwargs):
        x = db.session.execute(db.select(Task_Queue).filter_by(id=id)).scalars().first()
        # x.status = 'pending'
        for key, value in kwargs.copy().items():
            x.__setattr__(key, value) 
        db.session.commit()
    
