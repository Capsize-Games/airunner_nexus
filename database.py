import base64

from fernet import Fernet
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


class Txt2ImgSample(Base):
    __tablename__ = 'txt2img_samples'
    id = Column(Integer, primary_key=True)
    prompt_id = Column(Integer, ForeignKey('prompts.id'))
    negative_prompt_id = Column(Integer)
    sample = Column(String)
    seed = Column(Integer)
    width = Column(Integer)
    height = Column(Integer)
    steps = Column(Integer)
    scale = Column(String)
    model = Column(String)
    scheduler = Column(String)
    created_at = Column(DateTime)


class Img2ImgSample(Base):
    __tablename__ = 'img2img_samples'
    id = Column(Integer, primary_key=True)
    prompt_id = Column(Integer, ForeignKey('prompts.id'))
    negative_prompt_id = Column(Integer)
    sample = Column(String)
    seed = Column(Integer)
    width = Column(Integer)
    height = Column(Integer)
    steps = Column(Integer)
    scale = Column(String)
    model = Column(String)
    scheduler = Column(String)
    created_at = Column(DateTime)


class Prompt(Base):
    __tablename__ = 'prompts'
    id = Column(Integer, primary_key=True)
    prompt = Column(String)
    negative_prompt = Column(String)
    created_at = Column(DateTime)


class DBConnection:
    def __init__(self, db_name, key):
        self.engine = create_engine(f'sqlite:///{db_name}')
        self.key = key
        self.Session = sessionmaker(bind=self.engine)
        self.session = self.Session()
        Base.metadata.create_all(self.engine, checkfirst=True)

    def encrypt(self, data):
        fernet = Fernet(self.key)
        return fernet.encrypt(data.encode()).decode()

    def decrypt(self, data):
        fernet = Fernet(self.key)
        return fernet.decrypt(data.encode()).decode()

    def insert_txt2img_sample(
        self,
        prompt_id,
        negative_prompt_id,
        sample,
        seed,
        width,
        height,
        steps,
        scale,
        model,
        scheduler
    ):
        sample = self.encrypt(base64.b64encode(sample).decode())
        txt2img_sample = Txt2ImgSample(
            prompt_id=prompt_id,
            negative_prompt_id=negative_prompt_id,
            sample=sample,
            seed=seed,
            width=width,
            height=height,
            steps=steps,
            scale=scale,
            model=model,
            scheduler=scheduler
        )
        self.session.add(txt2img_sample)
        self.session.commit()

    def insert_img2img_sample(
        self,
        prompt_id,
        negative_prompt_id,
        sample,
        seed,
        width,
        height,
        steps,
        scale,
        model,
        scheduler
    ):
        sample = self.encrypt(base64.b64encode(sample).decode())
        img2img_sample = Img2ImgSample(
            prompt_id=prompt_id,
            negative_prompt_id=negative_prompt_id,
            sample=sample,
            seed=seed,
            width=width,
            height=height,
            steps=steps,
            scale=scale,
            model=model,
            scheduler=scheduler
        )
        self.session.add(img2img_sample)
        self.session.commit()

    def insert_prompt(self, prompt, negative_prompt):
        prompt = Prompt(prompt=prompt, negative_prompt=negative_prompt)
        self.session.add(prompt)
        self.session.commit()
