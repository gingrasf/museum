FROM jupyter/datascience-notebook

COPY museum.ipynb /home/jovyan/museum.ipynb
COPY museum.py /home/jovyan/museum.py
COPY requirements.txt /home/jovyan/requirements.txt

RUN pip install -r requirements.txt