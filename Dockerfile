FROM jupyter/datascience-notebook

COPY museum.ipynb /home/jovyan/museum.ipynb
COPY museum_v2.py /home/jovyan/museum_v2.py
COPY requirements.txt /home/jovyan/requirements.txt

RUN pip install -r requirements.txt