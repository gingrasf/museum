# To use in Collab

* Go [here](https://colab.research.google.com/drive/1DLjOVIUt7f2tcP7GP70BoI7HrvmitjKQ)
* Go to the left > menu and import museum.py from this repo (if it's not there already)
* Execute -> Execute all

Note that the first time it can be slow due to downloading the data. After it's download, it'll be persisted in a csv file for futre uses.

# To use using Docker

## Prerequisite

* Docker
* Docker Compose

## To launch
```
docker-compose up
```

##

To use on windows using docker toolbox, you need to use the ip of the docker machine. To get it you can do on the Docker Quickstart Terminal:
```
docker-machine ip
```

## To connect to Mongo Admin
```
http://(<docker-machine-ip> or <localhost>):8081/
```

## To connect to the Jupyter notebook

Once docker-compose is done, you should see a similar output

```
[I 16:13:51.921 NotebookApp] http://(934bad6a1e5e or 127.0.0.1):8888/?token=8a7a73260c2f985cf22e416000e7f6a03587c0710f9e608c
```

That will give you the token needed to connect to the Jupyter instance, you then need to go to:

```
http://(<docker-machine-ip> or <localhost>):8888/?token=<token-value>
```

Once there simply click on the following notebook to open it and then execute everything.

```
museum.ipynb
```

Note that the first run will be slow since it needs to download the data to the mongo instance.


