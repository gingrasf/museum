# Prerequisite

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


