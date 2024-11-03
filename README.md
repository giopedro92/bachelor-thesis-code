# Instructions to run the code

---
- [Instructions to run the code](#instructions-to-run-the-code)
    - [Analysis](#analysis)
        - [Docker Analysis](#docker-analysis)
        - [Run Analysis](#run-analysis)
    - [EDO Neural network](#edo-neural-network)
        - [Docker EDO](#docker-edo)
        - [Run Neural Network EDO](#neural-network-edo)
    - [Neural network](#neural-network)
        - [Training](#training)
            - [Docker Neural Network](#docker-neural-network)
            - [Train Neural Network](#train-neural-network)
        - [Real data](#real-data)
    - [Appendice](#appendice)
        - [Root Trees](#root-trees)
---

>Author: Giovanni Pedrelli


## Analysis
### Docker Analysis
Inside the directory where the `.dockerfile` is, build the `dockerfile` to get a docker image

```bash
sudo docker build -f analysis.dockerfile -t analysis .
```

See the images installed
```bash
sudo docker images
```

<!--
Rename an image
```bash
sudo docker tag <tag> <name>
```
-->

<!--
Remove a docker image
```bash
sudo docker rmi <name>:<tag>
```
-->

Run the docker image:
- as `root`
    ```bash
    sudo docker run \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /home/giovanni-pedrelli/Scrivania/TESI/Analysis:/opt \
    --rm \
    -it \
    analysis bash
    ```

- with `--user $(id -u)` **ALLOWS YOU TO USE TBrowser**
    ```bash
    sudo docker run \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /home/giovanni-pedrelli/Scrivania/TESI/Analysis:/opt \
    --rm \
    -it \
    --user $(id -u) \
    analysis bash
    ```


### Run Analysis
**Inside the docker container**

Move to the right folder
```bash
cd /home/giovanni-pedrelli/TESI/Analysis
```

Run the script
```bash
python3 analysis.py
```






## EDO Neural network
### Docker EDO

Run the docker image:
```bash
sudo docker run \
-e DISPLAY=$DISPLAY \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v /home/giovanni-pedrelli/Scrivania/SC-EXAM/:/app \
--rm \
-it \
neural-network bash
```

### Neural Network EDO
**Inside the docker container**

```bash
cd Python_Cat
```

```bash
python3 main.py Neural_Network
```





## Neural network
### Training
#### Docker Neural Network

Inside the directory where the `.dockerfile` is, build the `.dockerfile` to get a docker image

```bash
sudo docker build -f neural-network.dockerfile -t neural-network .
```

See the images installed
```bash
sudo docker images
```

Run the docker image:
```bash
sudo docker run \
-e DISPLAY=$DISPLAY \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v /home/giovanni-pedrelli/Scrivania/TESI/NeuralNetwork:/app \
--rm \
-it \
neural-network bash
```

#### Train Neural Network
**Inside the docker container**

Run the code to train the Neural Netrwork
```bash
python3 main.py Neural_Network
```


### Real data

...



## Appendice
### ROOT Trees
Inspect inside a `.root` file. **Inside ROOT**

```bash
std::unique_ptr<TFile> myFile( TFile::Open("*_*filename*_*.root") );
```

```bash
myFile->ls()
```