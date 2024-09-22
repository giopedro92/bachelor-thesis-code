# Instructions to run the code

---
- [Instructions to run the code](#instructions-to-run-the-code)
    - [Analysis](#analysis)
    - [Neural network](#neural-network)

---

>Author: Giovanni Pedrelli

## Analysis
### Docker
Build the `.dockerfile` to get a docker image

```bash
sudo docker build -f analysis.dockerfile .
```

See the images installed
```bash
sudo docker images
```

Rename an image
```bash
sudo docker tag <tag> <name>
```

Run the docker image:
- as `root`
    ```bash
    sudo docker run \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /home:/home \
    --rm \
    -it \
    root-uproot bash
    ```

- with `--user $(id -u)`
    ```bash
    sudo docker run \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /home:/home \
    --rm \
    -it \
    --user $(id -u) \
    root-uproot bash
    ```


### Analysis
Inside the docker container

Move to the right folder
```bash
cd /home/giovanni-pedrelli/TESI/Analysis
```

Run the script
```bash
python3 analysis.py
```


## Neural network
Run the code to train the neural netrwork