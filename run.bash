#!/bin/bash

docker run --rm -it --gpus '"device=0"' \
    --name diffuse2lose-docker \
    --network=host \
    -p 8116:8125 \
    -e "TERM=xterm-256color" \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -e XAUTHORITY=$XAUTH \
    -v "./data:/home/jason/data" \
    -v "./loaders:/home/jason/loaders" \
    -v "./utils:/home/jason/utils" \
    -v "./logs:/home/jason/logs" \
    -v "./tune.py:/home/jason/tune.py" \
    diffuse2lose:dev \
    bash
