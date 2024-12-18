#!/bin/bash

docker run --rm -it --gpus '"device=0"' \
    --name diffuse2lose-docker-2 \
    --network=host \
    -p 8116:8125 \
    -e "TERM=xterm-256color" \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -e XAUTHORITY=$XAUTH \
    -v "./data:/home/`whoami`/data" \
    -v "./loaders:/home/`whoami`/loaders" \
    -v "./utils:/home/`whoami`/utils" \
    -v "./logs:/home/`whoami`/logs" \
    -v "./train:/home/`whoami`/train" \
    -v "./models:/home/`whoami`/models" \
    -v "./eval:/home/`whoami`/eval" \
    diffuse2lose:dev \
    bash
