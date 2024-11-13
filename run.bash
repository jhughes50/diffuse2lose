#!/bin/bash

docker run --rm -it --gpus '"device=0"' \
    --name cis680-docker \
    --network=host \
    -p 8116:8125 \
    -e "TERM=xterm-256color" \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -e XAUTHORITY=$XAUTH \
    -v "./data:/home/jason/data" \
    diffuse2lose:dev \
    bash
