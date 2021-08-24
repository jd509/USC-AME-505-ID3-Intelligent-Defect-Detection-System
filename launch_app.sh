#!/bin/sh

echo "Loading the container and running the app............"

sudo docker run -it \
    --name="defect_detector" \
    --rm \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --env="XAUTHORITY=$XAUTH" \
    --runtime=nvidia \
    --expose 9090 \
    --net host \
    --privileged \
    --workdir="/app/defect_detector/scripts/" \
    "usc_id3_defect_detector:v1.0" \
        python3 user_interface.py
    /bin/bash
exit 0