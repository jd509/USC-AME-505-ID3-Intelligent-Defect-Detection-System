echo "Building the image............"

docker run --name="defect_detector" \
    --rm \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume=$HOME/.bash_history:/home/ros/.bash_history \
    --env="XAUTHORITY=$XAUTH" \
    --expose 9090 \
    --net host \
    --privileged \
    --workdir="/app/defect_detector/scripts/" \
    "usc_id3_defect_detector:v1.0" \
    python3 user_interface.py
    /bin/bash

exit 0