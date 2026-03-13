#!/bin/bash
# Script to easily start the CARLA simulator server in Docker
# Supports both headless (default) and GUI modes

CARLA_VERSION="0.9.14"
CONTAINER_NAME="carla_server"
GUI_MODE=0

# Clean up previously running or stopped container with the same name
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "Stopping and removing existing $CONTAINER_NAME container..."
    docker stop $CONTAINER_NAME > /dev/null 2>&1
    docker rm $CONTAINER_NAME > /dev/null 2>&1
fi

# Parse arguments
for arg in "$@"; do
    if [ "$arg" == "--gui" ] || [ "$arg" == "--view" ]; then
        GUI_MODE=1
    fi
done

if [ "$GUI_MODE" -eq 1 ]; then
    echo "Starting CARLA $CARLA_VERSION with GUI enabled..."
    
    # Allow local X server connections for this session
    xhost +local: > /dev/null 2>&1
    
    # Run Docker with X11 forwarding and without the headless flag
    docker run -d --name $CONTAINER_NAME \
        --privileged \
        --gpus all \
        --net=host \
        -e DISPLAY=$DISPLAY \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        carlasim/carla:$CARLA_VERSION \
        /bin/bash ./CarlaUE4.sh
    
    echo "CARLA is starting in the background. A window should appear shortly."
    echo "To view logs, run: docker logs -f $CONTAINER_NAME"
    echo "To stop, run: docker stop $CONTAINER_NAME"
else
    echo "Starting CARLA $CARLA_VERSION in headless mode (default)..."
    
    # Run Docker in standard headless mode
    docker run -d --name $CONTAINER_NAME \
        --privileged \
        --gpus all \
        --net=host \
        carlasim/carla:$CARLA_VERSION \
        /bin/bash ./CarlaUE4.sh -RenderOffScreen
        
    echo "CARLA is running in the background."
    echo "To view logs, run: docker logs -f $CONTAINER_NAME"
    echo "To stop, run: docker stop $CONTAINER_NAME"
fi
