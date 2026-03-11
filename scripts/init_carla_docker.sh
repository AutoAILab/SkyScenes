#!/bin/bash
# Script to initialize CARLA Docker and extract PythonAPI

CARLA_VERSION="0.9.14"
CONTAINER_NAME="carla_init_$(date +%s)"

echo "Step 1: Pulling CARLA $CARLA_VERSION Docker image..."
docker pull carlasim/carla:$CARLA_VERSION

echo "Step 2: Starting CARLA container to extract PythonAPI..."
# Run in detached mode to allow copying
docker run -d --name $CONTAINER_NAME --privileged --gpus all --net=host -v /tmp/.X11-unix:/tmp/.X11-unix:rw carlasim/carla:$CARLA_VERSION /bin/bash ./CarlaUE4.sh -RenderOffScreen

# Wait for container to be ready
echo "Waiting for container to initialize..."
sleep 10

echo "Step 3: Extracting PythonAPI from container to current directory..."
docker cp $CONTAINER_NAME:/home/carla/PythonAPI ./

echo "Step 4: Stopping and removing temporary container..."
docker stop $CONTAINER_NAME
docker rm $CONTAINER_NAME

echo "Setup Complete! You should now have a 'PythonAPI' directory in $(pwd)."
echo "Note: Ensure you have NVIDIA Docker support installed if using GPUs."
