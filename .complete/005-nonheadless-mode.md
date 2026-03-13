# Feature: Non-Headless Mode

## Description
Launch the CARLA server in non-headless mode (without the `-RenderOffScreen` flag) to allow the spectator camera and other graphical utilities to function properly. Currently, CARLA is launched exclusively in headless mode, which prevents visual testing with the spectator camera UI.

## Requirements
To run CARLA in a Docker container with an active window display (non-headless), the following adjustments to the launch process are required:

1. **Pass Display Variable**: Provide the host's display to the container using `-e DISPLAY=$DISPLAY` or specific X11 forwarding in the docker run command.
2. **X11 Socket Mount**: Ensure the X11 socket is volume-mounted `-v /tmp/.X11-unix:/tmp/.X11-unix:rw` (already present in the current `init_carla_docker.sh` script).
3. **Xhost Permissions**: The host must allow local connections to the X server, typically done by running `xhost +local:` on the host machine before starting the Docker container.
4. **Omit Headless Flag**: Remove the `-RenderOffScreen` parameter from the `./CarlaUE4.sh` execution.

## Proposed Strategy
1. Create a dedicated `scripts/start_carla.sh` utility script, or modify the existing initialization process to be a general-purpose launcher.
2. Implement a `--gui` or `--view` flag in the script that applies the X11 forwarding settings, prompts the user about `xhost`, and removes `-RenderOffScreen`.
3. Preserve the default background behavior as headless to ensure automated data generation pipelines continue to run uninterrupted without needing a display server.