# Feature: Spectator Camera Utility

## Description
Create a standalone utility script to easily attach a spectator camera to a specific town or running CARLA world. This will allow users to quickly visualize and explore the 3D environment, which is especially useful when running the pre-compiled or Docker version of CARLA without the Unreal Editor.

## Requirements
- [ ] Connect to the running CARLA server on a specified port (default `2000`).
- [ ] Allow the user to specify a new map/town to load (e.g., `Town10HD`), or attach to the currently active map.
- [ ] Provide clear terminal instructions on how to use the keyboard/mouse controls (W, A, S, D, Q, E, Mouse) to navigate the spectator view.
- [ ] Ensure the simulator is not running in pure headless mode (`-RenderOffScreen`), or warn the user if no display window can be created.
- [ ] Implement a clean exit strategy without crashing the underlying simulation server.

## References
- **Documentation:** `docs/carla_simulator.md` - Section: "Examining Towns in a 3D Environment"
- **CARLA API:** The script will need to utilize `carla.Client('localhost', 2000)` and `world.get_spectator()`.
- **Town Selection:** Changing maps natively uses `client.load_world('<TownName>')`.

## Use Cases
When debugging world generation, traffic behaviors, or sensor placement, a user can run this script to fly a free-cam around the map to visually confirm simulator conditions prior to running the massive data generation pipeline (`pipeline/run_generation.py`).
