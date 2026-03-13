#!/usr/bin/env python3
import argparse
import time

try:
    import sys
    sys.path.append('PythonAPI/carla/dist/carla-0.9.14-py3.7-linux-x86_64.egg')
    import carla
except ImportError:
    print("ERROR: carla Python API is not installed or available.")
    print("Please ensure your Python path includes the CARLA egg file or the module is installed via pip.")
    exit(1)

def main():
    parser = argparse.ArgumentParser(description="Connects to a CARLA server to view the map as a spectator.")
    parser.add_argument("--host", default="127.0.0.1", help="IP of the CARLA server (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=2000, help="Terminal port of the CARLA server (default: 2000)")
    parser.add_argument("--town", type=str, default=None, help="Specific town to load (e.g., Town10HD). If not specified, uses the currently active town.")
    args = parser.parse_args()

    # Connect to the CARLA server
    print(f"Connecting to CARLA Server at {args.host}:{args.port}...")
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
    except Exception as e:
        print(f"Failed to connect to CARLA server: {e}")
        return

    # Check for requested town
    if args.town:
        current_map = client.get_world().get_map().name
        if not current_map.endswith(args.town):
            print(f"Loading '{args.town}' (Current map: {current_map})")
            client.load_world(args.town)
            print("Map loaded successfully.")
        else:
            print(f"The requested town '{args.town}' is already loaded.")

    world = client.get_world()

    # Check rendering mode
    settings = world.get_settings()
    if settings.no_rendering_mode:
        print("\nWARNING: The CARLA server is currently running in 'no_rendering_mode' (-RenderOffScreen).")
        print("You will not be able to see the 3D spectator window.")
        print("Please restart the simulator without the '-RenderOffScreen' flag to use the spectator camera.")
    else:
        print("\n=== Spectator Camera Mode ===")
        print("A CARLA display window should be visible.")
        print("Controls in the CARLA window:")
        print("  W, A, S, D : Move Forward, Left, Backward, Right")
        print("  Q, E       : Move Up, Down")
        print("  Mouse      : Look Around")
        print("=============================\n")

    # Retrieve the spectator camera actor natively provided by CARLA
    spectator = world.get_spectator()
    print(f"Spectator camera attached at: {spectator.get_transform().location}")
    print("Press Ctrl+C to disconnect and exit safely.")

    # Loop to keep the script running and active
    try:
        while True:
            # We can optionally print the spectator transform or just wait
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nDisconnecting from the CARLA server...")
    
if __name__ == "__main__":
    main()
