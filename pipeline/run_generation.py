#!/usr/bin/env python3
import os
import yaml
import subprocess
import argparse
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_command(cmd, cwd=None):
    """Executes a shell command and logs output."""
    logger.info(f"Executing: {' '.join(cmd)}")
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            cwd=cwd
        )
        for line in process.stdout:
            logger.info(f"  [OP]: {line.strip()}")
        process.wait()
        if process.returncode != 0:
            logger.error(f"Command failed with exit code {process.returncode}")
        return process.returncode
    except Exception as e:
        logger.error(f"Error executing command: {e}")
        return -1

def main():
    parser = argparse.ArgumentParser(description="SkyScenes Data Generation Pipeline")
    parser.add_argument("--config", default="config/default_generation.yaml", help="Path to config file")
    parser.add_argument("--root_dir", help="Override ROOT_DIR from config")
    parser.add_argument("--save_seg", action='store_true', default=None, help="Override save_seg from config")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    root_dir = args.root_dir or config.get("ROOT_DIR", "/home/df/data/datasets")
    save_seg = args.save_seg if args.save_seg is not None else config.get("save_seg", False)
    baseline_conf = config.get("baseline", {})
    variation_conf = config.get("variations", {})
    exec_conf = config.get("execution", {})

    logger.info("Starting SkyScenes Data Generation Pipeline")
    logger.info(f"Root Directory: {root_dir}")

    # Baseline Generation Loop
    for town in baseline_conf.get("towns", []):
        for height in baseline_conf.get("heights", []):
            for pitch in baseline_conf.get("pitches", []):
                weather = baseline_conf.get("weather", "ClearNoon")
                num = baseline_conf.get("num_images", 10)

                # Check if baseline exists
                baseline_meta_dir = os.path.join(root_dir, f"H_{height}_P_{abs(pitch)}", weather, town, "metaData")
                if os.path.exists(baseline_meta_dir) and len(os.listdir(baseline_meta_dir)) >= num:
                    logger.info(f"Baseline for {town} H={height} P={pitch} already exists. Skipping.")
                else:
                    logger.info(f"Generating Baseline: {town}, H={height}, P={pitch}")
                    cmd = [
                        "uv", "run", "python", "manualSpawning.py",
                        "--town", town,
                        "--weather", weather,
                        "--height", str(height),
                        "--pitch", str(pitch),
                        "--num", str(num),
                        "--ROOT_DIR", root_dir
                    ]
                    if save_seg:
                        cmd.append("--save_seg")
                    # Note: Using sudo if required by the system environment, but assuming user-level uv setup
                    ret = run_command(cmd)
                    if ret != 0 and exec_conf.get("stop_on_error", False):
                        logger.error("Stop on error enabled. Aborting.")
                        return

                # Variation Generation Loop
                for v_weather in variation_conf.get("weather", []):
                    # Height & Pitch variations can be handled similarly if needed
                    logger.info(f"Generating Variation: {v_weather} for baseline {town} H={height} P={pitch}")
                    
                    var_meta_dir = os.path.join(root_dir, f"H_{height}_P_{abs(pitch)}", v_weather, town, "metaData")
                    # loadingAttributesWeather.py uses the baseline metaDataDir as input
                    if os.path.exists(var_meta_dir) and len(os.listdir(var_meta_dir)) >= num:
                         logger.info(f"Variation {v_weather} already exists. Skipping.")
                         continue

                    cmd_var = [
                        "uv", "run", "python", "loadingAttributesWeather.py",
                        "--town", town,
                        "--weather", v_weather,
                        "--height", str(height),
                        "--pitch", str(pitch),
                        "--metaDataDir", baseline_meta_dir,
                        "--ROOT_DIR", root_dir
                    ]
                    if save_seg:
                        cmd_var.append("--save_seg")
                    run_command(cmd_var)

                # Cross-variations for height/pitch
                for v_height in variation_conf.get("heights", []):
                    for v_pitch in variation_conf.get("pitches", []):
                        if v_height == height and v_pitch == pitch: continue
                        
                        logger.info(f"Generating H/P Variation: H={v_height} P={v_pitch} for baseline {town}")
                        var_meta_hp_dir = os.path.join(root_dir, f"H_{v_height}_P_{abs(v_pitch)}", weather, town, "metaData")
                        if os.path.exists(var_meta_hp_dir) and len(os.listdir(var_meta_hp_dir)) >= num:
                             logger.info(f"H/P Variation H={v_height} P={v_pitch} already exists. Skipping.")
                             continue

                        cmd_hp = [
                            "uv", "run", "python", "loadingAttributesWeather.py",
                            "--town", town,
                            "--weather", weather, # Reuse baseline weather but change params
                            "--height", str(v_height),
                            "--pitch", str(v_pitch),
                            "--metaDataDir", baseline_meta_dir,
                            "--ROOT_DIR", root_dir
                        ]
                        if save_seg:
                            cmd_hp.append("--save_seg")
                        run_command(cmd_hp)

    logger.info("Pipeline Execution Complete")

if __name__ == "__main__":
    main()
