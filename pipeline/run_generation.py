#!/usr/bin/env python3
import os
import yaml
import subprocess
import argparse
import logging
import shutil
import time
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
        
        # Give CARLA simulator time to clean up resources between runs
        logger.info("Waiting 5 seconds for simulator to stabilize...")
        time.sleep(5)
        return process.returncode
    except Exception as e:
        logger.error(f"Error executing command: {e}")
        return -1

def run_pipeline(cmd, dry_run=False, cwd=None):
    """Orchestrates command execution or dry run."""
    if dry_run:
        logger.info(f"DRY RUN: {' '.join(cmd)}")
        return 0
    return run_command(cmd, cwd=cwd)

def main():
    parser = argparse.ArgumentParser(description="SkyScenes Data Generation Pipeline")
    parser.add_argument("--config", default="config/default_generation.yaml", help="Path to config file")
    parser.add_argument("--root_dir", help="Override ROOT_DIR from config")
    parser.add_argument("--save_seg", action='store_true', default=None, help="Override save_seg from config")
    parser.add_argument("--extract_gbuffer", action='store_true', default=None, help="Override extract_gbuffer from config")
    parser.add_argument("--generate_lidar", action='store_true', default=None, help="Override generate_lidar from config")
    parser.add_argument("--force", action='store_true', help="Force regeneration by cleaning up existing data")
    parser.add_argument("--python", default="3.8", help="Python version to use for uv run")
    parser.add_argument("--dry-run", action='store_true', help="Log commands without executing them")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    root_dir = args.root_dir or config.get("ROOT_DIR", "/home/df/data/datasets")
    save_seg = args.save_seg if args.save_seg is not None else config.get("save_seg", False)
    extract_gbuffer = args.extract_gbuffer if args.extract_gbuffer is not None else config.get("extract_gbuffer", False)
    generate_lidar = args.generate_lidar if args.generate_lidar is not None else config.get("generate_lidar", False)
    python_ver = args.python
    baseline_conf = config.get("baseline", {})
    variation_conf = config.get("variations", {})
    exec_conf = config.get("execution", {})

    logger.info("Starting SkyScenes Data Generation Pipeline")
    logger.info(f"Root Directory: {root_dir}")

    tm_port_base = 8000
    tm_port_offset = 0

    def get_next_tm_port():
        nonlocal tm_port_offset
        port = tm_port_base + tm_port_offset
        if port == 8000: # Skip 8000 as it might be busy
            tm_port_offset += 1
            port = tm_port_base + tm_port_offset
        tm_port_offset = (tm_port_offset + 1) % 100
        return port

    # Handle baseline weather as list or string
    baseline_weathers = baseline_conf.get("weather", "ClearNoon")
    if isinstance(baseline_weathers, str):
        baseline_weathers = [baseline_weathers]
    
    num = baseline_conf.get("num_images", 10)

    # Baseline Generation Loop
    for weather in baseline_weathers:
        for town in baseline_conf.get("towns", []):
            for height in baseline_conf.get("heights", []):
                for pitch in baseline_conf.get("pitches", []):
                    # Check if baseline exists
                    baseline_dir = os.path.join(root_dir, f"H_{height}_P_{abs(pitch)}", weather, town)
                    baseline_meta_dir = os.path.join(baseline_dir, "metaData")
                    
                    if args.force and os.path.exists(baseline_dir):
                        logger.info(f"Force enabled. Cleaning up baseline directory: {baseline_dir}")
                        if not args.dry_run:
                            shutil.rmtree(baseline_dir)
                        else:
                            logger.info(f"DRY RUN: Would delete {baseline_dir}")

                    if not args.dry_run and os.path.exists(baseline_meta_dir) and len(os.listdir(baseline_meta_dir)) >= num:
                        logger.info(f"Baseline for {town} H={height} P={pitch} W={weather} already exists. Skipping.")
                    else:
                        logger.info(f"Generating Baseline: {town}, H={height}, P={pitch}, W={weather}")
                        cmd = [
                            "uv", "run", "--python", python_ver, "python", "manualSpawning.py",
                            "--town", town,
                            "--weather", weather,
                            "--height", str(height),
                            "--pitch", str(pitch),
                            "--num", str(num),
                            "--ROOT_DIR", root_dir,
                            "--tm_port", str(get_next_tm_port())
                        ]
                        if save_seg:
                            cmd.append("--save_seg")
                        if extract_gbuffer:
                            cmd.append("--extract_gbuffer")
                        if generate_lidar:
                            cmd.append("--generate_lidar")
                        
                        ret = run_pipeline(cmd, dry_run=args.dry_run)
                        if ret != 0 and exec_conf.get("stop_on_error", False):
                            logger.error("Stop on error enabled. Aborting.")
                            return
                    
                    # Safety check for variations: skip if baseline metadata missing (and not dry run)
                    if not args.dry_run and (not os.path.exists(baseline_meta_dir) or len(os.listdir(baseline_meta_dir)) == 0):
                        logger.error(f"Baseline metadata not found at {baseline_meta_dir}. Skipping variations.")
                        continue

                    # Variation Generation Loop (Weather Variations)
                    for v_weather in variation_conf.get("weather", []):
                        logger.info(f"Generating Variation: {v_weather} for baseline {town} H={height} P={pitch} W={weather}")
                        
                        var_dir = os.path.join(root_dir, f"H_{height}_P_{abs(pitch)}", v_weather, town)
                        var_meta_dir = os.path.join(var_dir, "metaData")
                        
                        if args.force and os.path.exists(var_dir):
                            logger.info(f"Force enabled. Cleaning up variation directory: {var_dir}")
                            if not args.dry_run:
                                shutil.rmtree(var_dir)
                            else:
                                logger.info(f"DRY RUN: Would delete {var_dir}")

                        if not args.dry_run and os.path.exists(var_meta_dir) and len(os.listdir(var_meta_dir)) >= num:
                             logger.info(f"Variation {v_weather} already exists. Skipping.")
                             continue

                        cmd_var = [
                            "uv", "run", "--python", python_ver, "python", "loadingAttributesWeather.py",
                            "--town", town,
                            "--weather", v_weather,
                            "--height", str(height),
                            "--pitch", str(pitch),
                            "--metaDataDir", baseline_meta_dir,
                            "--ROOT_DIR", root_dir,
                            "--tm_port", str(get_next_tm_port())
                        ]
                        if save_seg:
                            cmd_var.append("--save_seg")
                        if extract_gbuffer:
                            cmd_var.append("--extract_gbuffer")
                        if generate_lidar:
                            cmd_var.append("--generate_lidar")
                        run_pipeline(cmd_var, dry_run=args.dry_run)

                    # Cross-variations for height/pitch
                    for v_height in variation_conf.get("heights", []):
                        for v_pitch in variation_conf.get("pitches", []):
                            if v_height == height and v_pitch == pitch: continue
                            
                            logger.info(f"Generating H/P Variation: H={v_height} P={v_pitch} for baseline {town} W={weather}")
                            var_hp_dir = os.path.join(root_dir, f"H_{v_height}_P_{abs(v_pitch)}", weather, town)
                            var_meta_hp_dir = os.path.join(var_hp_dir, "metaData")

                            if args.force and os.path.exists(var_hp_dir):
                                logger.info(f"Force enabled. Cleaning up H/P Variation directory: {var_hp_dir}")
                                if not args.dry_run:
                                    shutil.rmtree(var_hp_dir)
                                else:
                                    logger.info(f"DRY RUN: Would delete {var_hp_dir}")

                            if not args.dry_run and os.path.exists(var_meta_hp_dir) and len(os.listdir(var_meta_hp_dir)) >= num:
                                 logger.info(f"H/P Variation H={v_height} P={v_pitch} already exists. Skipping.")
                                 continue

                            cmd_hp = [
                                "uv", "run", "--python", python_ver, "python", "loadingAttributesWeather.py",
                                "--town", town,
                                "--weather", weather, # Reuse baseline weather
                                "--height", str(v_height),
                                "--pitch", str(v_pitch),
                                "--metaDataDir", baseline_meta_dir,
                                "--ROOT_DIR", root_dir,
                                "--tm_port", str(get_next_tm_port())
                            ]
                            if save_seg:
                                cmd_hp.append("--save_seg")
                            if extract_gbuffer:
                                cmd_hp.append("--extract_gbuffer")
                            if generate_lidar:
                                cmd_hp.append("--generate_lidar")
                            run_pipeline(cmd_hp, dry_run=args.dry_run)

    if generate_lidar:
        logger.info("\n" + "="*50)
        logger.info("Starting automatic town merging...")
        merge_cmd = [
            "uv", "run", "--python", python_ver, "python", "scripts/build_town_map.py",
            "--input_dir", root_dir,
            "--output_dir", os.path.join(root_dir, "merged_towns")
        ]
        ret = run_pipeline(merge_cmd, dry_run=args.dry_run)
        if ret == 0:
            logger.info(f"Automatic merging complete. Maps saved to: {os.path.join(root_dir, 'merged_towns')}")
        else:
            logger.error("Automatic merging failed.")
        logger.info("="*50 + "\n")

    logger.info("Pipeline Execution Complete")

if __name__ == "__main__":
    main()
