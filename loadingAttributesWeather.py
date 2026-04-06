import sys
sys.path.append('PythonAPI/carla/dist/carla-0.9.14-py3.7-linux-x86_64.egg')
import argparse
import carla
import cv2
import json
import numpy as np
import os
import queue
import random
import time 
from PIL import Image
import open3d as o3d
import logging

# Add src to path for relative imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from skyscenes.generator import BaseGenerator

logger = logging.getLogger(__name__)

np.random.seed(0)

'''Color Dictionary for carla Simulartor 0.9.14, can be used for RGB to int mapping'''
carla_colordict_14 = {
                0: (0, 0, 0),                       # Unlabeled
                1: (70, 70, 70),                    # Building  
                2: (100, 40, 40),                   # Fence
                3: (55, 90, 80),                    # Other
                4: (220, 20, 60),                   # Pedestrian
                5: (153, 153, 153),                 # Pole
                6: (157, 234, 50),                  # RoadLine
                7: (128, 64, 128),                  # Road
                8: (244, 35, 232),                  # SideWalk
                9: (107, 142, 35),                  # Vegetation
                10: (0, 0, 142),                    # Vehicles -- Cars, vans, trucks, motorcycles, bikes, buses, trains. (old car class)
                11: (102, 102, 156),                # Wall
                12: (220, 220, 0),                  # TrafficSign
                13: (70, 130, 180),                 # Sky
                14: (81, 0, 81),                    # Ground
                15: (150, 100, 100),                # Bridge
                16: (230, 150, 140),                # RailTrack            
                17: (180, 165, 180),                # GuardRail
                18: (250, 170, 30),                 # TrafficLight
                19: (110, 190, 160),                # Static  [new class!]
                20: (170, 120, 50),                 # Dynamic [new class!]
                21: (45, 60, 150),                  # Water
                22: (145, 170, 100),                # Terrain (old low vegetation class)
                } 


class GenImage(BaseGenerator):
    def __init__(self, args, metaDataDir, index):
        super().__init__(args)
        self.height_arg = args.height
        self.pitch_arg = args.pitch
        
        logger.info(f"Saving DIR: {os.path.join(self.ROOT_DIR, f'H_{self.height}_P_{abs(self.pitch)}/{self.weather_str}/{self.town}')}")
        logger.info("__"*20)
        
        # Paths
        self.save_dir = os.path.join(self.ROOT_DIR, f"H_{self.height}_P_{abs(self.pitch)}/{self.weather_str}/{self.town}")
        os.makedirs(os.path.join(self.save_dir, "Images"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "CarlaSegment"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "Depth"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "Instance"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "metaData"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "Lidar"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "GBuffer"), exist_ok=True)
        
        self.extract_gbuffer = args.extract_gbuffer
        self.generate_lidar = args.generate_lidar
        self.save_metadata = True
        self.noon_json = args.noon_json
        
        # Meta data handling
        self.metaDataDir = metaDataDir
        self.index = index
        self.metaDataFiles = os.listdir(self.metaDataDir)
        self.metaDataFiles.sort()
        self.totalImages = len(self.metaDataFiles)

        self.IMG_WIDTH = 2160
        self.IMG_HEIGHT = 1440
        self.SIGMA_H = 2.5
        self.SIGMA_P = 5
        self.FOV = 110      # Does not change
        self.SENSOR_X = 5   # Location of sensor
        self.SENSOR_Z = 2.5 # Location of sensor
        self.carlaClassNum = 8
        
        ### External arguments
        self.town = args.town
        self.height = args.height
        self.heightCamera = self.height
        self.pitch = args.pitch
        self.pitchCamera = self.pitch
        self.metaDataDir = metaDataDir
        self.index = int(args.index) 
        self.weather_str = args.weather
        if args.weather == "ClearNoon":
            self.weather = carla.WeatherParameters.ClearNoon
        elif args.weather == "CloudyNoon":
            self.weather = carla.WeatherParameters.CloudyNoon
        elif args.weather == "MidRainyNoon":
            self.weather = carla.WeatherParameters.MidRainyNoon
        elif args.weather == "ClearSunset":
            self.weather = carla.WeatherParameters.ClearSunset
        elif args.weather == "ClearNight":
            self.weather = carla.WeatherParameters(
                                                    cloudiness = 0.0,
                                                    precipitation = 0.0,
                                                    precipitation_deposits = 0.0,
                                                    wind_intensity = 0.0,
                                                    sun_azimuth_angle = -1.0,
                                                    sun_altitude_angle = -90.0, 
                                                    fog_density = 0.0,
                                                    fog_distance = 0.0, 
                                                    wetness = 0.0
                                                    )
        
        self.totalImages = len(os.listdir(os.path.join(self.metaDataDir))[self.index:])
        self.files = sorted(os.listdir(os.path.join(self.metaDataDir)))[self.index:]

        ### Creating Directories
        ## Save Data
        os.makedirs(os.path.join(self.ROOT_DIR, f"H_{self.height}_P_{abs(self.pitch)}/{self.weather_str}"), exist_ok=True)
        os.makedirs(os.path.join(self.ROOT_DIR, f"H_{self.height}_P_{abs(self.pitch)}/{self.weather_str}/{self.town}"), exist_ok=True)
        os.makedirs(os.path.join(self.ROOT_DIR, f"H_{self.height}_P_{abs(self.pitch)}/{self.weather_str}/{self.town}/Images"), exist_ok=True)
        if self.save_seg:
            os.makedirs(os.path.join(self.ROOT_DIR, f"H_{self.height}_P_{abs(self.pitch)}/{self.weather_str}/{self.town}/CarlaSegment"), exist_ok=True)
            os.makedirs(os.path.join(self.ROOT_DIR, f"H_{self.height}_P_{abs(self.pitch)}/{self.weather_str}/{self.town}/Depth"), exist_ok=True)
            os.makedirs(os.path.join(self.ROOT_DIR, f"H_{self.height}_P_{abs(self.pitch)}/{self.weather_str}/{self.town}/Instance"), exist_ok=True)
        if self.extract_gbuffer:
            os.makedirs(os.path.join(self.ROOT_DIR, f"H_{self.height}_P_{abs(self.pitch)}/{self.weather_str}/{self.town}/GBuffer"), exist_ok=True)
        if self.generate_lidar:
            os.makedirs(os.path.join(self.ROOT_DIR, f"H_{self.height}_P_{abs(self.pitch)}/{self.weather_str}/{self.town}/Lidar"), exist_ok=True)
        ## Save metaData for everything
        if self.save_metadata:
            os.makedirs(os.path.join(self.save_dir, "metaData"), exist_ok=True)
            
        self.vehiclesSpawned = 0
        self.walkerSpawned = 0

        # Finalize setup for variation generation
        self.world.set_weather(self.weather)

        self.actor_list = []
        self.spawned_vehicle = []
        self.spawned_people = []
        self.spawned_people_idnum = []
        self.vehicleDict = {}
        self.walkerDict = {}
        
        # Use base class TM and finalize state
        self.tm.set_random_device_seed(0)

        ### Ego vehicle to attach sensors
        self.blueprint_library = self.world.get_blueprint_library()
        bp = self.blueprint_library.filter('crossbike')[0] # crossbike because no shadow when it is floating in the air
        transform = random.choice(self.world.get_map().get_spawn_points()) 
        self.m = self.world.get_map()
        self.waypoint = self.m.get_waypoint(transform.location)
        self.vehicle = self.world.spawn_actor(bp, transform) 
        # Spawning relative to road
        vehicle_transform = carla.Transform(carla.Location(x=transform.location.x, y=transform.location.y, z=self.waypoint.transform.location.z + self.heightCamera),
                                             carla.Rotation(pitch=0, yaw=transform.rotation.yaw, roll=0)) 
        self.vehicle.set_transform(vehicle_transform)
        self.actor_list.append(self.vehicle)
        self.vehicle.set_autopilot(True, self.tm.get_port())
        self.vehicle.set_enable_gravity(False) # disables gravity
        self.m = self.world.get_map()
        self.waypoint = self.m.get_waypoint(transform.location)
        self.recursion_counter = 0
        self.addSensors()
        self.startTime = time.time()
        try:
            self.tickClock()
        except Exception as e:
            print(f"Exception during generation: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.destroyActors()
        
        if hasattr(self, 'endTime') and hasattr(self, 'startTime'):
            print(f'Total Time taken to generate images: {self.endTime - self.startTime}')
            print(f'Time per image: {(self.endTime - self.startTime) / self.totalImages}')
        
    def spawnVehicles(self, idNum, ID, position):
        '''
        Parameters
        ----------
        ID : ID of the vehicle to spawn
        idNum : key to save the vehicles position in self.vehicleDict    
        position : Transform with location and rotation, used to spwan the vehicle in the scene

        Returns
        -------
        None
            
        The function spawns a vehicle based on ID at the given position 
        '''

        transform = position
        transform = carla.Transform(carla.Location(x=transform.location.x, y=transform.location.y, z=transform.location.z + 0.05),
                                             carla.Rotation(pitch=transform.rotation.pitch, yaw=transform.rotation.yaw, roll=transform.rotation.roll))
        other_vehicle = None
        try:
            bp_vehicle = self.blueprint_library.filter('vehicle').find(ID)
            other_vehicle = self.world.try_spawn_actor(bp_vehicle, transform)
        except IndexError as e:
            # This is for 9.9 to 9.14
            new_ID = ID.split('.')[1].split('-')[0].split('_')[0] # Root of the old ID
            bp_vehicle = self.blueprint_library.filter(new_ID)[0]
            other_vehicle = self.world.try_spawn_actor(bp_vehicle, transform)
        
        if self.args.load_old is not None: 
            if other_vehicle is not None:
                pass # replaced autopilot call
                self.spawned_vehicle.append(other_vehicle)
                self.spawned_people_idnum.append(idNum)
                self.vehicleDict[idNum] = position
            else:
                self.recursion_counter += 1
                if self.recursion_counter > 10:
                    return
                self.spawnVehicles(idNum, ID, transform)
        else:
            if other_vehicle is not None:
                pass # replaced autopilot call
                self.spawned_vehicle.append(other_vehicle)
                self.spawned_people_idnum.append(idNum)
                self.vehicleDict[idNum] = position
    
    def addSensors(self):
        '''
        Function Adds necessary sensors to the scene: GRB camera, segmentation camera, ground RGB camera, Depth camera
        Field of view: 110
        Height of the aerial cameras uses the self.heightCamera parameter
        Pitch, roll, yaw: 0
        '''
        ########################################################################################################################
        ####### IMAGES
        ########################################################################################################################
        ##### AERIAL VIEW
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('fov', f'{str(self.FOV)}')
        camera_bp.set_attribute('image_size_x', f'{self.IMG_WIDTH}')
        camera_bp.set_attribute('image_size_y', f'{self.IMG_HEIGHT}')
        camera_bp.set_attribute('motion_blur_intensity', '0')
        camera_bp.set_attribute('motion_blur_max_distortion', '0')
        camera_bp.set_attribute('motion_blur_min_object_screen_size', '0')
        camera_bp.set_attribute('blur_amount', '0')
        camera_bp.set_attribute('enable_postprocess_effects', 'True')
        camera_transform = carla.Transform(carla.Location(x=self.SENSOR_X,), 
                                           carla.Rotation(pitch=self.pitchCamera, yaw=0, roll=0))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.image_queue = queue.Queue()
        self.camera.listen(self.image_queue.put)
        if self.extract_gbuffer:
            self.gbuffer_queues = {}
            for gb in ["SceneDepth", "SceneStencil", "GBufferA", "GBufferB", "GBufferC"]:
                self.gbuffer_queues[gb] = queue.Queue()
                self.camera.listen_to_gbuffer(getattr(carla.GBufferTextureID, gb), self.gbuffer_queues[gb].put)
        self.actor_list.append(self.camera)
        ########################################################################################################################
        ####### SEMANTIC SEGMENTATION
        ########################################################################################################################
        ##### AERIAL VIEW
        if self.save_seg:
            camera_semseg = self.blueprint_library.find('sensor.camera.semantic_segmentation')
            camera_semseg.set_attribute('fov', f'{str(self.FOV)}')
            camera_semseg.set_attribute('image_size_x', f'{self.IMG_WIDTH}')
            camera_semseg.set_attribute('image_size_y', f'{self.IMG_HEIGHT}')
            camera_transform = carla.Transform(carla.Location(x=self.SENSOR_X,), 
                                               carla.Rotation(pitch=self.pitchCamera, yaw=0, roll=0))
            self.camera_seg = self.world.spawn_actor(camera_semseg, camera_transform, attach_to=self.vehicle)
            self.image_queue_seg = queue.Queue()
            self.camera_seg.listen(self.image_queue_seg.put)
            self.actor_list.append(self.camera_seg)
            ########################################################################################################################
            ####### DEPTH
            ########################################################################################################################
            ##### AERIAL VIEW
            camera_depth = self.blueprint_library.find('sensor.camera.depth')
            camera_depth.set_attribute('fov', f'{str(self.FOV)}')
            camera_depth.set_attribute('image_size_x', f'{self.IMG_WIDTH}')
            camera_depth.set_attribute('image_size_y', f'{self.IMG_HEIGHT}')
            camera_transform = carla.Transform(carla.Location(x=self.SENSOR_X,), 
                                                carla.Rotation(pitch=self.pitchCamera, yaw=0, roll=0))
            self.camera_depth = self.world.spawn_actor(camera_depth, camera_transform, attach_to=self.vehicle)
            self.image_queue_depth = queue.Queue()
            self.camera_depth.listen(self.image_queue_depth.put)
            self.actor_list.append(self.camera_depth)

            # Instance Segmentation Camera
            camera_instance_bp = self.blueprint_library.find('sensor.camera.instance_segmentation')
            camera_instance_bp.set_attribute('fov', f'{str(self.FOV)}')
            camera_instance_bp.set_attribute('image_size_x', f'{self.IMG_WIDTH}')
            camera_instance_bp.set_attribute('image_size_y', f'{self.IMG_HEIGHT}')
            self.camera_instance = self.world.spawn_actor(camera_instance_bp, camera_transform, attach_to=self.vehicle)
            self.image_queue_instance = queue.Queue()
            self.camera_instance.listen(self.image_queue_instance.put)
            self.actor_list.append(self.camera_instance)

        ########################################################################################################################
        ####### SEMANTIC LIDAR
        ########################################################################################################################
        if self.generate_lidar:
            lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast_semantic')
            lidar_bp.set_attribute('channels', '64')
            lidar_bp.set_attribute('range', '150.0')
            lidar_bp.set_attribute('points_per_second', '500000')
            lidar_bp.set_attribute('rotation_frequency', '20.0')
            lidar_bp.set_attribute('upper_fov', '15.0')
    def setSensors(self):
        self.camera = self.world.spawn_actor(self.camera_bp, self.camera_transform, attach_to=self.vehicle)
        self.camera.listen(lambda image: self.image_queue.put(image))
        self.actor_list.append(self.camera)

        if self.save_seg:
            self.camera_seg = self.world.spawn_actor(self.camera_seg_bp, self.camera_transform, attach_to=self.vehicle)
            self.camera_seg.listen(lambda image: self.image_queue_seg.put(image))
            self.actor_list.append(self.camera_seg)
            
            self.camera_depth = self.world.spawn_actor(self.camera_depth_bp, self.camera_transform, attach_to=self.vehicle)
            self.camera_depth.listen(lambda image: self.image_queue_depth.put(image))
            self.actor_list.append(self.camera_depth)
            
            self.camera_instance = self.world.spawn_actor(self.camera_instance_bp, self.camera_transform, attach_to=self.vehicle)
            self.camera_instance.listen(lambda image: self.image_queue_instance.put(image))
            self.actor_list.append(self.camera_instance)
        
    def spawnPeople(self, idNum, ID, location, rotation, transform):
        '''
        Parameters
        ----------
        ID : ID of the human to spawn
        idNum : key to save the vehicles position in self.walkerDict
        location: Carla.Location object, with the x,y and z position coordinates of the walker
        rotation: Carla.Rotation object, with roll, pitch and yaw of the walker
        position : Transform with location and rotation, used to spwan the walker in the scene
    
        Returns
        -------
        None
            
        The function spawns a walker based on ID at the given position 
        '''
        person = self.blueprint_library.filter(ID)[0]
        player = self.world.try_spawn_actor(person, transform)
        if player is not None:
            player_rotation = rotation
            player_control = carla.WalkerControl()
            player_control.speed = 3
            pedestrian_heading = 90
            player_rotation = carla.Rotation(pitch=transform.rotation.pitch, yaw=pedestrian_heading, roll=transform.rotation.roll)
            player_control.direction = player_rotation.get_forward_vector()
            player.apply_control(player_control)
            self.spawned_people.append(player)       
            self.walkerDict[idNum] =  transform
             
    def generateTransform(self, position):
        '''
        Parameters
        ----------
        position : Transform with location and rotation information, used to spwan the vehicle/walker in the scene
    
        Returns
        -------
        transform: Carla.Transform object, with carla.Location and carla.Rotation object
        location: carla.Location object: x, y and z position coordinates of the actor
        rotation: carla.Rotation object: roll, pitch and yaw of the actor
            
        The function takes in a string input of the complete position information of the actor and returns a carla.Transform object with the respective location and rotation
        '''
        x = float((position.split("(")[2]).split(",")[0].split('=')[1])
        y = float((position.split("(")[2]).split(",")[1].split('=')[1])
        z = float((position.split("(")[2]).split(",")[2].split('=')[1].split(")")[0])
        
        Rotation = position.split("(")[-1]. split(")")[0]
        pitch = float(Rotation.split(",")[0].split('=')[1])
        yaw = float(Rotation.split(",")[1].split('=')[1])
        roll = float(Rotation.split(",")[2].split('=')[1].split(")")[0])
        transform = carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(pitch=pitch, yaw=yaw, roll=roll))
        
        return transform, carla.Location(x=x, y=y, z=z), carla.Rotation(pitch=pitch, yaw=yaw, roll=roll)
    
    def read_json(self):
        '''
        Parameters
        ------
        None 
        
        Returns
        ------
        None
        
        The function reads a file, makes function calls to spawn actors in the scene
        '''
        i = 0
        filename = os.path.join(self.metaDataDir, str(self.files[self.counter]))

        with open(filename, "r") as json_file:
            self.data = json.load(json_file)

        if self.data["height"] != self.height or self.data["pitch"] != self.pitch:
            self.h_and_p = True
        vehiclePos = self.data["ego_vehicle"]
        self.vehiclePos, _, _ = self.generateTransform(vehiclePos)
        # Note: heightCamera and pitchCamera will be set per-frame in tickClock
        self.vehiclesNum = self.data["total_num_vehicles"]
        self.walkersNum = self.data["total_num_walkers"]

        for veh in self.data["vehicles"]:
            attributes, position = veh.split("\n")[:2]
            idNum = int(attributes.split("(")[1].split(",")[0].split("=")[1])
            ID = ((attributes.split('(')[1]).split(',')[1]).split('=')[-1].split(')')[0]
            typeActor = (((attributes.split('(')[1]).split(',')[1]).split('=')[-1]).split('.')[0]
            transform, loc, rotation = self.generateTransform(position)
            if rotation.roll <= -1 or rotation.roll >= 1:
                continue
            if((idNum in self.vehicleDict) == False):
                self.spawnVehicles(idNum, ID, transform)

        if self.args.load_old is not None: 
            filename = filename.replace(self.args.ROOT_DIR, self.args.load_old) # loading original clearnoon generated file

        with open(filename, "r") as json_file:
            self.data_p = json.load(json_file)

        for veh in self.data_p["walkers"]:
            attributes, position = veh.split("\n")[:2]
            idNum = int(attributes.split("(")[1].split(",")[0].split("=")[1])
            ID = ((attributes.split('(')[1]).split(',')[1]).split('=')[-1].split(')')[0]
            typeActor = (((attributes.split('(')[1]).split(',')[1]).split('=')[-1]).split('.')[0]
            transform, loc, rotation = self.generateTransform(position)
            if((idNum in self.walkerDict) == False):
                self.spawnPeople(idNum, ID, loc, rotation, transform)

    def tickClock(self):
        self.add_sensors(self.vehicle)
        self.counter = 0
        '''
        Function call to generate images
        Algorithm:
            * Read the file
            * Set the sensor carrying vehicle position
            * Tick the world, capture images from all the sensors 
            * Save all the images and the associated metaData
        '''
        from tqdm import tqdm 
        pbar = tqdm(total=self.totalImages)
        while(self.counter<self.totalImages):
            self.read_json()
            if self.noon_json:
                filename = os.path.join(self.metaDataDir, str(self.files[self.counter]))
                filename = filename.replace("H_35_P_45", f"H_{self.height}_P_{abs(self.pitch)}")
                # filename = filename.replace("second_regen_new", "second_regen_new_regen") # P = 90 only
                filename = filename.replace("second_regen_new", "fix_meta") # P = 0/45/60 | H = 15/35/60
                with open(filename, "r") as json_file:
                    noon_data = json.load(json_file)
                self.heightCamera = noon_data["actual_height"]
                self.pitchCamera = noon_data["actual_pitch"]
            else:
                if self.h_and_p:
                    if self.height <= 5.0:
                        actual_sigma_h = 0.1
                        actual_sigma_p = 0.5
                    else:
                        actual_sigma_h = self.SIGMA_H
                        actual_sigma_p = self.SIGMA_P
                    self.heightCamera = np.random.normal(self.height, actual_sigma_h)
                    self.pitchCamera = np.random.normal(self.pitch, actual_sigma_p)
                else:
                    self.heightCamera = self.data["actual_height"]
                    self.pitchCamera = self.data["actual_pitch"]

            # Ensure we are relative to road
            waypoint = self.m.get_waypoint(self.vehiclePos.location)
            veh_transform = carla.Transform(carla.Location(x=self.vehiclePos.location.x, y=self.vehiclePos.location.y, z=waypoint.transform.location.z + self.heightCamera), 
                                            carla.Rotation(pitch=0, yaw=self.vehiclePos.rotation.yaw, roll=0))
            self.vehicle.set_transform(veh_transform)
            cam_transform = carla.Transform(carla.Location(x=self.SENSOR_X,), 
                                           carla.Rotation(pitch=self.pitchCamera, yaw=0, roll=0))
            self.camera.set_transform(cam_transform)
            if self.save_seg:
                self.camera_seg.set_transform(cam_transform)
                self.camera_depth.set_transform(cam_transform)
                self.camera_instance.set_transform(cam_transform)
            if self.generate_lidar:
                self.lidar.set_transform(cam_transform)
            
            # Freeze physics for all spawned actors just before the first tick for this frame
            for actor in self.spawned_vehicle + self.spawned_people:
                if actor.is_alive:
                    actor.set_simulate_physics(False)

            self.world.tick()
            time.sleep(5)
            ########################################################################################################################
            ####### IMAGES
            ########################################################################################################################
            ##### AERIAL VIEW
            try:
                image = self.sensor_queues['rgb'].get(timeout=20.0)
                self.current_gbuffers = self.get_gbuffers(timeout=20.0)
                image_segCarla  = self.sensor_queues['seg'].get(timeout=20.0)
                image_segCarla.convert(carla.ColorConverter.CityScapesPalette)
                image_depth = self.sensor_queues['depth'].get(timeout=20.0)
                image_instance = self.sensor_queues['instance'].get(timeout=20.0)
                if self.generate_lidar:
                    point_cloud = self.sensor_queues['lidar'].get(timeout=20.0)
            except queue.Empty:
                logger.error("Error: Sensor queue timeout. Simulator might be lagging or sensor failed.")
                continue

            ########################################################################################################################
            if True:
                logger.info(f"Saving sample {self.counter + 1}/{self.totalImages} (Frame {image.frame})")
                ## Save the image, carla Dict segmented map, depth image, ground scene and carla dict segmented map for ground map
                ########################################################################################################################
                ####### IMAGES
                ########################################################################################################################
                ##### AERIAL VIEW
                imgName = str(self.files[self.counter]).split(".")[0]
                IMG_PATH = os.path.join(self.save_dir, f"Images/{imgName}.png")
                image.save_to_disk(IMG_PATH)
                ########################################################################################################################
                ####### SEMANTIC SEGMENTATION
                ########################################################################################################################
                ##### AERIAL VIEW
                if self.save_seg:
                    image_segCarla.save_to_disk(os.path.join(self.save_dir, f"CarlaSegment/{imgName}.png"))
                    image_depth.save_to_disk(os.path.join(self.save_dir, f"Depth/{imgName}.png"), carla.ColorConverter.LogarithmicDepth)
                    image_instance.save_to_disk(os.path.join(self.save_dir, f"Instance/{imgName}.png"))
                if self.extract_gbuffer:
                    npz_path = os.path.join(self.save_dir, f"GBuffer/{imgName}_gbuffer.npz")
                    np.savez_compressed(npz_path, **self.current_gbuffers)
                
                if self.generate_lidar:
                    points_lidar, colors_lidar, labels_lidar = self.process_lidar(point_cloud, image)
                    
                    ply_path = os.path.join(self.save_dir, f"Lidar/{imgName}.ply")
                    with open(ply_path, 'wb') as f_ply:
                        header = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(points_lidar)}\nproperty float x\nproperty float y\nproperty float z\nproperty uint8 red\nproperty uint8 green\nproperty uint8 blue\nproperty uint ObjTag\nend_header\n"
                        f_ply.write(header.encode('utf-8'))
                        ply_data = np.zeros(len(points_lidar), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('ObjTag', 'u4')])
                        ply_data['x'] = points_lidar[:, 0]
                        ply_data['y'] = points_lidar[:, 1]
                        ply_data['z'] = points_lidar[:, 2]
                        ply_data['red'] = colors_lidar[:, 0]
                        ply_data['green'] = colors_lidar[:, 1]
                        ply_data['blue'] = colors_lidar[:, 2]
                        ply_data['ObjTag'] = labels_lidar
                        f_ply.write(ply_data.tobytes())
            ########################################################################################################################
            if self.save_metadata:
                data = {}
                data["image_path"] = IMG_PATH
                data["h_and_p"] = self.h_and_p
                data["generated"] = False
                data["re-generated"] = True
                data["town"] = self.town
                data["weather"] = self.weather_str
                data["IMG_HEIGHT"] = self.IMG_HEIGHT
                data["IMG_WIDTH"] = self.IMG_WIDTH
                data["SIGMA_H"] = self.SIGMA_H
                data["SIGMA_P"] = self.SIGMA_P
                data["height"] = self.height
                data["pitch"] = self.pitch
                data["actual_height"] = self.heightCamera
                data["actual_pitch"] = self.pitchCamera
                data["ego_vehicle"] = str(self.vehicle.get_transform())
                
                data["num_walkers_spawned"] = self.data["num_walkers_spawned"]
                data["num_walkers_spawned_sidewalk"] = self.data["num_walkers_spawned_sidewalk"]
                data["total_num_walkers"] = self.data["num_walkers_spawned"] + self.data["num_walkers_spawned_sidewalk"]
                data["total_num_vehicles"] = self.data["total_num_vehicles"]
                data["actual_num_walkers"] = len(self.spawned_people) 
                data["actual_num_vehicles"] = len(self.spawned_vehicle) 
                data["sensors"] = [str(item) + "\n" + str(item.get_transform()) + "\n" for item in self.actor_list]
                data["vehicles"] = [str(item) + "\n" + str(item.get_transform()) + "\n" for item in self.spawned_vehicle]
                data["walkers"] = [str(item) + "\n" + str(item.get_transform()) + "\n" for item in self.spawned_people]
                data["veh_dict"] = [str(item) + "\n" + str(self.vehicleDict[item]) + "\n" for item in self.vehicleDict.keys()]
                data["walker_dict"] = [str(item) + "\n" + str(self.walkerDict[item]) + "\n" for item in self.walkerDict.keys()]
                JSON_PATH = os.path.join(self.ROOT_DIR, f"H_{self.height}_P_{abs(self.pitch)}/{self.weather_str}/{self.town}/metaData/{imgName}.json")
                with open(JSON_PATH, "w") as json_file:
                    json.dump(data, json_file)
            
            self.counter += 1
            pbar.update(1)
            self.destroyVehPeople()
        self.destroyVehPeople()
        pbar.close()
        
    def destroyVehPeople(self):
        '''
        Function call the destroy all the actors spawned in the current iteration
        '''
        try:
            if self.spawned_people:
                self.client.apply_batch_sync([carla.command.DestroyActor(x) for x in self.spawned_people])
            if self.spawned_vehicle:
                self.client.apply_batch_sync([carla.command.DestroyActor(x) for x in self.spawned_vehicle])
        except Exception as e:
            print(f"Error destroying vehicle and people: {e}")
        finally:
            self.spawned_vehicle = []
            self.spawned_people = []
            self.vehicleDict = {}
            self.walkerDict = {}
    
    def destroyActors(self):
        super().cleanup()
        if hasattr(self, 'vehiclesList') and self.vehiclesList:
            logger.info(f"Destroying {len(self.vehiclesList)} vehicles...")
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehiclesList])
        
        if hasattr(self, 'walkersList') and self.walkersList:
            logger.info(f"Destroying {len(self.walkersList)} walkers...")
            for i in range(0, len(self.all_id), 2):
                self.all_actors[i].stop()
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.all_id])
        self.client.apply_batch_sync([carla.command.DestroyActor(x) for x in self.spawned_people])
        try:
            if hasattr(self, 'world'):
                self.world.tick()
        except Exception as e:
            print(f"Error during actor destruction: {e}")
        finally:
            self.endTime = time.time()
        

if __name__ == "__main__":
    ###########################################################################################################################################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default="127.0.0.1", help="host location")
    parser.add_argument('--port', type=int, default=2000, help="port number")
    parser.add_argument('--town', type=str, default="Town10HD", help="Town01 Town02 Town03 Town04 Town05 Town06 Town07 Town10HD")
    parser.add_argument('--weather', type=str, default="ClearNoon", help="ClearNoon CloudyNoon MidRainyNoon ClearSunset ClearNight")
    parser.add_argument('--height', type=float, default=35, help="height")
    parser.add_argument('--pitch', type=int, default=-45, help="pitch")
    parser.add_argument('--metaDataDir', type=str, help="metaDataDir")
    parser.add_argument('--ROOT_DIR', type=str, default="/home/df/data/datasets/SkyScenes/proof_of_concept", help="ROOT_DIR")
    parser.add_argument('--index', type=int, default=0, help="index of the last image generated, incase generation stops midway")
    parser.add_argument('--load_old', type=str, default=None, help="loading old json file for missing vehicle/walker")
    parser.add_argument('--noon_json', type=bool, default=False, help="Fixing missing")
    parser.add_argument('--save_seg', action='store_true', default=False, help="Save segmentation, depth and instance maps")
    parser.add_argument('--extract_gbuffer', action='store_true', default=False, help="Extract G-Buffer layers and save as NPZ")
    parser.add_argument('--generate_lidar', action='store_true', default=False, help='Build whole-town colorized maps')
    
    # Sensor settings
    parser.add_argument('--fov', type=float, default=110.0, help='Camera FOV')
    parser.add_argument('--width', type=int, default=2160, help='Image width')
    parser.add_argument('--height_img', type=int, default=1440, help='Image height')
    parser.add_argument('--sensor_x', type=float, default=5.0, help='Sensor X offset')
    parser.add_argument('--lidar_range', type=float, default=150.0, help='LiDAR range')
    parser.add_argument('--voxel_size', type=float, default=0.1, help='Voxel size for mapping')

    parser.add_argument('--tm_port', type=int, default=8000, help="Traffic Manager port")

    args = parser.parse_args()
    args.index = 0
    ###
    if args.metaDataDir is None:
        meta_dir = "./meta_data/second_regen_new/H_35_P_45/ClearNoon/Town01/metaData"
        args.metaDataDir = meta_dir.replace("Town01", args.town)
    ###
    print("__"*20)
    print(f"The arguments for generation are as follows: ")
    print(f"Town:          {args.town}")
    print(f"Weather:       {args.weather}")
    print(f"Height:        {args.height}")
    print(f"Pitch:         {args.pitch}")
    print(f"metaDataDir:   {args.metaDataDir}")

    gn = GenImage(args, args.metaDataDir, args.index)