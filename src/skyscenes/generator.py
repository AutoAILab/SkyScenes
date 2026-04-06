import os
import time
import queue
import numpy as np
import carla
import open3d as o3d
import logging

# Set up standard logging for generators
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BaseGenerator:
    """
    Base class for SkyScenes data generation.
    Handles CARLA world initialization, sensor setup, and point cloud processing.
    """
    def __init__(self, args):
        self.args = args
        self.town = args.town
        self.weather_str = args.weather
        self.height = args.height
        self.pitch = args.pitch
        self.ROOT_DIR = args.ROOT_DIR
        
        # Sensor parameters (with defaults if not provided)
        self.img_width = getattr(args, 'width', 2160)
        self.img_height = getattr(args, 'height_img', 1440)
        self.fov = getattr(args, 'fov', 110)
        self.sensor_x = getattr(args, 'sensor_x', 5.0)
        self.lidar_range = getattr(args, 'lidar_range', 150.0)
        self.voxel_size = getattr(args, 'voxel_size', 0.1)
        
        # Internal state
        self.actor_list = []
        self.sensor_queues = {}
        self.global_pcd = o3d.geometry.PointCloud()
        
        # CARLA initialization
        self.client = carla.Client(args.host, args.port)
        self.client.set_timeout(40.0)
        self.world = self.client.load_world(self.town)
        self.blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()
        
        # Traffic Manager
        self.tm = self.client.get_trafficmanager(args.tm_port)
        self.tm.set_synchronous_mode(True)
        
        # Synchronous mode settings
        self.original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)
        
    def add_sensors(self, parent_actor):
        """
        Attaches the suite of sensors to the parent actor.
        """
        # Camera sensors
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(self.img_width))
        camera_bp.set_attribute('image_size_y', str(self.img_height))
        camera_bp.set_attribute('fov', str(self.fov))
        
        spawn_point = carla.Transform(carla.Location(x=self.sensor_x), 
                                     carla.Rotation(pitch=self.pitch))
        
        self.rgb_camera = self.world.spawn_actor(camera_bp, spawn_point, attach_to=parent_actor)
        self.register_sensor('rgb', self.rgb_camera)
        
        # G-Buffer capture
        if getattr(self.args, 'extract_gbuffer', False):
            self.gbuffer_queues = {}
            for gb in ["SceneDepth", "SceneStencil", "GBufferA", "GBufferB", "GBufferC"]:
                self.gbuffer_queues[gb] = queue.Queue()
                self.rgb_camera.listen_to_gbuffer(getattr(carla.GBufferTextureID, gb), self.gbuffer_queues[gb].put)

        # Semantic Segmentation
        seg_bp = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        seg_bp.set_attribute('image_size_x', str(self.img_width))
        seg_bp.set_attribute('image_size_y', str(self.img_height))
        seg_bp.set_attribute('fov', str(self.fov))
        self.seg_camera = self.world.spawn_actor(seg_bp, spawn_point, attach_to=parent_actor)
        self.register_sensor('seg', self.seg_camera)
        
        # Depth Camera
        depth_bp = self.blueprint_library.find('sensor.camera.depth')
        depth_bp.set_attribute('image_size_x', str(self.img_width))
        depth_bp.set_attribute('image_size_y', str(self.img_height))
        depth_bp.set_attribute('fov', str(self.fov))
        self.depth_camera = self.world.spawn_actor(depth_bp, spawn_point, attach_to=parent_actor)
        self.register_sensor('depth', self.depth_camera)
        
        # Instance Segmentation
        instance_bp = self.blueprint_library.find('sensor.camera.instance_segmentation')
        instance_bp.set_attribute('image_size_x', str(self.img_width))
        instance_bp.set_attribute('image_size_y', str(self.img_height))
        instance_bp.set_attribute('fov', str(self.fov))
        self.instance_camera = self.world.spawn_actor(instance_bp, spawn_point, attach_to=parent_actor)
        self.register_sensor('instance', self.instance_camera)

        # Semantic LiDAR
        if getattr(self.args, 'generate_lidar', False):
            lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast_semantic')
            lidar_bp.set_attribute('channels', '64')
            lidar_bp.set_attribute('range', str(self.lidar_range))
            lidar_bp.set_attribute('points_per_second', '500000')
            lidar_bp.set_attribute('rotation_frequency', '20.0')
            lidar_bp.set_attribute('upper_fov', '15.0')
            lidar_bp.set_attribute('lower_fov', '-25.0')
            self.lidar = self.world.spawn_actor(lidar_bp, spawn_point, attach_to=parent_actor)
            self.register_sensor('lidar', self.lidar)

    def register_sensor(self, name, actor):
        q = queue.Queue()
        actor.listen(q.put)
        self.sensor_queues[name] = q
        self.actor_list.append(actor)

    def get_gbuffers(self, timeout=20.0):
        """
        Retrieves all active G-Buffer textures.
        """
        gbuffers = {}
        if not getattr(self.args, 'extract_gbuffer', False):
            return gbuffers
            
        for gb in ["SceneDepth", "SceneStencil", "GBufferA", "GBufferB", "GBufferC"]:
            gb_img = self.gbuffer_queues[gb].get(timeout=timeout)
            gb_array = np.frombuffer(gb_img.raw_data, dtype=np.dtype("uint8"))
            gb_array = np.reshape(gb_array, (gb_img.height, gb_img.width, 4))[:, :, :3][:, :, ::-1]
            gbuffers[gb] = gb_array
        return gbuffers

    def get_intrinsics(self):
        f = self.img_width / (2.0 * np.tan(self.fov * np.pi / 360.0))
        cx = self.img_width / 2.0
        cy = self.img_height / 2.0
        return f, cx, cy

    def process_lidar(self, lidar_data, rgb_image):
        """
        Colorizes LiDAR points and accumulates them into the global map.
        """
        # Parse points
        data = np.frombuffer(lidar_data.raw_data, dtype=np.dtype([
            ('x', np.float32), ('y', np.float32), ('z', np.float32),
            ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)]))
        
        points = np.array([data['x'], -data['y'], data['z']]).T
        tags = np.array(data['ObjTag'])
        
        # 1. Project to image
        rgb_array = np.frombuffer(rgb_image.raw_data, dtype=np.dtype("uint8"))
        rgb_array = np.reshape(rgb_array, (rgb_image.height, rgb_image.width, 4))[:, :, :3]
        
        f, cx, cy = self.get_intrinsics()
        
        # Project points: x_cam = y_lidar, y_cam = -z_lidar, z_cam = x_lidar
        z_cam = points[:, 0]
        x_cam = points[:, 1]
        y_cam = -points[:, 2]
        
        u = (x_cam * f / z_cam) + cx
        v = (y_cam * f / z_cam) + cy
        
        valid_idx = (u >= 0) & (u < self.img_width) & (v >= 0) & (v < self.img_height) & (z_cam > 0)
        u_valid = u[valid_idx].astype(np.int32)
        v_valid = v[valid_idx].astype(np.int32)
        
        colors = np.zeros((len(points), 3), dtype=np.uint8)
        colors[valid_idx] = rgb_array[v_valid, u_valid][:, ::-1] # BGR to RGB
        
        # 2. Transform to world space
        trans = self.lidar.get_transform().get_matrix()
        points_hom = np.c_[points, np.ones(len(points))]
        points_world = np.dot(trans, points_hom.T).T[:, :3]
        
        # 3. Accumulate
        frame_pcd = o3d.geometry.PointCloud()
        frame_pcd.points = o3d.utility.Vector3dVector(points_world)
        frame_pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
        self.global_pcd += frame_pcd
        
        return points, colors, tags

    def cleanup(self):
        """
        Stops sensors and destroys actors.
        """
        logger.info("Starting cleanup...")
        # Save global map if lidar was enabled
        if getattr(self.args, 'generate_lidar', False) and len(self.global_pcd.points) > 0:
            self.global_pcd = self.global_pcd.voxel_down_sample(self.voxel_size)
            save_path = os.path.join(self.ROOT_DIR, f"H_{self.height}_P_{abs(self.pitch)}", self.weather_str, self.town, "town_accumulated.ply")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            o3d.io.write_point_cloud(save_path, self.global_pcd)
            logger.info(f"Global map saved to {save_path}")

        # Stop listening
        for q in self.sensor_queues.values():
            # There isn't a direct .stop() on queue, but we should stop the sensors
            pass
        
        # Destroy actors in batch
        self.world.apply_settings(self.original_settings)
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
        logger.info("Cleanup complete.")
