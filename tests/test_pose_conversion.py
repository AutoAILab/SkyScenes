import unittest
import numpy as np
from scripts.convert_poses_to_kitti import parse_carla_transform, carla_to_matrix, transform_to_kitti, verify_pose

class TestPoseConversion(unittest.TestCase):
    def test_parse_carla_transform(self):
        s = "Transform(Location(x=50.2, y=-12.5, z=35.0), Rotation(pitch=-45.0, yaw=90.0, roll=0.0))"
        res = parse_carla_transform(s)
        self.assertEqual(res['x'], 50.2)
        self.assertEqual(res['y'], -12.5)
        self.assertEqual(res['z'], 35.0)
        self.assertEqual(res['pitch'], -45.0)
        self.assertEqual(res['yaw'], 90.0)
        self.assertEqual(res['roll'], 0.0)

    def test_parse_prefixed_transform(self):
        s = "sensor.camera.rgb(id=18) Transform(Location(x=1, y=2, z=3), Rotation(pitch=4, yaw=5, roll=6))"
        res = parse_carla_transform(s)
        self.assertEqual(res['x'], 1.0)
        self.assertEqual(res['pitch'], 4.0)

    def test_identity_matrix(self):
        loc_rot = {'x': 0, 'y': 0, 'z': 0, 'pitch': 0, 'yaw': 0, 'roll': 0}
        m = carla_to_matrix(loc_rot)
        self.assertTrue(np.allclose(m, np.eye(4)))
        self.assertTrue(verify_pose(m))

    def test_kitti_transformation(self):
        # CARLA Identity should map to KITTI Identity (if T_C2K is used correctly)
        m_c = np.eye(4)
        m_k = transform_to_kitti(m_c)
        self.assertTrue(np.allclose(m_k, np.eye(4)))
        
        # 90 degree yaw in CARLA (rotation around Z)
        # In CARLA: X becomes Y, Y becomes -X
        loc_rot = {'x': 0, 'y': 0, 'z': 0, 'pitch': 0, 'yaw': 90, 'roll': 0}
        m_c = carla_to_matrix(loc_rot)
        # Verify M_c looks right: [0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]
        self.assertAlmostEqual(m_c[0, 0], 0)
        self.assertAlmostEqual(m_c[0, 1], -1)
        self.assertAlmostEqual(m_c[1, 0], 1)
        self.assertAlmostEqual(m_c[1, 1], 0)
        
        m_k = transform_to_kitti(m_c)
        # In KITTI, Y is down, Z is forward.
        # CARLA Yaw 90 means the camera turned left.
        # In KITTI (X-right, Y-down, Z-forward), turning left means rotation around Y axis.
        # R_y(theta) = [[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]]
        # For theta = -90 (left turn): [[0, 0, -1], [0, 1, 0], [1, 0, 0]]
        self.assertTrue(verify_pose(m_k))

if __name__ == '__main__':
    unittest.main()
