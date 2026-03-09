import unittest
from unittest.mock import patch, MagicMock
import os
import yaml
from pipeline.run_generation import main

class TestPipelineOrchestrator(unittest.TestCase):
    def setUp(self):
        self.config = {
            "ROOT_DIR": "/tmp/datasets",
            "save_seg": False,
            "baseline": {
                "towns": ["Town01"],
                "weather": "ClearNoon",
                "heights": [35],
                "pitches": [-45],
                "num_images": 2
            },
            "variations": {
                "weather": ["ClearSunset"],
                "heights": [],
                "pitches": []
            }
        }
        with open("test_config.yaml", "w") as f:
            yaml.dump(self.config, f)

    def tearDown(self):
        if os.path.exists("test_config.yaml"):
            os.remove("test_config.yaml")

    @patch("pipeline.run_generation.run_command")
    @patch("os.path.exists")
    @patch("os.listdir")
    def test_pipeline_logic(self, mock_listdir, mock_exists, mock_run):
        # Mocking skip scenarios
        mock_exists.return_value = False
        mock_run.return_value = 0
        
        # We need to mock the argparse and we'll just run main with our test config
        with patch("sys.argv", ["run_generation.py", "--config", "test_config.yaml"]):
            main()
        
        # Check if manualSpawning.py was called
        # The expected command should NOT have --save_seg because config has save_seg: False
        baseline_call = mock_run.call_args_list[0][0][0]
        self.assertIn("manualSpawning.py", baseline_call)
        self.assertNotIn("--save_seg", baseline_call)
        self.assertIn("--num", baseline_call)
        self.assertIn("2", baseline_call)

        # Check if loadingAttributesWeather.py was called for variation
        variation_call = mock_run.call_args_list[1][0][0]
        self.assertIn("loadingAttributesWeather.py", variation_call)
        self.assertIn("ClearSunset", variation_call)
        self.assertNotIn("--save_seg", variation_call)

    @patch("pipeline.run_generation.run_command")
    @patch("os.path.exists")
    @patch("os.listdir")
    def test_pipeline_with_seg(self, mock_listdir, mock_exists, mock_run):
        mock_exists.return_value = False
        mock_run.return_value = 0
        
        self.config["save_seg"] = True
        with open("test_config.yaml", "w") as f:
            yaml.dump(self.config, f)

        with patch("sys.argv", ["run_generation.py", "--config", "test_config.yaml"]):
            main()
        
        baseline_call = mock_run.call_args_list[0][0][0]
        self.assertIn("--save_seg", baseline_call)

if __name__ == "__main__":
    unittest.main()
