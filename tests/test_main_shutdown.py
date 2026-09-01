import unittest
from pathlib import Path


class MainShutdownOwnershipTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.main_source = (root / 'main.py').read_text()
        cls.controller_source = (root / 'drone_controller.py').read_text()

    def test_executor_owner_observes_controller_shutdown_request(self):
        loop = self.main_source[
            self.main_source.index('while ('):
            self.main_source.index('if stop_requested:')
        ]
        self.assertIn("getattr(controller_node, 'shutdown_requested', False)", loop)

    def test_controller_does_not_shutdown_context_from_callback_thread(self):
        start = self.controller_source.index('def comando_exit(self):')
        end = self.controller_source.index('def image_timestamp_s(', start)
        block = self.controller_source[start:end]
        self.assertIn('self.shutdown_requested = True', block)
        self.assertNotIn('rclpy.shutdown()', block)

    def test_main_remains_the_context_shutdown_owner(self):
        finalizer = self.main_source[self.main_source.index('finally:'):]
        self.assertIn('if rclpy.ok():', finalizer)
        self.assertIn('rclpy.shutdown()', finalizer)



if __name__ == '__main__':
    unittest.main()
