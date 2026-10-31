import unittest
import numpy as np

from spatial_mapping.geometry import CameraIntrinsics
from spatial_mapping.projected_free_evidence import projected_free_voxels


class ProjectedFreeEvidenceTest(unittest.TestCase):
    intrinsics = CameraIntrinsics(20.,20.,50.,50.)

    def prove(self, lower, voxels={(0,0,4)}, transform=None):
        return projected_free_voxels(voxels,1.,self.intrinsics,
            np.eye(4) if transform is None else transform,lower)

    def test_full_footprint_clears_far_face(self):
        self.assertEqual(self.prove(np.full((100,100),10.)),{(0,0,4)})

    def test_one_background_ray_cannot_erase_a_partial_obstacle(self):
        lower = np.full((100,100),10.)
        lower[52,52] = 3.
        self.assertFalse(self.prove(lower))

    def test_far_face_not_center_must_be_cleared(self):
        self.assertFalse(self.prove(np.full((100,100),4.8)))

    def test_unknown_pixel_and_partial_field_of_view_are_not_free_proof(self):
        lower = np.full((100,100),10.)
        lower[52,52] = np.nan
        self.assertFalse(self.prove(lower))
        self.assertFalse(self.prove(np.full((100,100),10.),{(100,0,4)}))
        self.assertFalse(self.prove(np.full((100,100),10.),{(0,0,-4)}))

    def test_pose_rotation_and_translation_are_used(self):
        lower = np.full((100,100),10.)
        lower[52,52] = 3.
        transform = np.eye(4)
        transform[0,3] = 2.
        self.assertFalse(self.prove(lower,{(2,0,4)},transform))
        transform = np.eye(4)
        transform[:3,:3] = [[0,-1,0],[1,0,0],[0,0,1]]
        self.assertFalse(self.prove(lower,{(-1,0,4)},transform))
