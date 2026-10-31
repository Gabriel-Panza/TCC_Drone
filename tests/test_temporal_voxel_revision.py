import unittest
import numpy as np
from scripts.temporal_voxel_revision import TemporalVoxelRevision
from spatial_mapping.occupancy import OccupancyGrid3D
from spatial_mapping.projected_free_evidence import projected_free_voxels
from spatial_mapping.geometry import CameraIntrinsics

class TemporalRevisionTests(unittest.TestCase):
    voxel=(0,0,4)
    def test_static_camera_and_duplicate_views_never_retire(self):
        r=TemporalVoxelRevision();p={self.voxel}
        for i in range(20):self.assertFalse(r.observe(p,p if i==0 else set(),p,[0,0,0],i))
        for i in range(20):self.assertFalse(r.observe(p,set(),p,[1,0,0],20+i))
    def test_three_separated_free_views_retire_then_unknown_requires_real_rays(self):
        r=TemporalVoxelRevision();p={self.voxel}
        r.observe(p,p,set(),[0,0,0],1)
        self.assertFalse(r.observe(p,set(),p,[1,0,0],2))
        self.assertFalse(r.observe(p,set(),p,[2,0,0],3))
        self.assertEqual(r.observe(p,set(),p,[3,0,0],4),p)
        g=OccupancyGrid3D();g._confirmed_occupied.add(self.voxel);g._log_odds[self.voxel]=3.
        r.retire_to_unknown(g,p)
        self.assertEqual(g.state(self.voxel),"unknown")
        self.assertNotIn(self.voxel,g.free_voxels())
        origin=np.array([.2,.2,.2]);endpoint=np.array([[.2,.2,8.]])
        g.integrate_rays(origin,endpoint,np.array([False]))
        self.assertNotIn(self.voxel,g.free_voxels())
        g.integrate_rays(origin,endpoint,np.array([False]))
        self.assertIn(self.voxel,g.free_voxels())
    def test_new_positive_hit_resets_earlier_free_votes(self):
        r=TemporalVoxelRevision();p={self.voxel}
        r.observe(p,p,set(),[0,0,0],1)
        r.observe(p,set(),p,[1,0,0],2);r.observe(p,set(),p,[2,0,0],3)
        self.assertFalse(r.observe(p,p,p,[3,0,0],4))
        self.assertFalse(r.observe(p,set(),p,[4,0,0],5))
        self.assertFalse(r.observe(p,set(),p,[5,0,0],6))
        self.assertEqual(r.observe(p,set(),p,[6,0,0],7),p)
    def test_unknown_and_thin_foreground_block_full_footprint_evidence(self):
        k=CameraIntrinsics(20.,20.,50.,50.);p={self.voxel}
        for bad in [np.nan,3.]:
            image=np.full((100,100),10.);image[52,52]=bad
            self.assertFalse(projected_free_voxels(p,1.,k,np.eye(4),image))
    def test_no_expiry_without_explicit_free_proof(self):
        r=TemporalVoxelRevision();p={self.voxel}
        r.observe(p,p,set(),[0,0,0],1)
        for i in range(2,200):self.assertFalse(r.observe(p,set(),set(),[i,0,0],i))
    def test_invalid_configuration_or_origin_rejected(self):
        with self.assertRaises(ValueError):TemporalVoxelRevision(2)
        with self.assertRaises(ValueError):TemporalVoxelRevision(baseline_m=np.nan)
        with self.assertRaises(ValueError):TemporalVoxelRevision().observe(set(),set(),set(),[0,np.nan,0],1)
