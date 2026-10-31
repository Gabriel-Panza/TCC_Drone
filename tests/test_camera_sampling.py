import unittest
import cv2
import numpy as np

from spatial_mapping.camera_sampling import optical_rays,matched_nearest_pair,box_depth,reprojection_metrics


class CameraSamplingTest(unittest.TestCase):
    def test_gazebo_pixel_centres_produce_symmetric_native_rays(self):
        k=np.array([[2.,0,2],[0,2.,1],[0,0,1]])
        rays=optical_rays(4,2,k)
        np.testing.assert_allclose(rays[0,:,0],[-.75,-.25,.25,.75])
        np.testing.assert_allclose(rays[:,:,2],1.)

    def test_matched_sampling_preserves_thin_foreground_label(self):
        rgb=np.zeros((8,16,3),dtype=np.uint8);rgb[:,:,0]=255
        rgb[:,4]=[0,255,0]
        depth=np.full((8,16),10.);depth[:,4]=2.
        k=np.array([[10.,0,8],[0,10.,4],[0,0,1]])
        mixed=cv2.resize(rgb,(4,2),interpolation=cv2.INTER_AREA)
        label=cv2.resize(depth,(4,2),interpolation=cv2.INTER_NEAREST)<5
        self.assertFalse(np.array_equal(mixed[:,:,1]>mixed[:,:,0],label))
        paired=matched_nearest_pair(rgb,depth,k,(4,2))
        np.testing.assert_array_equal(paired['rgb_rgb'][:,:,1]>paired['rgb_rgb'][:,:,0],paired['reference_depth_m']<5)

    def test_noninteger_resize_keeps_exact_native_coordinates(self):
        rgb=np.zeros((6,6,3),dtype=np.uint8);depth=np.ones((6,6))
        k=np.array([[3.,0,3],[0,3.,3],[0,0,1]])
        paired=matched_nearest_pair(rgb,depth,k,(4,4))
        np.testing.assert_array_equal(paired['source_columns'],[0,1,3,4])
        np.testing.assert_allclose(paired['rays_optical'][0,:,0],(np.array([0,1,3,4])+.5-3)/3)

    def test_box_intersection_preserves_optical_depth_not_radial_range(self):
        rays=np.array([[1.,0.,1.],[1.,-1.,0.]])
        depth=box_depth([0.,0.,0.],rays,[10.,-20.,-20.],[10.1,20.,20.])
        np.testing.assert_allclose(depth,[10.,10.])

    def test_identical_pose_reprojection_is_exact(self):
        k=np.array([[10.,0,10],[0,10.,10],[0,0,1]])
        depth=np.full((20,20),5.)
        result=reprojection_metrics(depth,optical_rays(20,20,k),np.eye(4),depth,k,np.eye(4),stride=1)
        self.assertEqual(result['agreement_fraction'],1.)
        self.assertEqual(result['disappeared_fraction'],0.)

    def test_occlusion_is_not_reported_as_disappearance(self):
        k=np.array([[10.,0,10],[0,10.,10],[0,0,1]])
        source=np.full((20,20),10.);target=np.full((20,20),5.)
        result=reprojection_metrics(source,optical_rays(20,20,k),np.eye(4),target,k,np.eye(4),stride=1)
        self.assertEqual(result['occluded_fraction'],1.)
        self.assertEqual(result['disappeared_fraction'],0.)
