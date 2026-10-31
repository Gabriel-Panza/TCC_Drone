import unittest
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np

from scripts.run_monocular_resolution_640 import sampling_intrinsics,prepare_capture
from estudos_e_analises.avaliar_mapeamento_monocular_offline import frame_geometry
from spatial_mapping.camera_sampling import optical_rays
from spatial_mapping.geometry import CameraIntrinsics,backproject_depth


class ResolutionGeometryTest(unittest.TestCase):
    def test_integer_decimation_backprojects_to_actual_native_sample(self):
        k=np.array([[540.,0,640.],[0,540.,480.],[0,0,1.]])
        native=optical_rays(1280,960,k)
        for width,height in [(320,240),(640,480)]:
            intrinsics=CameraIntrinsics(*sampling_intrinsics(k,1280,width))
            depth=np.full((height,width),5.)
            points=backproject_depth(depth,intrinsics,stride=1)
            # Exact physical rays, including first and last pixels, must agree.
            expected=native[::1280//width,::1280//width].reshape(-1,3)*5.
            np.testing.assert_allclose(points,expected,atol=1e-12)

    def test_legacy_convention_is_explicit_and_not_silently_reinterpreted(self):
        k=np.array([[540.,0,640.],[0,540.,480.],[0,0,1.]])
        old=sampling_intrinsics(k,1280,320,legacy=True)
        np.testing.assert_allclose(old,[135.,135.,160.,120.])
        self.assertNotEqual(old[2],sampling_intrinsics(k,1280,320)[2])

    def test_noninteger_decimation_requires_explicit_rays(self):
        with self.assertRaises(ValueError):
            sampling_intrinsics(np.eye(3),1280,686)

    def test_reference_geometry_is_independent_of_estimator_resolution_and_pose(self):
        estimated=np.eye(4);reference=np.eye(4);reference[0,3]=2.
        frame=dict(camera_to_ned=estimated,reference_camera_to_ned=reference,
                   intrinsics=np.array([135.,135.,160.,120.]),
                   reference_intrinsics=np.array([270.,270.,319.75,239.75]))
        transform,k,ref_transform,ref_k=frame_geometry(frame)
        self.assertEqual(k.fx,135.)
        self.assertEqual(ref_k.fx,270.)
        self.assertEqual(transform[0,3],0.)
        self.assertEqual(ref_transform[0,3],2.)

    def test_legacy_frames_keep_their_original_shared_geometry(self):
        frame=dict(camera_to_ned=np.eye(4),intrinsics=np.array([135.,135.,160.,120.]))
        transform,k,ref_transform,ref_k=frame_geometry(frame)
        np.testing.assert_array_equal(transform,ref_transform)
        self.assertEqual(k,ref_k)

    def test_preparation_preserves_one_common_gt_and_native_pixel_pairing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);capture=root/"native";legacy=root/"trajectory"
            capture.mkdir();legacy.mkdir()
            k=[540.,0.,640.,0.,540.,480.,0.,0.,1.]
            rgb=np.zeros((960,1280,3),dtype=np.uint8);rgb[:,:,0]=255;rgb[:,4]=[0,255,0]
            depth=np.full((960,1280),10.,dtype=np.float32);depth[:,4]=2.
            np.savez_compressed(capture/"pair.npz",rgb_rgb=rgb,reference_depth_m=depth,camera_to_world=np.eye(4))
            def save(path,payload):path.write_text(json.dumps(payload))
            save(capture/"summary.json",dict(completed=True,render_resources_complete=True))
            save(capture/"protocol.json",dict(trajectory_run=str(legacy),trajectory_frames=["frames/frame_000001.npz"]))
            save(capture/"camera_info.json",dict(rgb=dict(k=k),depth=dict(k=k)))
            save(capture/"pairs.json",[dict(trajectory_frame_file="frames/frame_000001.npz",file="pair.npz",
                ros_rgb_payload_identical=True,ros_depth_payload_identical=True,
                stamp_ns=100,rgb_stamp_ns=100,depth_stamp_ns=100,
                rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                depth_sha256=hashlib.sha256(depth.tobytes()).hexdigest())])
            save(legacy/"manifest.json",dict(metadata={}))
            (legacy/"events.jsonl").write_text(json.dumps(dict(event="frame",file="frames/frame_000001.npz",timestamp_s=1.))+"\n")
            variants=prepare_capture(capture,legacy,root/"prepared")
            with np.load(variants["legacy320"]/"frames/frame_000001.npz") as old, np.load(variants["aligned640"]/"frames/frame_000001.npz") as new:
                np.testing.assert_array_equal(old["reference_depth_m"],new["reference_depth_m"])
                np.testing.assert_array_equal(old["reference_intrinsics"],new["reference_intrinsics"])
                self.assertEqual(old["rgb_bgr"].shape,(240,320,3))
                self.assertEqual(new["rgb_bgr"].shape,(480,640,3))
                self.assertEqual(new["reference_depth_m"][100,2],2.)
                np.testing.assert_array_equal(new["rgb_bgr"][100,2],[0,255,0])
                self.assertGreater(old["rgb_bgr"][100,1,2],old["rgb_bgr"][100,1,1])
