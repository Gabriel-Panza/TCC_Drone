import tempfile
from pathlib import Path
import unittest
import cv2
import numpy as np

from spatial_mapping.depth_preprocessing import preprocess_metric_bgr
from spatial_mapping.depth_model import MetricDepthOnnx

try:
    import torch
    from scripts.finetune_depth_anything_baylands import SpatialFrameDataset,prediction_to_target_grid,masked_loss
except ImportError:
    torch=None


class SharedPreprocessingTest(unittest.TestCase):
    def test_inference_remains_identical_to_original_uint8_pipeline(self):
        rng=np.random.default_rng(42)
        bgr=rng.integers(0,256,(240,320,3),dtype=np.uint8)
        mean=np.array([.485,.456,.406],dtype=np.float32).reshape(1,1,3)
        std=np.array([.229,.224,.225],dtype=np.float32).reshape(1,1,3)
        rgb=cv2.cvtColor(cv2.resize(bgr,(686,518),interpolation=cv2.INTER_CUBIC),cv2.COLOR_BGR2RGB).astype(np.float32)/255.
        expected=np.ascontiguousarray(np.transpose((rgb-mean)/std,(2,0,1))[None],dtype=np.float32)
        np.testing.assert_array_equal(preprocess_metric_bgr(bgr,686,518,mean,std),expected)


@unittest.skipIf(torch is None,"Torch training environment required")
class AlignedTrainingTest(unittest.TestCase):
    def test_validation_input_is_identical_to_inference_and_gt_keeps_native_grid(self):
        rng=np.random.default_rng(1)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/"frames").mkdir()
            rgb=rng.integers(0,256,(480,640,3),dtype=np.uint8)
            gt=np.full((480,640),10.,dtype=np.float32);gt[:,13]=2.
            np.savez_compressed(root/"frames/frame_000001.npz",rgb_bgr=rgb,reference_depth_m=gt)
            dataset=SpatialFrameDataset([root],training=False,aligned_640=True)
            image,target=dataset[0]
            expected=preprocess_metric_bgr(rgb,686,518,[.485,.456,.406],[.229,.224,.225])[0]
            np.testing.assert_array_equal(image.numpy(),expected)
            np.testing.assert_array_equal(target.numpy(),gt)
            self.assertEqual(tuple(target.shape),(480,640))

    def test_postprocess_matches_opencv_and_has_finite_gradients(self):
        raw=np.linspace(.2,30,518*686,dtype=np.float32).reshape(518,686)
        raw[100:110,200:210]=100.
        raw[0,0]=-.5
        value=torch.tensor(raw[None],requires_grad=True)
        result=prediction_to_target_grid(value,(480,640),True)
        runtime=MetricDepthOnnx.__new__(MetricDepthOnnx)
        runtime.output_representation="metric_depth";runtime.output_scale=1.;runtime.output_shift=0.
        runtime.min_depth_m=.1;runtime.max_depth_m=80.
        expected=runtime.postprocess(raw,(640,480))
        np.testing.assert_allclose(result.detach().numpy()[0],expected,atol=.001,rtol=3e-5)
        loss=masked_loss(result,torch.full_like(result,8.))
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(torch.isfinite(value.grad).all().item())

    def test_invalid_sky_stays_outside_loss_without_nonfinite_gradients(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/"frames").mkdir()
            gt=np.full((480,640),10.,dtype=np.float32)
            gt[:100]=np.inf;gt[100:200]=np.nan
            np.savez_compressed(root/"frames/frame_000001.npz",rgb_bgr=np.zeros((480,640,3),dtype=np.uint8),reference_depth_m=gt)
            _,target=SpatialFrameDataset([root],training=False,aligned_640=True)[0]
            self.assertTrue(torch.isfinite(target).all().item())
            prediction=torch.full_like(target,7.,requires_grad=True)
            loss=masked_loss(prediction[None],target[None]);loss.backward()
            self.assertTrue(torch.isfinite(loss).item())
            self.assertTrue(torch.isfinite(prediction.grad).all().item())

    def test_legacy_dataset_keeps_original_target_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/"frames").mkdir()
            np.savez_compressed(root/"frames/frame_000001.npz",rgb_bgr=np.zeros((240,320,3),dtype=np.uint8),reference_depth_m=np.full((240,320),5.))
            _,target=SpatialFrameDataset([root],training=False)[0]
            self.assertEqual(tuple(target.shape),(518,686))
