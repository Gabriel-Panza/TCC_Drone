"""Shared uint8 RGB preparation for metric-depth training and inference."""
import cv2
import numpy as np


def preprocess_metric_bgr(image,width,height,mean,std,aspect_tolerance=.03):
    image=np.asarray(image)
    if image.ndim!=3 or image.shape[2]!=3:
        raise ValueError("bgr_image deve possuir formato (H, W, 3)")
    ratio=image.shape[1]/image.shape[0];model_ratio=width/height
    error=abs(ratio-model_ratio)/ratio
    if error>aspect_tolerance:
        raise ValueError("proporcao da imagem incompativel com a entrada ONNX: "
                         f"imagem={ratio:.4f}, modelo={model_ratio:.4f}, erro={error:.4f}")
    resized=cv2.resize(image,(width,height),interpolation=cv2.INTER_CUBIC)
    rgb=cv2.cvtColor(resized,cv2.COLOR_BGR2RGB).astype(np.float32)/255.
    normalized=(rgb-np.asarray(mean,dtype=np.float32).reshape(1,1,3))/np.asarray(std,dtype=np.float32).reshape(1,1,3)
    return np.ascontiguousarray(np.transpose(normalized,(2,0,1))[None],dtype=np.float32)
