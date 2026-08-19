"""Inferencia de profundidade metrico-monocular com um modelo ONNX externo."""

from pathlib import Path

import cv2
import numpy as np


class MetricDepthOnnx:
    """Adapta modelos ONNX que produzem profundidade ou profundidade inversa."""

    def __init__(
        self,
        model_path,
        *,
        input_width=518,
        input_height=518,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        output_representation="metric_depth",
        output_scale=1.0,
        output_shift=0.0,
        min_depth_m=0.1,
        max_depth_m=50.0,
        use_cuda=False,
    ):
        path = Path(model_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"modelo ONNX nao encontrado: {path}")
        if output_representation not in {"metric_depth", "inverse_depth"}:
            raise ValueError(
                "output_representation deve ser metric_depth ou inverse_depth"
            )
        self.input_width = int(input_width)
        self.input_height = int(input_height)
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
        self.output_representation = output_representation
        self.output_scale = float(output_scale)
        self.output_shift = float(output_shift)
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self.net = cv2.dnn.readNetFromONNX(str(path))
        if use_cuda:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

    def predict(self, bgr_image):
        """Executa o modelo e retorna profundidade float32 em metros."""

        image = np.asarray(bgr_image)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("bgr_image deve possuir formato (H, W, 3)")
        original_size = (image.shape[1], image.shape[0])
        resized = cv2.resize(
            image,
            (self.input_width, self.input_height),
            interpolation=cv2.INTER_CUBIC,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - self.mean) / self.std
        blob = np.transpose(normalized, (2, 0, 1))[None, ...]
        self.net.setInput(np.ascontiguousarray(blob, dtype=np.float32))
        raw = self.net.forward()
        depth = self.postprocess(raw, original_size)
        return depth

    def postprocess(self, raw_output, output_size):
        """Converte a saida configurada em metros e restaura a resolucao RGB."""

        raw = np.asarray(raw_output, dtype=np.float32)
        while raw.ndim > 2 and raw.shape[0] == 1:
            raw = raw[0]
        if raw.ndim == 0:
            raw = raw.reshape(1, 1)
        elif raw.ndim == 1 and raw.size == 1:
            raw = raw.reshape(1, 1)
        if raw.ndim != 2:
            raise ValueError(f"saida ONNX inesperada: {raw.shape}")
        converted = raw * self.output_scale + self.output_shift
        if self.output_representation == "inverse_depth":
            converted = 1.0 / np.maximum(converted, 1e-6)
        converted[~np.isfinite(converted)] = 0.0
        converted = np.clip(converted, 0.0, self.max_depth_m)
        converted[converted < self.min_depth_m] = 0.0
        return cv2.resize(converted, output_size, interpolation=cv2.INTER_CUBIC)
