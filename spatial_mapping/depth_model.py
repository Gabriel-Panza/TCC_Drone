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
        backend="opencv",
        input_aspect_tolerance=0.03,
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
        self.backend = str(backend).strip().lower()
        self.input_aspect_tolerance = float(input_aspect_tolerance)
        if self.input_aspect_tolerance < 0:
            raise ValueError("input_aspect_tolerance nao pode ser negativo")
        self.net = None
        self.session = None
        self.input_name = None
        if self.backend == "opencv":
            self.net = cv2.dnn.readNetFromONNX(str(path))
            if use_cuda:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        elif self.backend == "onnxruntime":
            try:
                import onnxruntime as ort
            except ImportError as error:
                raise RuntimeError(
                    "backend onnxruntime solicitado, mas onnxruntime nao esta instalado"
                ) from error
            available = ort.get_available_providers()
            provider = "CUDAExecutionProvider" if use_cuda else "CPUExecutionProvider"
            if provider not in available:
                raise RuntimeError(
                    f"provider {provider} indisponivel; encontrados: {available}"
                )
            self.session = ort.InferenceSession(
                str(path),
                providers=[provider],
            )
            model_input = self.session.get_inputs()[0]
            self.input_name = model_input.name
            expected = [1, 3, self.input_height, self.input_width]
            if list(model_input.shape) != expected:
                raise ValueError(
                    f"entrada ONNX {model_input.shape} difere da configurada {expected}"
                )
        else:
            raise ValueError("backend deve ser opencv ou onnxruntime")

    def preprocess(self, bgr_image):
        """Aplica o resize e a normalizacao usados na exportacao do modelo."""

        image = np.asarray(bgr_image)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("bgr_image deve possuir formato (H, W, 3)")
        image_ratio = image.shape[1] / image.shape[0]
        model_ratio = self.input_width / self.input_height
        relative_error = abs(image_ratio - model_ratio) / image_ratio
        if relative_error > self.input_aspect_tolerance:
            raise ValueError(
                "proporcao da imagem incompativel com a entrada ONNX: "
                f"imagem={image_ratio:.4f}, modelo={model_ratio:.4f}, "
                f"erro={relative_error:.4f}"
            )
        resized = cv2.resize(
            image,
            (self.input_width, self.input_height),
            interpolation=cv2.INTER_CUBIC,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - self.mean) / self.std
        return np.ascontiguousarray(
            np.transpose(normalized, (2, 0, 1))[None, ...],
            dtype=np.float32,
        )

    def forward(self, blob):
        """Executa somente o grafo ONNX no backend configurado."""

        if self.backend == "onnxruntime":
            return self.session.run(None, {self.input_name: blob})[0]
        self.net.setInput(blob)
        return self.net.forward()

    def predict(self, bgr_image):
        """Executa o modelo e retorna profundidade float32 em metros."""

        image = np.asarray(bgr_image)
        original_size = (image.shape[1], image.shape[0])
        blob = self.preprocess(image)
        raw = self.forward(blob)
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
