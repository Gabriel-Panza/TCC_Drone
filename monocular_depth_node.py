"""No ROS 2 que publica profundidade metrico-monocular a partir de um ONNX."""

import threading
import time

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from spatial_mapping.depth_model import MetricDepthOnnx


class MonocularMetricDepthNode(Node):
    """Converte imagens RGB em mapas 32FC1 com distancia em metros."""

    def __init__(self):
        super().__init__('monocular_metric_depth_node')
        image_topic = str(
            self.declare_parameter(
                'image_topic',
                '/world/baylands/model/x500_mono_cam_0/link/camera_link/sensor/camera/image',
            ).value
        )
        output_topic = str(
            self.declare_parameter('output_topic', '/monocular_depth').value
        )
        model_path = str(self.declare_parameter('model_path', '').value)
        if not model_path:
            raise ValueError('model_path deve apontar para um modelo metrico ONNX')

        mean = self.declare_parameter(
            'normalization_mean',
            [0.485, 0.456, 0.406],
        ).value
        std = self.declare_parameter(
            'normalization_std',
            [0.229, 0.224, 0.225],
        ).value
        self.inference = MetricDepthOnnx(
            model_path,
            input_width=int(self.declare_parameter('input_width', 518).value),
            input_height=int(self.declare_parameter('input_height', 518).value),
            mean=mean,
            std=std,
            output_representation=str(
                self.declare_parameter(
                    'output_representation',
                    'metric_depth',
                ).value
            ),
            output_scale=float(self.declare_parameter('output_scale', 1.0).value),
            output_shift=float(self.declare_parameter('output_shift', 0.0).value),
            min_depth_m=float(self.declare_parameter('min_depth_m', 0.1).value),
            max_depth_m=float(self.declare_parameter('max_depth_m', 50.0).value),
            use_cuda=bool(self.declare_parameter('use_cuda', False).value),
            backend=str(
                self.declare_parameter('inference_backend', 'opencv').value
            ),
            input_aspect_tolerance=float(
                self.declare_parameter('input_aspect_tolerance', 0.03).value
            ),
        )
        self.bridge = CvBridge()
        self.frames = 0
        self.frames_received = 0
        self.frames_replaced = 0
        self.last_log_s = time.perf_counter()
        self._pending_lock = threading.Lock()
        self._pending_image = None
        self._inference_event = threading.Event()
        self._stop_event = threading.Event()
        self.publisher = self.create_publisher(
            Image,
            output_topic,
            qos_profile_sensor_data,
        )
        self.subscription = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            1,
        )
        self._inference_worker = threading.Thread(
            target=self.inference_loop,
            name='monocular-latest-frame-inference',
            daemon=True,
        )
        self._inference_worker.start()
        self.get_logger().info(
            f'Profundidade monocular: {image_topic} -> {output_topic}.'
        )

    def image_callback(self, msg):
        """Mantem apenas o RGB mais recente para impedir backlog de inferencia."""

        with self._pending_lock:
            self.frames_received += 1
            if self._pending_image is not None:
                self.frames_replaced += 1
            self._pending_image = (msg, time.perf_counter())
        self._inference_event.set()

    def inference_loop(self):
        """Executa ONNX fora do callback ROS e descarta frames intermediarios."""

        while not self._stop_event.is_set():
            self._inference_event.wait(timeout=0.2)
            if self._stop_event.is_set():
                break
            with self._pending_lock:
                msg = self._pending_image
                self._pending_image = None
                self._inference_event.clear()
            if msg is None:
                continue
            image_msg, received_s = msg
            self.infer_and_publish(image_msg, received_s)

    def infer_and_publish(self, msg, received_s):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            started = time.perf_counter()
            depth_m = self.inference.predict(bgr)
            output = self.bridge.cv2_to_imgmsg(
                np.asarray(depth_m, dtype=np.float32),
                encoding='32FC1',
            )
            output.header = msg.header
            self.publisher.publish(output)
            self.frames += 1
            now = time.perf_counter()
            input_age_s = now - received_s
            if now - self.last_log_s >= 5.0:
                valid = depth_m[depth_m > 0.0]
                depth_summary = (
                    'sem pixels metricos validos'
                    if valid.size == 0
                    else (
                        f'min={float(np.min(valid)):.2f} m, '
                        f'mediana={float(np.median(valid)):.2f} m, '
                        f'max={float(np.max(valid)):.2f} m'
                    )
                )
                self.get_logger().info(
                    f'Depth publicado: frame={self.frames}, '
                    f'inferencia={(now - started) * 1000.0:.1f} ms, '
                    f'idade_entrada={input_age_s * 1000.0:.1f} ms, '
                    f'recebidos={self.frames_received}, '
                    f'substituidos={self.frames_replaced}, '
                    f'{depth_summary}.'
                )
                self.last_log_s = now
        except Exception as error:
            self.get_logger().error(f'Falha na inferencia de profundidade: {error}')

    def destroy_node(self):
        self._stop_event.set()
        self._inference_event.set()
        if self._inference_worker.is_alive():
            self._inference_worker.join(timeout=2.0)
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MonocularMetricDepthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
