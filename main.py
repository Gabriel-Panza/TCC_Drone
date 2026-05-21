import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from px4_msgs.msg import VehicleOdometry
import json
import numpy as np
import os
from datetime import datetime
from drone_controller import DroneOffboardNode

class DataLogger(Node):
    """
    Registra em memmap as variacoes da odometria e do controlador reativo.

    O logger continua ouvindo VehicleOdometry, mas as amostras novas passam a ser deltas
    entre mensagens consecutivas. Assim, a analise deixa de depender de posicao absoluta
    do mundo e fica alinhada com a ideia de deslocamento entre atualizacoes.

    Fontes:
    [ROS 2 Nodes] https://docs.ros.org/en/humble/Concepts/Basic/About-Nodes.html
    [PX4 VehicleOdometry] https://docs.px4.io/main/en/msg_docs/VehicleOdometry
    [NumPy memmap] https://numpy.org/doc/stable/reference/generated/numpy.memmap.html
    """

    def __init__(self, controller_node=None):
        """
        Inicializa os memmaps e a subscription de odometria do logger.

        Quando controller_node e informado, as metricas de evasao calculadas pelo fluxo
        optico sao lidas no mesmo instante da odometria, mas sempre salvas como variacoes
        em relacao a leitura anterior.

        Fontes:
        [ROS 2 QoS] https://docs.ros.org/en/humble/Concepts/Intermediate/About-Quality-of-Service-Settings.html
        [Python datetime] https://docs.python.org/3/library/datetime.html
        """

        super().__init__('data_logger')

        self.controller_node = controller_node
        self.log_dir = os.path.expanduser('~/TCC_Drone/logs')
        os.makedirs(self.log_dir, exist_ok=True)
        agora = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.log_dir, f'voo_teste_{agora}')
        os.makedirs(self.run_dir, exist_ok=True)
        self.manifest_path = os.path.join(self.run_dir, 'manifest.json')
        self.capacity = max(16, int(self.declare_parameter('flight_memmap_capacity', 20000).value))
        self.samples_saved = 0
        self.prev_state = None
        self.capacity_warned = False
        self.flight_intervals = np.lib.format.open_memmap(
            os.path.join(self.run_dir, 'flight_intervals.npy'),
            mode='w+',
            dtype=self.dtype_flight_interval(),
            shape=(self.capacity,),
        )
        self.atualizar_manifesto()

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.subscription = self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self.odometry_callback,
            qos_profile
        )

        self.get_logger().info(f'Data Logger iniciado com sucesso. A guardar deltas em: {self.run_dir}')

    def dtype_flight_interval(self):
        """Schema numerico dos deltas de odometria/comando."""

        return np.dtype([
            ('sample_id', 'i4'),
            ('dt_s', 'f4'),
            ('delta_x_m', 'f4'),
            ('delta_y_m', 'f4'),
            ('delta_z_m', 'f4'),
            ('delta_roll_speed_rad_s', 'f4'),
            ('delta_pitch_speed_rad_s', 'f4'),
            ('delta_yaw_speed_rad_s', 'f4'),
            ('delta_obstacle_risk', 'f4'),
            ('delta_avoid_lateral_body', 'f4'),
            ('delta_avoid_brake', 'f4'),
            ('delta_evasao_visual_ativa', 'i1'),
            ('pan_comp_delta_rad', 'f4'),
            ('pan_comp_source_code', 'i2'),
        ])

    def atualizar_manifesto(self):
        """Atualiza o manifesto do log em memmap."""

        manifesto = {
            'schema_version': 'flight_interval_memmap_v1',
            'description': (
                'Cada linha representa a variacao entre duas mensagens consecutivas '
                'de odometria, sem salvar pose absoluta.'
            ),
            'num_samples': int(self.samples_saved),
            'capacity': int(self.capacity),
            'arrays': {
                'flight_intervals': 'flight_intervals.npy',
            },
            'interval_fields': list(self.dtype_flight_interval().names),
            'pan_comp_source_codes': {
                'none': 0,
                'disabled': 1,
                'attitude': 2,
                'gyro': 3,
                'attitude+gyro': 4,
            },
        }
        with open(self.manifest_path, mode='w', encoding='utf-8') as fp:
            json.dump(manifesto, fp, indent=2)

    def codificar_pan_source(self, source):
        """Codifica a origem textual do pan em inteiro para manter o memmap numerico."""

        return {
            'none': 0,
            'disabled': 1,
            'attitude': 2,
            'gyro': 3,
            'attitude+gyro': 4,
        }.get(str(source), -1)

    def capturar_estado_odometria(self, msg):
        """Captura valores correntes apenas para calcular variacoes."""

        obstacle_risk = getattr(self.controller_node, 'obstacle_risk', 0.0)
        avoid_lateral_body = getattr(self.controller_node, 'avoid_lateral_body', 0.0)
        avoid_brake = getattr(self.controller_node, 'avoid_brake', 0.0)
        evasao_visual_ativa = int(bool(getattr(self.controller_node, 'evasao_visual_ativa', False)))
        pan_comp_delta_rad = getattr(self.controller_node, 'last_pan_delta_rad', 0.0)
        pan_comp_source = getattr(self.controller_node, 'last_pan_delta_source', 'none')

        return {
            'timestamp_s': msg.timestamp / 1_000_000.0,
            'x': float(msg.position[0]),
            'y': float(msg.position[1]),
            'z': float(msg.position[2]),
            'roll_speed': float(msg.angular_velocity[0]),
            'pitch_speed': float(msg.angular_velocity[1]),
            'yaw_speed': float(msg.angular_velocity[2]),
            'obstacle_risk': float(obstacle_risk),
            'avoid_lateral_body': float(avoid_lateral_body),
            'avoid_brake': float(avoid_brake),
            'evasao_visual_ativa': evasao_visual_ativa,
            'pan_comp_delta_rad': float(pan_comp_delta_rad),
            'pan_comp_source_code': self.codificar_pan_source(pan_comp_source),
        }

    def odometry_callback(self, msg):
        """
        Grava uma linha de variacoes de odometria com metricas reativas.

        A primeira mensagem vira referencia em memoria. A partir da segunda, o arquivo
        recebe apenas deltas entre a leitura anterior e a leitura atual.

        Fontes:
        [PX4 VehicleOdometry] https://docs.px4.io/main/en/msg_docs/VehicleOdometry
        [Python getattr] https://docs.python.org/3/library/functions.html#getattr
        """

        estado_atual = self.capturar_estado_odometria(msg)
        estado_anterior = self.prev_state
        self.prev_state = estado_atual

        if estado_anterior is None:
            return

        if self.samples_saved >= self.capacity:
            if not self.capacity_warned:
                self.get_logger().warning(
                    'Capacidade do memmap de voo esgotada. Aumente flight_memmap_capacity para runs maiores.'
                )
                self.capacity_warned = True
            return

        idx = self.samples_saved
        linha = np.zeros(1, dtype=self.dtype_flight_interval())
        linha['sample_id'][0] = idx + 1
        linha['dt_s'][0] = estado_atual['timestamp_s'] - estado_anterior['timestamp_s']
        linha['delta_x_m'][0] = estado_atual['x'] - estado_anterior['x']
        linha['delta_y_m'][0] = estado_atual['y'] - estado_anterior['y']
        linha['delta_z_m'][0] = estado_atual['z'] - estado_anterior['z']
        linha['delta_roll_speed_rad_s'][0] = estado_atual['roll_speed'] - estado_anterior['roll_speed']
        linha['delta_pitch_speed_rad_s'][0] = estado_atual['pitch_speed'] - estado_anterior['pitch_speed']
        linha['delta_yaw_speed_rad_s'][0] = estado_atual['yaw_speed'] - estado_anterior['yaw_speed']
        linha['delta_obstacle_risk'][0] = estado_atual['obstacle_risk'] - estado_anterior['obstacle_risk']
        linha['delta_avoid_lateral_body'][0] = estado_atual['avoid_lateral_body'] - estado_anterior['avoid_lateral_body']
        linha['delta_avoid_brake'][0] = estado_atual['avoid_brake'] - estado_anterior['avoid_brake']
        linha['delta_evasao_visual_ativa'][0] = (
            estado_atual['evasao_visual_ativa'] - estado_anterior['evasao_visual_ativa']
        )
        linha['pan_comp_delta_rad'][0] = estado_atual['pan_comp_delta_rad']
        linha['pan_comp_source_code'][0] = estado_atual['pan_comp_source_code']

        self.flight_intervals[idx] = linha[0]
        self.samples_saved += 1
        self.flight_intervals.flush()
        self.atualizar_manifesto()

    def destroy_node(self):
        """
        Descarrega o memmap antes de destruir o node ROS 2.

        O flush explicito garante que os ultimos buffers do arquivo sejam descarregados
        mesmo quando a execucao e encerrada por Ctrl+C.

        Fontes:
        [Python File Objects] https://docs.python.org/3/tutorial/inputoutput.html#reading-and-writing-files
        [ROS 2 Nodes] https://docs.ros.org/en/humble/Concepts/Basic/About-Nodes.html
        """

        if self.flight_intervals is not None:
            self.flight_intervals.flush()
        self.atualizar_manifesto()
        self.get_logger().info('Memmap de voo fechado e guardado com sucesso.')
        super().destroy_node()

def main(args=None):
    """
    Inicializa o controlador, o logger e o executor multithread do ROS 2.

    O logger recebe uma referencia ao controlador para registrar as metricas reais de evasao
    junto com a odometria. Assim, as novas runs de voo deixam de depender do proxy usado pelo
    dashboard para logs antigos.

    Fontes:
    [ROS 2 Executors] https://docs.ros.org/en/humble/Concepts/Intermediate/About-Executors.html
    [ROS 2 rclpy] https://docs.ros.org/en/humble/p/rclpy/
    """

    rclpy.init(args=args)
    
    controller_node = DroneOffboardNode()
    logger_node = DataLogger(controller_node)
    executor = MultiThreadedExecutor()
    executor.add_node(controller_node)
    executor.add_node(logger_node)
    
    try:
        controller_node.get_logger().info('Iniciando Controlador e Gravador de Dados simultaneamente...')
        executor.spin()
    except KeyboardInterrupt:
        controller_node.get_logger().info('Processo encerrado pelo usuário (Ctrl+C).')
        logger_node.get_logger().info('Finalizando a gravação do voo...')
    finally:
        executor.remove_node(controller_node)
        executor.remove_node(logger_node)
        controller_node.destroy_node()
        logger_node.destroy_node() 
        rclpy.shutdown()

if __name__ == '__main__':
    main()
