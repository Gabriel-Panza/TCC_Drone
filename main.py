import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from px4_msgs.msg import VehicleOdometry
import csv
import os
from datetime import datetime
from drone_controller import DroneOffboardNode

class DataLogger(Node):
    """
    Registra em CSV a odometria e as metricas internas do controlador reativo.

    O logger continua ouvindo VehicleOdometry para manter a mesma base temporal dos logs
    antigos, mas tambem recebe uma referencia opcional ao DroneOffboardNode para salvar
    obstacle_risk, avoid_lateral_body e avoid_brake. Esses tres campos removem a necessidade
    de inferir a evasao por proxy no dashboard quando uma run nova for coletada.

    Fontes:
    [ROS 2 Nodes] https://docs.ros.org/en/humble/Concepts/Basic/About-Nodes.html
    [PX4 VehicleOdometry] https://docs.px4.io/main/en/msg_docs/VehicleOdometry
    [Python csv] https://docs.python.org/3/library/csv.html
    """

    def __init__(self, controller_node=None):
        """
        Inicializa o arquivo CSV e a subscription de odometria do logger.

        Quando controller_node e informado, as metricas de evasao calculadas pelo fluxo
        optico sao amostradas no mesmo instante em que a odometria e gravada. O parametro e
        opcional para preservar compatibilidade com execucoes de teste do logger isolado.

        Fontes:
        [ROS 2 QoS] https://docs.ros.org/en/humble/Concepts/Intermediate/About-Quality-of-Service-Settings.html
        [Python datetime] https://docs.python.org/3/library/datetime.html
        """

        super().__init__('data_logger')

        self.controller_node = controller_node
        self.log_dir = os.path.expanduser('~/TCC_Drone/logs')
        os.makedirs(self.log_dir, exist_ok=True)
        agora = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename = os.path.join(self.log_dir, f'voo_teste_{agora}.csv')
        self.file = open(self.csv_filename, mode='w', newline='')
        self.csv_writer = csv.writer(self.file)
        self.csv_writer.writerow([
            'timestamp',
            'x',
            'y',
            'z',
            'roll_speed',
            'pitch_speed',
            'yaw_speed',
            'obstacle_risk',
            'avoid_lateral_body',
            'avoid_brake',
            'evasao_visual_ativa',
            'pan_comp_delta_rad',
            'pan_comp_source'
        ])

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

        self.get_logger().info(f'Data Logger iniciado com sucesso. A guardar dados em: {self.csv_filename}')

    def odometry_callback(self, msg):
        """
        Grava uma linha de odometria com as metricas reativas sincronizadas.

        A odometria vem diretamente do PX4. As metricas obstacle_risk, avoid_lateral_body e
        avoid_brake sao lidas do controlador no mesmo processo ROS 2, permitindo que o
        dashboard avalie a evasao visual real das runs novas. Se o logger estiver sem
        controller_node, os campos sao preenchidos com valores neutros.

        Fontes:
        [PX4 VehicleOdometry] https://docs.px4.io/main/en/msg_docs/VehicleOdometry
        [Python getattr] https://docs.python.org/3/library/functions.html#getattr
        """

        timestamp_sec = msg.timestamp / 1_000_000.0

        x = msg.position[0]
        y = msg.position[1]
        z = msg.position[2]
        
        roll_speed = msg.angular_velocity[0]
        pitch_speed = msg.angular_velocity[1]
        yaw_speed = msg.angular_velocity[2]

        obstacle_risk = getattr(self.controller_node, 'obstacle_risk', 0.0)
        avoid_lateral_body = getattr(self.controller_node, 'avoid_lateral_body', 0.0)
        avoid_brake = getattr(self.controller_node, 'avoid_brake', 0.0)
        evasao_visual_ativa = int(bool(getattr(self.controller_node, 'evasao_visual_ativa', False)))
        pan_comp_delta_rad = getattr(self.controller_node, 'last_pan_delta_rad', 0.0)
        pan_comp_source = getattr(self.controller_node, 'last_pan_delta_source', 'none')

        self.csv_writer.writerow([
            timestamp_sec,
            x,
            y,
            z,
            roll_speed,
            pitch_speed,
            yaw_speed,
            obstacle_risk,
            avoid_lateral_body,
            avoid_brake,
            evasao_visual_ativa,
            pan_comp_delta_rad,
            pan_comp_source
        ])

    def destroy_node(self):
        """
        Fecha o arquivo CSV antes de destruir o node ROS 2.

        O fechamento explicito garante que os ultimos buffers do arquivo sejam descarregados
        mesmo quando a execucao e encerrada por Ctrl+C.

        Fontes:
        [Python File Objects] https://docs.python.org/3/tutorial/inputoutput.html#reading-and-writing-files
        [ROS 2 Nodes] https://docs.ros.org/en/humble/Concepts/Basic/About-Nodes.html
        """

        self.file.close()
        self.get_logger().info('Ficheiro CSV fechado e guardado com sucesso.')
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
