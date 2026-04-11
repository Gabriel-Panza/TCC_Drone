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
    def __init__(self):
        super().__init__('data_logger')

        self.log_dir = os.path.expanduser('~/TCC_Drone/logs')
        os.makedirs(self.log_dir, exist_ok=True)
        agora = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename = os.path.join(self.log_dir, f'voo_teste_{agora}.csv')
        self.file = open(self.csv_filename, mode='w', newline='')
        self.csv_writer = csv.writer(self.file)
        self.csv_writer.writerow(['timestamp', 'x', 'y', 'z', 'roll_speed', 'pitch_speed', 'yaw_speed'])

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
        timestamp_sec = msg.timestamp / 1_000_000.0

        x = msg.position[0]
        y = msg.position[1]
        z = msg.position[2]
        
        roll_speed = msg.angular_velocity[0]
        pitch_speed = msg.angular_velocity[1]
        yaw_speed = msg.angular_velocity[2]

        self.csv_writer.writerow([timestamp_sec, x, y, z, roll_speed, pitch_speed, yaw_speed])

    def destroy_node(self):
        self.file.close()
        self.get_logger().info('Ficheiro CSV fechado e guardado com sucesso.')
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    controller_node = DroneOffboardNode()
    logger_node = DataLogger()
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