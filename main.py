import rclpy
from drone_controller import DroneOffboardNode

def main(args=None):
    rclpy.init(args=args)
    node = DroneOffboardNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Processo encerrado pelo usuário (Ctrl+C).')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()