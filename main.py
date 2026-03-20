import rclpy
from drone_controller import DroneOffboardNode
from interface_terminal import TerminalInterface

def main(args=None):
    rclpy.init(args=args)
    
    node = DroneOffboardNode()
    
    cli_thread = TerminalInterface(node)
    cli_thread.start()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()