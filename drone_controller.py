import rclpy
import math
import numpy as np
import time
import os
import threading
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition
from sensor_msgs.msg import Image

class DroneOffboardNode(Node):
    def __init__(self):
        super().__init__('drone_offboard_node')

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE, 
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Publishers e Subscribers
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)

        self.local_pos_sub = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1', self.pos_callback, qos_profile)

        self.depth_sub = self.create_subscription(
            Image, '/depth_camera', self.depth_callback, qos_profile)

        # Posição atual do drone
        self.current_x = None
        self.current_y = None
        self.current_z = None
        self.current_yaw = 0.0
        
        # Ponto Final (Alvo)
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = 0.0

        self.ciclos = 0
        self.voo_iniciado = False
        self.destino_alcancado = False
        self.tempo_chegada = 0      
        self.encerrando = False

        self.velocidade_maxima = 2.0  # Vai voar a 2 m/s

        self.timer = self.create_timer(0.04, self.timer_callback)

    def pos_callback(self, msg):
        """ Atualiza a percepção espacial do drone. Na primeira leitura, trava o destino. """
        if self.current_x is None:
            self.get_logger().info('Sensores travados! Calculando rota (10m Frente, 5m Esquerda)...')
            
            self.target_x = msg.x + 20.0
            self.target_y = msg.y + 5.0
            self.target_z = msg.z - 7.5
            
        self.current_x = msg.x
        self.current_y = msg.y
        self.current_z = msg.z
        self.current_yaw = msg.heading

    def timer_callback(self):
        if self.current_x is None:
            return

        self.publish_offboard_control_mode()

        if self.ciclos == 50:
            self.arm()
            self.engage_offboard_mode()
            self.voo_iniciado = True

        if self.voo_iniciado:
            self.calcular_e_publicar_velocidade()
            
            if self.destino_alcancado and not self.encerrando:
                tempo_pairando = (self.ciclos - self.tempo_chegada) * 0.04
                if tempo_pairando >= 2.0:
                    self.encerrando = True
                    threading.Thread(target=self.comando_exit).start()

        self.ciclos += 1

    def force_disarm(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0, param2=21196.0)
        self.get_logger().info('CORTANDO MOTORES (Force Disarm Acionado!)...')
    
    def comando_exit(self):
        self.get_logger().info("Encerrando a missão... Descendo para pouso!") 
        
        self.target_z = self.current_z + 7.5
        self.velocidade_maxima = 2
        time.sleep(3)
        self.force_disarm()
        time.sleep(1)
        os._exit(0) 
    
    def calcular_e_publicar_velocidade(self):
        """ Calcula a diferença entre onde estou e onde quero ir, e cria o vetor velocidade """
        
        erro_x = self.target_x - self.current_x
        erro_y = self.target_y - self.current_y
        erro_z = self.target_z - self.current_z
        
        vx, vy, vz = 0.0, 0.0, 0.0
        
        distancia = math.sqrt(erro_x**2 + erro_y**2 + erro_z**2)
        if distancia > 0.3:
            vx = (erro_x / distancia) * self.velocidade_maxima
            vy = (erro_y / distancia) * self.velocidade_maxima
            vz = (erro_z / distancia) * self.velocidade_maxima
        else:
            if not self.destino_alcancado:
                self.get_logger().info('Destino alcançado! Pairando por 1 segundo...')
                self.destino_alcancado = True
                self.tempo_chegada = self.ciclos/2
                self.velocidade_maxima = 0.0
        
        # Gira a câmera/frente do drone (Yaw) para olhar para onde está voando
        yaw_alvo = self.current_yaw
        if math.hypot(vx, vy) > 0.2:
            yaw_alvo = math.atan2(vy, vx)

        # Desativa o controle de coordenadas injetando NaN
        msg = TrajectorySetpoint()
        msg.position = [float('nan'), float('nan'), float('nan')] 
        msg.velocity = [vx, vy, vz]
        
        msg.acceleration = [float('nan'), float('nan'), float('nan')]
        msg.jerk = [float('nan'), float('nan'), float('nan')]
        msg.yaw = yaw_alvo
        msg.yawspeed = float('nan')
        
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.position = False
        msg.velocity = True
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def arm(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Ligando rotores...')

    def engage_offboard_mode(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info('Decolando e navegando para as coordenadas alvo...')

    def publish_vehicle_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher.publish(msg)
    
    def depth_callback(self, msg):
        # A lógica de evasão por matriz matemática entrará aqui
        pass