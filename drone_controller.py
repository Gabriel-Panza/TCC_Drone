import os
import time
import numpy as np
import math
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

        # Publishers
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)

        # Subscriber
        self.local_pos_sub = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1', self.pos_callback, qos_profile)

        self.start_x = None
        self.start_y = None
        self.start_z = None
        self.current_yaw = 0.0
        
        self.cmd_x = 0.0
        self.cmd_y = 0.0
        self.cmd_z = 0.0
        
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = 0.0

        self.ciclos = 0
        self.voo_iniciado = False
        self.pronto_para_comando = False
        self.tempo_no_estado = 0.0
        
        # Velocidade do drone (ciclo de 50Hz)
        self.velocidade_movimento = 0.1 # 5m/s

        self.timer = self.create_timer(0.04, self.timer_callback)

        # Inscreve o drone para "enxergar" a câmera de profundidade
        self.depth_sub = self.create_subscription(
            Image, 
            '/depth_camera', 
            self.depth_callback, 
            qos_profile
        )

    def comando_pairar(self):
        self.get_logger().info('Ação: PAIRAR. Mantendo a posição estática.')
        self.resetar_tempo_espera()

    def comando_esquerda(self):
        self.get_logger().info('Ação: ESQUERDA. Virando a câmera e avançando...')
        self.current_yaw = self.start_yaw - (math.pi / 2.0)  # Gira -90 graus
        self.target_y -= 10.0
        self.resetar_tempo_espera()

    def comando_direita(self):
        self.get_logger().info('Ação: DIREITA. Virando a câmera e avançando...')
        self.current_yaw = self.start_yaw + (math.pi / 2.0)  # Gira +90 graus
        self.target_y += 10.0
        self.resetar_tempo_espera()

    def comando_frente(self):
        self.get_logger().info('Ação: FRENTE. Avançando...')
        self.current_yaw = self.start_yaw + 0.0  # Fica reto
        self.target_x += 10.0
        self.resetar_tempo_espera()

    def comando_tras(self):
        self.get_logger().info('Ação: TRAS. Dando meia volta e avançando...')
        self.current_yaw = self.start_yaw + math.pi  # Gira 180 graus
        self.target_x -= 10.0
        self.resetar_tempo_espera()

    def comando_exit(self):
        print("Encerrando o sistema de forma segura... Aguarde uns instantes...")
        self.target_z = 5
        self.timer_callback()
        time.sleep(3)
        self.force_disarm()
        time.sleep(1)
        os._exit(0)        
    def resetar_tempo_espera(self):
        """ Trava o terminal e inicia o cronômetro para os 5 segundos da nova ação """
        self.tempo_no_estado = 0.0
        self.pronto_para_comando = False

    def pos_callback(self, msg):
        """ Lê a posição inicial exata do drone na pista antes de ligar os motores """
        if self.start_x is None:
            self.start_x = msg.x
            self.start_y = msg.y
            self.start_z = msg.z
            self.start_yaw = msg.heading
            self.current_yaw = msg.heading
            
            self.cmd_x = self.target_x = self.start_x
            self.cmd_y = self.target_y = self.start_y
            self.cmd_z = self.target_z = self.start_z
            
            self.get_logger().info('GPS e Sensores travados! Preparando para decolar...')

    def timer_callback(self):
        """ Loop rodando a 50Hz para manter o drone no ar e calcular a física """
        if self.start_x is None:
            return

        self.publish_offboard_control_mode()
        self.publish_trajectory_setpoint()

        if self.ciclos == 50:
            self.arm()
            self.engage_offboard_mode()
            self.voo_iniciado = True
            self.target_z = self.start_z - 5

        if self.voo_iniciado:
            self.atualizar_movimento_suave()
            self.tempo_no_estado += self.velocidade_movimento
            
            if self.tempo_no_estado >= 2.5 and not self.pronto_para_comando:
                self.pronto_para_comando = True

        self.ciclos += 1

    def atualizar_movimento_suave(self):
        self.cmd_x = self.aproximar_valor(self.cmd_x, self.target_x, self.velocidade_movimento)
        self.cmd_y = self.aproximar_valor(self.cmd_y, self.target_y, self.velocidade_movimento)
        self.cmd_z = self.aproximar_valor(self.cmd_z, self.target_z, self.velocidade_movimento)

    def aproximar_valor(self, atual, alvo, passo_max):
        if atual < alvo:
            return min(atual + passo_max, alvo)
        elif atual > alvo:
            return max(atual - passo_max, alvo)
        return atual

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.position = True    
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def publish_trajectory_setpoint(self):
        msg = TrajectorySetpoint()
        msg.position = [self.cmd_x, self.cmd_y, self.cmd_z]
        msg.velocity = [float('nan'), float('nan'), float('nan')]
        msg.acceleration = [float('nan'), float('nan'), float('nan')]
        msg.jerk = [float('nan'), float('nan'), float('nan')]
        msg.yaw = self.current_yaw
        msg.yawspeed = float('nan')
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)

    def arm(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Ligando rotores...')

    def force_disarm(self):
        # param1 = 0.0 (Desarmar)
        # param2 = 21196.0 (Flag de força bruta do PX4 para ignorar a verificação de pouso)
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0, param2=21196.0)
        self.get_logger().info('CORTANDO MOTORES (Force Disarm Acionado!)...')
    
    def engage_offboard_mode(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info('Decolando para pairar a 5m...')

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
        # Logica de detecção de obstaculos perto do drone
        return