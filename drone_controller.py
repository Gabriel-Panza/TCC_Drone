import os
import time
import numpy as np
import math
import threading
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition
from sensor_msgs.msg import Image

class DroneOffboardNode(Node):
    # ======================================================================================
    # O script inicializa o nó do ROS 2. Na função pos_callback, ele lê a coordenada em que 
    # o drone "nasceu" (Marco Zero) e soma os seus waypoints relativos [5.0, 2.5, -2.5] a
    # essa origem para gerar alvos absolutos.
    # 
    # A Fonte: 
    # Repositório oficial PX4/px4_ros_com (Arquivo: offboard_control.py).
    # ======================================================================================
    def __init__(self):
        super().__init__('drone_offboard_node')

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE, 
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

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

        self.current_x = None
        self.current_y = None
        self.current_z = None
        self.current_yaw = 0.0
        
        self.start_x = None
        self.start_y = None
        self.start_z = None

        self.waypoints_relativos = [[5.0, 2.5, -2.5],
            [7.5, 5.0, -5.0],
            [10.0, 7.5, -7.5],
            [12.5, 10.0, -10.0],
            [10.0, 7.5, -7.5],
            [7.5, 5.0, -5.0],
            [5.0, 2.5, -2.5],
            [0.0, 0.0, -5.0]]
        
        self.lista_alvos_absolutos = []
        self.wp_atual_index = 0

        self.ciclos = 0
        self.voo_iniciado = False
        self.missao_concluida = False
        self.tempo_chegada = 0
        self.encerrando = False
        
        self.velocidade_maxima = 2.0 # Velocidade do vetor m/s
        self.raio_de_aceitacao = 0.5 # Distância em metros para trocar de waypoint

        self.timer = self.create_timer(0.04, self.timer_callback)

    def pos_callback(self, msg):
        """ Na primeira leitura, trava a origem e mapeia a rota de waypoints """
        if self.current_x is None:
            self.start_x = msg.x
            self.start_y = msg.y
            self.start_z = msg.z
            
            for wp in self.waypoints_relativos:
                self.lista_alvos_absolutos.append([
                    self.start_x + wp[0],
                    self.start_y + wp[1],
                    self.start_z + wp[2]])
            self.get_logger().info(f'Rota mapeada com {len(self.lista_alvos_absolutos)} waypoints. Decolando...')
            
        self.current_x = msg.x
        self.current_y = msg.y
        self.current_z = msg.z
        self.current_yaw = msg.heading


    # ==================================================================================
    # Existe um timer rodando a 25Hz (0.04s) que envia o modo de controle 
    # (OffboardControlMode) ininterruptamente. Somente após 50 ciclos (2 segundos), 
    # o script emite a ordem para armar (arm()) e decolar.
    # 
    # A Fonte: 
    # https://docs.px4.io/main/en/ros2/offboard_control
    # ==================================================================================
    def timer_callback(self):
        if self.current_x is None:
            return

        self.publish_offboard_control_mode()

        if self.ciclos == 50:
            self.arm()
            self.engage_offboard_mode()
            self.voo_iniciado = True

        if self.voo_iniciado:
            self.navegar_por_waypoints()
            
            if self.missao_concluida and not self.encerrando:
                tempo_pairando = (self.ciclos - self.tempo_chegada) * 0.04
                if tempo_pairando >= 2.0:
                    self.encerrando = True
                    import threading
                    threading.Thread(target=self.comando_exit).start()

        self.ciclos += 1

    # ==================================================================================
    # A função calcula a distância até o waypoint alvo. Se a distância for maior que 
    # a margem de corte (distancia_corte = 1.0 metro / ou 0.5 no código atual), ele converte 
    # a distância restante em um vetor de velocidade (v) normalizado. Quando a distância cai 
    # abaixo desse limite, o script não freia o drone; ele simplesmente muda o alvo para o 
    # próximo ponto da lista.
    # ==================================================================================
    def navegar_por_waypoints(self):
        alvo_atual = self.lista_alvos_absolutos[self.wp_atual_index]
        target_x, target_y, target_z = alvo_atual[0], alvo_atual[1], alvo_atual[2]
        
        pos_x = target_x - self.current_x
        pos_y = target_y - self.current_y
        pos_z = target_z - self.current_z
        distancia = math.sqrt(pos_x**2 + pos_y**2 + pos_z**2)
        
        vx, vy, vz = 0.0, 0.0, 0.0
        yaw_alvo = self.current_yaw
        
        distancia_corte = 0.3 if self.wp_atual_index == (len(self.lista_alvos_absolutos) - 1) else self.raio_de_aceitacao
        if distancia > distancia_corte:
            # Vetor de velocidade baseado no Calculo de Hipotenusa
            vx = (pos_x / distancia) * self.velocidade_maxima
            vy = (pos_y / distancia) * self.velocidade_maxima
            vz = (pos_z / distancia) * self.velocidade_maxima
        else:
            if self.wp_atual_index < len(self.lista_alvos_absolutos) - 1:
                self.wp_atual_index += 1
                self.get_logger().info(f'Waypoint {self.wp_atual_index} alcançado. Indo para o próximo...')
            else:
                if not self.missao_concluida:
                    self.get_logger().info('DESTINO FINAL ALCANÇADO! Pairando por 2 segundos...')
                    self.missao_concluida = True
                    self.tempo_chegada = self.ciclos
                    self.velocidade_maxima = 0.0
        
        if math.hypot(vx, vy) > 0.2:
            yaw_alvo = math.atan2(vy, vx)

        # ==================================================================================
        # O código instrui o PX4 a priorizar a Posição (msg.position = True, msg.velocity = False), 
        # mas no envio da trajetória (TrajectorySetpoint), ele preenche a posição alvo e envia o 
        # vetor calculado na variável msg.velocity. Além disso, ele injeta float('nan') nos eixos de aceleração e jerk.
        # ==================================================================================
        msg = TrajectorySetpoint()
        msg.position = [target_x, target_y, target_z] 
        msg.velocity = [vx, vy, vz]
        msg.acceleration = [float('nan'), float('nan'), float('nan')]
        msg.jerk = [float('nan'), float('nan'), float('nan')]
        msg.yaw = yaw_alvo
        msg.yawspeed = float('nan')
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def arm(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Rotores ligados.')

    def engage_offboard_mode(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info('Modo Feedforward (Posição + Velocidade) Ativado.')

    def force_disarm(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0, param2=21196.0)
        self.get_logger().info('CORTANDO MOTORES...')

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

    def comando_exit(self):
        self.get_logger().info("Encerrando a missão... Iniciando pouso!")
        
        # Altera o eixo Z do waypoint alvo final para o chão
        self.lista_alvos_absolutos[self.wp_atual_index][2] = 0.0
        self.velocidade_maxima = 4.0
        
        time.sleep(4)
        self.force_disarm()
        time.sleep(1)
        os._exit(0)

    def depth_callback(self, msg):
        # A lógica de desvio entrará aqui
        pass
