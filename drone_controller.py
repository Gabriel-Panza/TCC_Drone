import os
import time
import csv
import numpy as np
import math
import cv2
from pathlib import Path
from datetime import datetime
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleAttitude
from sensor_msgs.msg import Image

try:
    from px4_msgs.msg import SensorCombined
except ImportError:
    SensorCombined = None

class DroneOffboardNode(Node):
    """
    ======================================================================================
    Inicializa o nó principal de controle autônomo em ROS 2, criando os publishers,
    subscribers, parâmetros de missão, matriz intrínseca da câmera e variáveis de estado.
    
    O nó publica mensagens de controle Offboard para o PX4, envia setpoints de trajetória
    pelo tópico /fmu/in/trajectory_setpoint, envia comandos de veículo pelo tópico
    /fmu/in/vehicle_command e recebe posição local, atitude e imagem da câmera simulada.
    A imagem monocular pode ser compensada por IMU/atitude para reduzir tilt, roll e pan
    antes do fluxo optico. Opcionalmente, também recebe o mapa de profundidade renderizado
    pelo Gazebo apenas como ground truth sintético para treino/validação, sem usá-lo como
    entrada da evasão visual.
    
    Também são configurados parâmetros de voo adaptativo, como velocidade máxima,
    raio de aceitação reduzido, zona de frenagem por curvatura, aceleração lateral máxima
    e suavização de velocidade/yaw. Esses parâmetros foram adicionados para permitir
    curvas mais estreitas sem aumentar excessivamente o raio de aceitação, preservando
    a proposta de navegação em ambientes complexos, como florestas.
    
    Fontes:
    [PX4 Offboard ROS 2] https://docs.px4.io/main/en/ros2/offboard_control
    [PX4 Offboard Mode] https://docs.px4.io/main/en/flight_modes/offboard
    [ROS 2 QoS] https://docs.ros.org/en/humble/Concepts/Intermediate/About-Quality-of-Service-Settings.html
    [cv_bridge] https://docs.ros.org/en/jade/api/cv_bridge/html/python/
    [Gazebo DepthCamera] https://gazebosim.org/api/rendering/7/classgz_1_1rendering_1_1DepthCamera.html
    [Artigo - Tarrio2015] https://doi.org/10.1109/iccv.2015.87
    ======================================================================================
    """
    
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

        # Inicializa a ponte de conversão ROS -> OpenCV
        self.bridge = CvBridge()

        self.depth_gt_topic = str(self.declare_parameter('ground_truth_depth_topic', '').value)
        self.depth_gt_max_age_s = float(self.declare_parameter('ground_truth_depth_max_age_s', 0.08).value)
        self.save_ground_truth_dataset = bool(self.declare_parameter('save_ground_truth_dataset', False).value)
        self.ground_truth_dataset_dir = str(
            self.declare_parameter(
                'ground_truth_dataset_dir',
                os.path.expanduser('~/TCC_Drone/datasets/depth_ground_truth')
            ).value
        )
        self.ground_truth_save_every_n = max(
            1,
            int(self.declare_parameter('ground_truth_save_every_n', 5).value)
        )
        self.use_imu_raw = bool(self.declare_parameter('use_imu_raw', True).value)
        self.imu_raw_topic = str(self.declare_parameter('imu_raw_topic', '/fmu/out/sensor_combined').value)
        self.compensacao_imu_ativa = bool(self.declare_parameter('compensacao_imu_ativa', True).value)
        self.compensar_tilt_roll = bool(self.declare_parameter('compensar_tilt_roll', True).value)
        self.compensar_pan_yaw = bool(self.declare_parameter('compensar_pan_yaw', True).value)
        self.stabilization_zoom = float(self.declare_parameter('stabilization_zoom', 1.25).value)
        self.stabilization_output_width = int(self.declare_parameter('stabilization_output_width', 640).value)
        self.stabilization_output_height = int(self.declare_parameter('stabilization_output_height', 480).value)
        self.pan_gyro_weight = float(self.declare_parameter('pan_gyro_weight', 0.35).value)
        self.pan_gyro_weight = max(0.0, min(1.0, self.pan_gyro_weight))
        self.pan_yaw_gain = float(self.declare_parameter('pan_yaw_gain', 1.0).value)
        self.pan_max_delta_rad = float(
            self.declare_parameter('pan_max_delta_rad', math.radians(12.0)).value
        )

        self.latest_depth_gt = None
        self.latest_depth_gt_stamp_s = None
        self.latest_depth_gt_encoding = ''
        self.depth_gt_frames = 0
        self.rgb_frames_seen = 0
        self.depth_pairs_saved = 0
        self.depth_gt_csv_file = None
        self.depth_gt_csv_writer = None
        self.depth_gt_rgb_dir = None
        self.depth_gt_depth_dir = None
        self.depth_gt_shape_warned = False

        self.current_gyro_rad_s = np.zeros(3, dtype=float)
        self.current_accel_m_s2 = np.zeros(3, dtype=float)
        self.last_imu_timestamp_s = None
        self.prev_stabilization_stamp_s = None
        self.prev_stabilization_yaw = None
        self.last_pan_delta_rad = 0.0
        self.last_pan_delta_source = 'none'

        self.camera_sub = self.create_subscription(
            Image, 
            '/world/baylands/model/x500_mono_cam_0/link/camera_link/sensor/camera/image',
            self.image_callback, 
            qos_profile_sensor_data)

        if self.depth_gt_topic:
            self.depth_gt_sub = self.create_subscription(
                Image,
                self.depth_gt_topic,
                self.depth_ground_truth_callback,
                qos_profile_sensor_data)
            self.get_logger().info(
                f'Ground truth de profundidade habilitado em {self.depth_gt_topic}. '
                'A navegacao continua usando somente a camera monocular RGB.'
            )
        else:
            self.depth_gt_sub = None

        if self.use_imu_raw and SensorCombined is not None:
            self.imu_raw_sub = self.create_subscription(
                SensorCombined,
                self.imu_raw_topic,
                self.imu_raw_callback,
                qos_profile_sensor_data)
            self.get_logger().info(f'IMU bruto habilitado em {self.imu_raw_topic}.')
        else:
            self.imu_raw_sub = None
        
        # --- MATRIZ INTRÍNSECA DA CÂMERA (K) ---
        fov_rad = 1.74
        focal_length = 1280.0 / (2.0 * math.tan(fov_rad / 2.0)) # (f = largura / (2 * tan(FOV/2)))

        # Matriz para a resolução de 1280x960
        self.K = np.array([
            [focal_length, 0, 640.0], # 640 é o centro X (1280/2)
            [0, focal_length, 480.0], # 480 é o centro Y (960/2)
            [0, 0, 1]
        ])

        self.attitude_sub = self.create_subscription(
            VehicleAttitude, 
            '/fmu/out/vehicle_attitude', 
            self.attitude_callback, 
            qos_profile)

        self.current_x = None
        self.current_y = None
        self.current_z = None
        self.current_roll = 0.0
        self.current_pitch = 0.0
        self.current_yaw = 0.0
        self.current_yaw_attitude = 0.0
        self.smooth_yaw = 0.0
        self.smooth_vx = 0.0
        self.smooth_vy = 0.0
        self.velocity_smooth_alpha = 0.3
        self.yaw_smooth_alpha = 0.6

        self.prev_gray_avoidance = None
        self.prev_points_avoidance = None
        self.obstacle_risk = 0.0
        self.avoid_lateral_body = 0.0
        self.avoid_brake = 0.0
        self.avoid_side_memory = 0.9
        self.avoidance_max_brake = 0.3
        self.raio_finalizacao = 2.0
        self.raio_desativa_evasao_final = 10.0
        self.evasao_visual_ativa = True
        self.max_lateral_acceleration = 7.5

        self.start_x = None
        self.start_y = None
        self.start_z = None

        self.waypoints_relativos = [
            [-25.0, 25.0, -1.75],
            [-50.0, 70.0, -1.75],
            [-25.0, 25.0, -1.75],
            [0.0, 0.0, -1.75]
        ]
        
        self.lista_alvos_absolutos = []
        self.wp_atual_index = 0

        self.ciclos = 0
        self.voo_iniciado = False
        self.missao_concluida = False
        self.encerrando = False
        
        self.velocidade_maxima = 12.0    # Velocidade do vetor m/s
        self.raio_de_aceitacao = 5.0     # Raio de aceitação para mudar de waypoint
        
        self.zona_frenagem_curva = 6.5
        self.angulo_curva_forte = math.radians(35)

        self.dt = 0.04  # (25Hz)
        self.timer = self.create_timer(self.dt, self.timer_callback)

    def pos_callback(self, msg):
        """
        ==================================================================================
        Recebe a posição local estimada pelo PX4 e atualiza o estado atual do drone.
        
        Na primeira leitura válida, a função fixa o ponto de partida como origem local da
        missão e converte a lista de waypoints relativos em waypoints absolutos. Isso evita
        depender de coordenadas fixas do mundo e permite que a mesma rota seja executada a
        partir do ponto em que o drone nasceu na simulação.
        
        O PX4 informa a posição local no referencial NED: x = Norte, y = Leste e z = Down
        (altitude negativa para cima). O campo heading é o yaw em radianos no plano local.
        
        Fontes:
        [PX4 VehicleLocalPosition] https://docs.px4.io/main/en/msg_docs/VehicleLocalPosition
        [PX4 Offboard Mode - NED] https://docs.px4.io/main/en/flight_modes/offboard
        ==================================================================================
        """

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

    def timer_callback(self):
        """
        ==================================================================================
        Existe um timer rodando a 25Hz (0.04s) que envia o modo de controle (OffboardControlMode). 
        Somente após 50 ciclos (2 segundos), o script emite a ordem para armar (arm()) e decolar.
        
        Depois que o voo é iniciado, a função chama navegar_por_waypoints(), responsável por
        gerar os setpoints de velocidade, posição vertical e yaw. Quando a missão termina,
        uma thread separada executa o procedimento de encerramento para não bloquear o timer.
        
        Fontes:
        [PX4 ROS 2 Offboard Control Example] https://docs.px4.io/main/en/ros2/offboard_control
        [PX4 OffboardControlMode] https://docs.px4.io/main/en/msg_docs/OffboardControlMode
        [ROS 2 Node Timers] https://docs.ros.org/en/humble/Concepts/Basic/About-Nodes.html
        ==================================================================================
        """
        
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
                self.encerrando = True
                import threading
                threading.Thread(target=self.comando_exit).start()

        self.ciclos += 1

    def calcular_angulo_curva_wp_atual_rad(self):
        """
        ==================================================================================
        Calcula o ângulo geométrico da curva formada por três waypoints consecutivos:
        waypoint anterior, waypoint atual e próximo waypoint.
        
        O cálculo utiliza produto escalar entre vetores 2D no plano horizontal (x, y):
        cos(theta) = (v1 . v2) / (|v1| |v2|). O resultado final é retornado em radianos.
        
        Essa função foi adicionada para detectar curvas fechadas antes da troca de waypoint.
        Quando o ângulo ultrapassa o limite configurado em angulo_curva_forte, o controlador
        reduz progressivamente a velocidade dentro da zona_frenagem_curva. Assim, o drone
        mantém raio de aceitação pequeno, mas evita entrar em curvas estreitas com velocidade
        incompatível com a aceleração lateral disponível.
        
        Fontes:
        [Python math.acos] https://docs.python.org/3/library/math.html#math.acos
        [PX4 Offboard Mode - setpoints em NED] https://docs.px4.io/main/en/flight_modes/offboard
        ==================================================================================
        """
        
        if self.wp_atual_index == 0 or self.wp_atual_index >= len(self.lista_alvos_absolutos) - 1:
            return 0.0

        prev_wp = self.lista_alvos_absolutos[self.wp_atual_index - 1]
        curr_wp = self.lista_alvos_absolutos[self.wp_atual_index]
        next_wp = self.lista_alvos_absolutos[self.wp_atual_index + 1]

        v1x = curr_wp[0] - prev_wp[0]
        v1y = curr_wp[1] - prev_wp[1]
        v2x = next_wp[0] - curr_wp[0]
        v2y = next_wp[1] - curr_wp[1]

        mag1 = math.sqrt(v1x**2 + v1y**2)
        mag2 = math.sqrt(v2x**2 + v2y**2)

        if mag1 == 0 or mag2 == 0:
            return 0.0

        dot = v1x * v2x + v1y * v2y
        cos_angle = dot / (mag1 * mag2)
        cos_angle = max(-1.0, min(1.0, cos_angle))

        return math.acos(cos_angle)

    def navegar_por_waypoints(self):
        """
        ==================================================================================
        Executa a lógica principal de navegação por waypoints.
        
        A função calcula a distância até o waypoint atual, define uma velocidade dinâmica,
        aplica frenagem progressiva em curvas fechadas, limita a aceleração lateral, suaviza
        o vetor de velocidade e ajusta o yaw com look-ahead. Em seguida, publica um
        TrajectorySetpoint para o PX4.
        
        A estratégia atual não aumenta o raio de aceitação para estabilizar o voo. Em vez
        disso, mantém raio reduzido e reduz a velocidade apenas quando detecta curva forte
        próxima ao waypoint. Isso preserva a proposta de navegação em ambientes estreitos,
        como corredores entre árvores, evitando que o drone corte caminho cedo demais.
        
        O TrajectorySetpoint usa position = [NaN, NaN, target_z], mantendo controle de altura
        pelo eixo z, e velocity = [vx, vy, NaN], usando velocidade horizontal como comando
        principal. No PX4, valores NaN indicam campos não comandados; valores não-NaN de
        velocidade podem atuar como feedforward ou setpoint conforme a combinação enviada.

        Os comandos laterais de evasão são aplicados no referencial do corpo do drone:
        avoid_lateral_body positivo desloca o drone para a direita.
        
        Fontes:
        [PX4 Offboard Mode - TrajectorySetpoint] https://docs.px4.io/main/en/flight_modes/offboard
        [PX4 ROS 2 Offboard Control] https://docs.px4.io/main/en/ros2/offboard_control
        [PX4 VehicleLocalPosition - NED] https://docs.px4.io/main/en/msg_docs/VehicleLocalPosition
        ==================================================================================
        """
        
        alvo_atual = self.lista_alvos_absolutos[self.wp_atual_index]
        target_x, target_y, target_z = alvo_atual[0], alvo_atual[1], alvo_atual[2]
        
        pos_x = target_x - self.current_x
        pos_y = target_y - self.current_y
        pos_z = target_z - self.current_z
        distancia = math.sqrt(pos_x**2 + pos_y**2 + pos_z**2)
        
        vx, vy = 0.0, 0.0
        if self.smooth_yaw is None or self.smooth_yaw == 0.0:
            self.smooth_yaw = self.current_yaw
        
        is_ultimo_wp = (self.wp_atual_index == len(self.lista_alvos_absolutos) - 1)
        distancia_corte = self.raio_finalizacao if is_ultimo_wp else self.raio_de_aceitacao

        # ---- VELOCIDADE ADAPTATIVA BASEADA EM CURVATURA ----
        velocidade_maxima_atual = self.velocidade_maxima

        if not is_ultimo_wp:
            angulo_curva = self.calcular_angulo_curva_wp_atual_rad()

            if angulo_curva > self.angulo_curva_forte and distancia < self.zona_frenagem_curva:
                velocidade_segura_curva = math.sqrt(
                    self.max_lateral_acceleration * max(self.raio_de_aceitacao, 1.0)
                )

                velocidade_segura_curva = max(
                    self.raio_de_aceitacao,
                    min(velocidade_segura_curva, 6.0)
                )

                t = (distancia - self.raio_de_aceitacao) / (
                    self.zona_frenagem_curva - self.raio_de_aceitacao
                )
                t = max(0.0, min(1.0, t))

                t = t * t * (3.0 - 2.0 * t)

                velocidade_maxima_atual = (
                    velocidade_segura_curva +
                    (self.velocidade_maxima - velocidade_segura_curva) * t
                )

        # ---- LÓGICA DE VELOCIDADE DINÂMICA PARA CADA WAYPOINT ----
        if distancia > distancia_corte:
            if is_ultimo_wp:
                dist_inicio_frenagem = velocidade_maxima_atual * 1.2
            else:
                dist_inicio_frenagem = velocidade_maxima_atual * 0.6
            velocidade_minima = velocidade_maxima_atual * 0.1
            
            if distancia > dist_inicio_frenagem:
                velocidade_dinamica = velocidade_maxima_atual
            else:
                proporcao = (distancia - distancia_corte) / (dist_inicio_frenagem - distancia_corte)
                proporcao = proporcao ** 2.0
                
                if is_ultimo_wp:
                    proporcao = proporcao ** 1.5 

                velocidade_dinamica = velocidade_minima + (velocidade_maxima_atual - velocidade_minima) * proporcao

            vx = (pos_x / distancia) * velocidade_dinamica
            vy = (pos_y / distancia) * velocidade_dinamica
            
        else:
            if not is_ultimo_wp:
                self.wp_atual_index += 1
                self.get_logger().info(f'Indo para o Waypoint {self.wp_atual_index}...')
            else:
                if not self.missao_concluida:
                    self.get_logger().info('MISSÃO FINALIZADA! Estabilizando e descendo...')
                    self.missao_concluida = True

        # ---- EVASAO REATIVA POR VISAO ----
        evasao_habilitada = (
            not self.missao_concluida and
            not (is_ultimo_wp and distancia <= self.raio_desativa_evasao_final)
        )
        self.evasao_visual_ativa = evasao_habilitada

        if not evasao_habilitada:
            self.obstacle_risk = 0.0
            self.avoid_lateral_body = 0.0
            self.avoid_brake = 0.0

        if evasao_habilitada and self.obstacle_risk > 0.07:
            brake_scale = max(0.7, 1.0 - self.avoid_brake)
            vx *= brake_scale
            vy *= brake_scale

            right_x = -math.sin(self.current_yaw)
            right_y = math.cos(self.current_yaw)
            vx += right_x * self.avoid_lateral_body
            vy += right_y * self.avoid_lateral_body

            velocidade_cmd = math.sqrt(vx**2 + vy**2)
            if velocidade_cmd > velocidade_maxima_atual:
                escala = velocidade_maxima_atual / velocidade_cmd
                vx *= escala
                vy *= escala

        # ---- LIMITAÇÃO DE ACELERAÇÃO LATERAL ----
        accel_x = (vx - self.smooth_vx) / (self.dt * 4)
        accel_y = (vy - self.smooth_vy) / (self.dt * 4)
        accel_lateral = math.sqrt(accel_x**2 + accel_y**2)
        if accel_lateral > self.max_lateral_acceleration:
            scale = self.max_lateral_acceleration / accel_lateral
            vx = self.smooth_vx + accel_x * scale * (self.dt * 4)
            vy = self.smooth_vy + accel_y * scale * (self.dt * 4)

        # ---- FILTRAGEM DE VELOCIDADE ----
        self.smooth_vx += self.velocity_smooth_alpha * (vx - self.smooth_vx)
        self.smooth_vy += self.velocity_smooth_alpha * (vy - self.smooth_vy)

        # ---- AJUSTE DE DIREÇÃO (YAW) COM LOOK-AHEAD ----
        if self.smooth_yaw is None or self.smooth_yaw == 0.0:
            self.smooth_yaw = self.current_yaw

        yaw_alvo = self.calcular_yaw_com_look_ahead(target_x, target_y)
        erro_yaw = math.atan2(math.sin(yaw_alvo - self.smooth_yaw), math.cos(yaw_alvo - self.smooth_yaw))
        yaw_gain = self.yaw_smooth_alpha * (0.75 if abs(erro_yaw) > 0.75 else 1.2)
        self.smooth_yaw += (erro_yaw * yaw_gain)

        msg = TrajectorySetpoint()
        msg.position = [float('nan'), float('nan'), target_z] 
        msg.velocity = [self.smooth_vx, self.smooth_vy, float('nan')]
        msg.acceleration = [float('nan'), float('nan'), float('nan')]
        msg.jerk = [float('nan'), float('nan'), float('nan')]
        msg.yaw = self.smooth_yaw
        msg.yawspeed = float('nan')
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)

    def calcular_yaw_com_look_ahead(self, target_x, target_y):
        """ 
        ==================================================================================
        Calcula o yaw desejado do drone com uma pequena antecipação para o próximo waypoint.
        
        Quando o drone está distante do waypoint atual, o yaw aponta para o alvo atual. Quando
        se aproxima do waypoint e ainda existe um próximo alvo, a função mistura gradualmente
        a direção do waypoint atual com a direção do próximo waypoint. Isso reduz mudanças
        instantâneas de orientação no momento da troca de waypoint.
        
        A correção angular usa atan2(sin(erro), cos(erro)) para normalizar o erro no intervalo
        [-pi, pi], evitando saltos bruscos quando o ângulo cruza a descontinuidade de -pi/pi.
        
        Fontes:
        [Python math.atan2] https://docs.python.org/3/library/math.html#math.atan2
        [PX4 VehicleLocalPosition - heading] https://docs.px4.io/main/en/msg_docs/VehicleLocalPosition
        ==================================================================================
        """
        
        distancia_atual = math.sqrt((target_x - self.current_x)**2 + (target_y - self.current_y)**2)
        
        if self.wp_atual_index < len(self.lista_alvos_absolutos) - 1 and distancia_atual < 4.0:
            next_wp = self.lista_alvos_absolutos[self.wp_atual_index + 1]
            yaw_next = math.atan2(next_wp[1] - self.current_y, next_wp[0] - self.current_x)
            blend_factor = max(0.0, (4.0 - distancia_atual) / 4.0)
            yaw_base = math.atan2(target_y - self.current_y, target_x - self.current_x)
            erro = math.atan2(math.sin(yaw_next - yaw_base), math.cos(yaw_next - yaw_base))
            return yaw_base + erro * blend_factor
        else:
            return math.atan2(target_y - self.current_y, target_x - self.current_x)

    def publish_offboard_control_mode(self):
        """
        Publica o modo de controle Offboard usado pelo PX4 nesta missão.

        A combinação atual habilita setpoints de posição e velocidade, mantendo aceleração,
        atitude e body rate desabilitados.

        Fontes:
        [PX4 OffboardControlMode] https://docs.px4.io/main/en/msg_docs/OffboardControlMode
        [PX4 Offboard Mode] https://docs.px4.io/main/en/flight_modes/offboard
        """

        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = True
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def arm(self):
        """
        Envia o comando MAVLink/PX4 para armar o drone.

        Fontes:
        [PX4 VehicleCommand] https://docs.px4.io/main/en/msg_docs/VehicleCommand
        [MAVLink MAV_CMD_COMPONENT_ARM_DISARM] https://mavlink.io/en/messages/common.html#MAV_CMD_COMPONENT_ARM_DISARM
        """

        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Rotores ligados.')

    def engage_offboard_mode(self):
        """
        Solicita ao PX4 a entrada no modo Offboard antes do envio contínuo dos setpoints.

        Fontes:
        [PX4 Offboard Mode] https://docs.px4.io/main/en/flight_modes/offboard
        [MAVLink MAV_CMD_DO_SET_MODE] https://mavlink.io/en/messages/common.html#MAV_CMD_DO_SET_MODE
        """

        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info('Modo Feedforward (Posição + Velocidade) Ativado.')

    def force_disarm(self):
        """
        Envia o comando de desarme forçado usado no encerramento da missão.

        Fontes:
        [PX4 VehicleCommand] https://docs.px4.io/main/en/msg_docs/VehicleCommand
        [MAVLink MAV_CMD_COMPONENT_ARM_DISARM] https://mavlink.io/en/messages/common.html#MAV_CMD_COMPONENT_ARM_DISARM
        """

        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0, param2=21196.0)
        self.get_logger().info('CORTANDO MOTORES...')

    def publish_vehicle_command(self, command, param1=0.0, param2=0.0):
        """
        Monta e publica uma mensagem VehicleCommand para o PX4.

        Os campos de sistema/componente seguem o padrão do exemplo Offboard em ROS 2,
        com from_external=True para indicar origem externa ao autopiloto.

        Fontes:
        [PX4 VehicleCommand] https://docs.px4.io/main/en/msg_docs/VehicleCommand
        [PX4 ROS 2 Offboard Control] https://docs.px4.io/main/en/ros2/offboard_control
        """

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
        """
        Executa o encerramento da missão fora do timer principal.

        O waypoint final é ajustado para o nível do chão, aguarda-se uma breve estabilização
        e então é enviado o desarme forçado antes de finalizar o processo.

        Fontes:
        [PX4 VehicleCommand] https://docs.px4.io/main/en/msg_docs/VehicleCommand
        [Python threading] https://docs.python.org/3/library/threading.html
        """

        self.get_logger().info("Encerrando a missão em 3s... Iniciando pouso!")
        
        self.lista_alvos_absolutos[self.wp_atual_index][2] = 0.0
        
        time.sleep(3)
        self.force_disarm()
        time.sleep(1)
        os._exit(0)

    def image_timestamp_s(self, msg):
        """
        ==================================================================================
        Retorna o timestamp ROS de uma mensagem de imagem em segundos.

        A funcao usa preferencialmente o campo header.stamp preenchido pelo ROS/Gazebo
        Bridge. Quando esse campo nao esta disponivel, ou vem zerado, usa o relogio local
        do no como fallback para permitir sincronizacao aproximada em testes de bancada.

        Esse timestamp e usado para parear a imagem RGB monocular com o mapa de profundidade
        sintetico do Gazebo. O pareamento serve apenas para montar dataset e visualizacao de
        ground truth; a evasao reativa continua usando a imagem monocular estabilizada.

        Fontes:
        [sensor_msgs/Image] https://docs.ros2.org/latest/api/sensor_msgs/msg/Image.html
        [ROS 2 Clock] https://docs.ros.org/en/humble/Concepts/Intermediate/About-Time.html
        ==================================================================================
        """

        stamp = getattr(getattr(msg, 'header', None), 'stamp', None)
        if stamp is not None and (stamp.sec != 0 or stamp.nanosec != 0):
            return stamp.sec + stamp.nanosec * 1e-9
        return self.get_clock().now().nanoseconds * 1e-9

    def converter_depth_gt_para_metros(self, msg):
        """
        ==================================================================================
        Converte a imagem de profundidade do Gazebo para matriz float32 em metros.

        O Gazebo/bridge pode entregar o depth como float, normalmente ja em metros, ou como
        uint16, frequentemente em milimetros. A funcao normaliza esses formatos para uma
        matriz NumPy float32, remove valores nao finitos e preserva zeros como pixels sem
        profundidade valida.

        O resultado e tratado como ground truth sintetico do simulador. Ele pode ser salvo
        junto da imagem RGB para treino offline, mas nao e usado diretamente pela logica de
        navegacao ou pela evasao visual em tempo real.

        Fontes:
        [Gazebo DepthCamera] https://gazebosim.org/api/rendering/7/classgz_1_1rendering_1_1DepthCamera.html
        [cv_bridge] https://docs.ros.org/en/jade/api/cv_bridge/html/python/
        [NumPy astype] https://numpy.org/doc/stable/reference/generated/numpy.ndarray.astype.html
        ==================================================================================
        """

        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        depth = np.asarray(depth)

        if depth.ndim == 3:
            depth = depth[:, :, 0]

        if depth.dtype == np.uint16:
            depth_m = depth.astype(np.float32) / 1000.0
        else:
            depth_m = depth.astype(np.float32)

        depth_m[~np.isfinite(depth_m)] = 0.0
        return depth_m

    def depth_ground_truth_callback(self, msg):
        """
        ==================================================================================
        Recebe e armazena o ultimo mapa de profundidade sintetico publicado pelo Gazebo.

        A funcao converte a mensagem ROS Image para profundidade em metros, guarda o
        timestamp correspondente e registra estatisticas basicas no primeiro frame recebido.
        A sincronizacao efetiva com o RGB acontece em image_callback(), mantendo a imagem
        monocular como referencia principal de cada amostra do dataset.

        Importante: este callback nao injeta profundidade no controlador. O depth existe
        apenas como ground truth externo para validacao, treinamento supervisionado e
        depuracao visual do que o simulador renderizou.

        Fontes:
        [Gazebo DepthCameraSensor] https://gazebosim.org/api/sensors/7/classgz_1_1sensors_1_1DepthCameraSensor.html
        [ROS 2 QoS sensor data] https://docs.ros.org/en/humble/Concepts/Intermediate/About-Quality-of-Service-Settings.html
        [NumPy min] https://numpy.org/doc/stable/reference/generated/numpy.min.html
        ==================================================================================
        """

        try:
            self.latest_depth_gt = self.converter_depth_gt_para_metros(msg)
            self.latest_depth_gt_stamp_s = self.image_timestamp_s(msg)
            self.latest_depth_gt_encoding = msg.encoding
            self.depth_gt_frames += 1

            if self.depth_gt_frames == 1:
                valid = self.latest_depth_gt[self.latest_depth_gt > 0.0]
                if valid.size > 0:
                    self.get_logger().info(
                        f'Primeiro depth GT recebido: {self.latest_depth_gt.shape}, '
                        f'encoding={msg.encoding}, min={float(np.min(valid)):.2f}m, '
                        f'max={float(np.max(valid)):.2f}m.'
                    )
                else:
                    self.get_logger().info(
                        f'Primeiro depth GT recebido: {self.latest_depth_gt.shape}, '
                        f'encoding={msg.encoding}, sem pixels validos positivos.'
                    )
        except Exception as e:
            self.get_logger().error(f'Erro ao converter ground truth de profundidade: {e}')

    def imu_raw_callback(self, msg):
        """
        ==================================================================================
        Guarda as leituras brutas de giroscopio e acelerometro publicadas pelo PX4.

        A mensagem SensorCombined fornece velocidade angular em rad/s e aceleracao linear
        em m/s^2. O eixo z do giroscopio e usado como apoio de alta frequencia para estimar
        o pan/yaw entre frames; o acelerometro fica registrado no dataset porque a correcao
        de translacao da imagem depende da profundidade por pixel.

        A implementacao atual nao usa o acelerometro para alterar a navegacao. A compensacao
        visual combina atitude estimada por VehicleAttitude com o giroscopio, e a navegacao
        continua recebendo apenas o fluxo residual calculado na imagem monocular.

        Fontes:
        [PX4 SensorCombined] https://docs.px4.io/main/en/msg_docs/SensorCombined.html
        [PX4 VehicleAttitude] https://docs.px4.io/main/en/msg_docs/VehicleAttitude
        [Artigo - Tarrio2015] https://doi.org/10.1109/iccv.2015.87
        ==================================================================================
        """

        self.current_gyro_rad_s = np.array(getattr(msg, 'gyro_rad', [0.0, 0.0, 0.0]), dtype=float)
        self.current_accel_m_s2 = np.array(getattr(msg, 'accelerometer_m_s2', [0.0, 0.0, 0.0]), dtype=float)
        self.last_imu_timestamp_s = getattr(msg, 'timestamp', 0) / 1_000_000.0

    def obter_depth_gt_sincronizado(self, rgb_msg):
        """
        ==================================================================================
        Retorna o mapa de profundidade ground truth mais proximo do frame RGB atual.

        A funcao compara o timestamp da imagem monocular com o timestamp do ultimo depth
        recebido. Se a diferenca absoluta for maior que depth_gt_max_age_s, o depth e
        descartado para evitar salvar pares desalinhados temporalmente.

        O retorno inclui o depth em metros, o timestamp do RGB e a diferenca temporal entre
        RGB e depth. Quando nao existe par confiavel, retorna None no lugar do mapa de
        profundidade.

        Fontes:
        [ROS 2 Time] https://docs.ros.org/en/humble/Concepts/Intermediate/About-Time.html
        [sensor_msgs/Image] https://docs.ros2.org/latest/api/sensor_msgs/msg/Image.html
        ==================================================================================
        """

        if self.latest_depth_gt is None or self.latest_depth_gt_stamp_s is None:
            return None, None, None

        rgb_stamp_s = self.image_timestamp_s(rgb_msg)
        depth_age_s = abs(rgb_stamp_s - self.latest_depth_gt_stamp_s)

        if depth_age_s > self.depth_gt_max_age_s:
            return None, rgb_stamp_s, depth_age_s

        return self.latest_depth_gt, rgb_stamp_s, depth_age_s

    def preparar_dataset_ground_truth(self):
        """
        ==================================================================================
        Prepara a estrutura de pastas e metadados do dataset de ground truth.

        Cada execucao cria uma pasta run_<data_hora> dentro de ground_truth_dataset_dir,
        separando imagens RGB monoculares em rgb/, mapas de profundidade em depth_m/ e
        metadados em metadata.csv. O CSV armazena timestamps, caminhos dos arquivos,
        pose aproximada, atitude, IMU bruta, delta de pan compensado e estatisticas do depth.

        Essa estrutura foi pensada para treino offline: a entrada do modelo e a imagem
        monocular, enquanto o depth do Gazebo funciona como alvo supervisionado.

        Fontes:
        [Python pathlib] https://docs.python.org/3/library/pathlib.html
        [Python csv] https://docs.python.org/3/library/csv.html
        [NumPy save] https://numpy.org/doc/stable/reference/generated/numpy.save.html
        ==================================================================================
        """

        base_dir = Path(os.path.expanduser(self.ground_truth_dataset_dir))
        run_dir = base_dir / datetime.now().strftime('run_%Y%m%d_%H%M%S')
        self.depth_gt_rgb_dir = run_dir / 'rgb'
        self.depth_gt_depth_dir = run_dir / 'depth_m'
        self.depth_gt_rgb_dir.mkdir(parents=True, exist_ok=True)
        self.depth_gt_depth_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = run_dir / 'metadata.csv'
        self.depth_gt_csv_file = open(metadata_path, mode='w', newline='')
        self.depth_gt_csv_writer = csv.writer(self.depth_gt_csv_file)
        self.depth_gt_csv_writer.writerow([
            'sample_id', 'rgb_timestamp_s', 'depth_timestamp_s', 'depth_age_s',
            'rgb_path', 'depth_path',
            'x', 'y', 'z', 'roll', 'pitch', 'yaw',
            'gyro_x', 'gyro_y', 'gyro_z',
            'accel_x', 'accel_y', 'accel_z',
            'pan_comp_delta_rad', 'pan_comp_source',
            'depth_min_m', 'depth_mean_m', 'depth_max_m'
        ])
        self.get_logger().info(f'Dataset de depth GT sendo salvo em: {run_dir}')

    def salvar_par_ground_truth(self, rgb_bgr, depth_m, rgb_stamp_s, depth_age_s):
        """
        ==================================================================================
        Salva um par supervisionado formado por RGB monocular e depth ground truth.

        A funcao respeita save_ground_truth_dataset e ground_truth_save_every_n para evitar
        gravacao excessiva em disco. O RGB e salvo em PNG, o mapa de profundidade em metros
        e salvo como NPY float32, e os metadados da amostra sao adicionados ao CSV da
        execucao atual, incluindo o delta de pan/yaw aplicado na compensacao visual.

        O depth salvo nao e entrada do controlador. Ele representa o alvo sintetico que pode
        treinar ou validar uma rede monocular de profundidade, risco de colisao ou analise de
        fluxo compensado.

        Fontes:
        [OpenCV imwrite] https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html
        [NumPy save] https://numpy.org/doc/stable/reference/generated/numpy.save.html
        [Python csv] https://docs.python.org/3/library/csv.html
        ==================================================================================
        """

        if not self.save_ground_truth_dataset:
            return

        if self.rgb_frames_seen % self.ground_truth_save_every_n != 0:
            return

        if self.depth_gt_csv_writer is None:
            self.preparar_dataset_ground_truth()

        self.depth_pairs_saved += 1
        sample_id = f'{self.depth_pairs_saved:06d}'
        rgb_path = self.depth_gt_rgb_dir / f'{sample_id}.png'
        depth_path = self.depth_gt_depth_dir / f'{sample_id}.npy'

        cv2.imwrite(str(rgb_path), rgb_bgr)
        np.save(str(depth_path), depth_m.astype(np.float32))

        valid = depth_m[np.isfinite(depth_m) & (depth_m > 0.0)]
        if valid.size > 0:
            depth_min = float(np.min(valid))
            depth_mean = float(np.mean(valid))
            depth_max = float(np.max(valid))
        else:
            depth_min = depth_mean = depth_max = float('nan')

        self.depth_gt_csv_writer.writerow([
            sample_id,
            f'{rgb_stamp_s:.6f}',
            f'{self.latest_depth_gt_stamp_s:.6f}',
            f'{depth_age_s:.6f}',
            str(rgb_path),
            str(depth_path),
            self.current_x if self.current_x is not None else float('nan'),
            self.current_y if self.current_y is not None else float('nan'),
            self.current_z if self.current_z is not None else float('nan'),
            self.current_roll,
            self.current_pitch,
            self.current_yaw,
            self.current_gyro_rad_s[0],
            self.current_gyro_rad_s[1],
            self.current_gyro_rad_s[2],
            self.current_accel_m_s2[0],
            self.current_accel_m_s2[1],
            self.current_accel_m_s2[2],
            self.last_pan_delta_rad,
            self.last_pan_delta_source,
            depth_min,
            depth_mean,
            depth_max
        ])
        self.depth_gt_csv_file.flush()

    def criar_visualizacao_depth_gt(self, depth_m):
        """
        ==================================================================================
        Gera uma visualizacao colorida do mapa de profundidade ground truth.

        A funcao usa apenas pixels positivos e finitos para calcular uma normalizacao robusta
        entre os percentis 2 e 98. Em seguida, inverte a escala para destacar objetos proximos
        e aplica o colormap TURBO do OpenCV.

        A imagem resultante serve somente para inspecao em cv2.imshow(). Ela nao e salva como
        label de treino e nao participa da evasao visual.

        Fontes:
        [OpenCV applyColorMap] https://docs.opencv.org/4.x/d3/d50/group__imgproc__colormap.html
        [NumPy percentile] https://numpy.org/doc/stable/reference/generated/numpy.percentile.html
        ==================================================================================
        """

        valid = np.isfinite(depth_m) & (depth_m > 0.0)
        if np.count_nonzero(valid) == 0:
            return np.zeros((*depth_m.shape[:2], 3), dtype=np.uint8)

        p2, p98 = np.percentile(depth_m[valid], [2, 98])
        if p98 <= p2:
            p98 = p2 + 1.0

        depth_norm = np.clip((depth_m - p2) / (p98 - p2), 0.0, 1.0)
        depth_norm = ((1.0 - depth_norm) * 255).astype(np.uint8)
        depth_norm[~valid] = 0
        return cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)

    def normalizar_angulo_rad(self, angulo):
        """
        ==================================================================================
        Normaliza um angulo em radianos para o intervalo [-pi, pi].

        Essa normalizacao e usada no calculo do delta de yaw entre frames consecutivos.
        Sem ela, pequenas passagens pela descontinuidade de -pi/pi poderiam ser vistas como
        giros quase completos, gerando um warp incorreto na compensacao de pan.

        Fontes:
        [Python atan2] https://docs.python.org/3/library/math.html#math.atan2
        [Artigo - Garcia2016] https://doi.org/10.1109/icarsc.2016.46
        ==================================================================================
        """

        return math.atan2(math.sin(angulo), math.cos(angulo))

    def calcular_delta_pan_imu(self, frame_stamp_s):
        """
        ==================================================================================
        Estima o delta de pan/yaw da camera entre frames para compensacao visual.

        A funcao combina duas fontes inerciais: a diferenca de yaw estimada por
        VehicleAttitude e a integracao curta do eixo z do giroscopio bruto. O peso do
        giroscopio e controlado por pan_gyro_weight. O resultado e invertido antes de ser
        aplicado na homografia, pois a imagem atual precisa ser projetada de volta para
        reduzir o pan aparente entre frames consecutivos.

        O acelerometro nao e subtraido diretamente da imagem porque a compensacao de
        translacao depende da profundidade de cada pixel. Por isso, a translacao fica como
        fluxo residual para a evasao/estimativa de profundidade, e o acelerometro e salvo no
        dataset para o modelo futuro.

        Fontes:
        [PX4 VehicleAttitude] https://docs.px4.io/main/en/msg_docs/VehicleAttitude
        [PX4 SensorCombined] https://docs.px4.io/main/en/msg_docs/SensorCombined.html
        [Artigo - Tarrio2015] https://doi.org/10.1109/iccv.2015.87
        [Artigo - Garcia2016] https://doi.org/10.1109/icarsc.2016.46
        ==================================================================================
        """

        if not self.compensar_pan_yaw:
            self.prev_stabilization_stamp_s = frame_stamp_s
            self.prev_stabilization_yaw = self.current_yaw_attitude
            self.last_pan_delta_rad = 0.0
            self.last_pan_delta_source = 'disabled'
            return 0.0, 'disabled'

        attitude_delta = None
        if self.prev_stabilization_yaw is not None:
            attitude_delta = self.normalizar_angulo_rad(
                self.current_yaw_attitude - self.prev_stabilization_yaw
            )

        gyro_delta = None
        if self.prev_stabilization_stamp_s is not None:
            dt = frame_stamp_s - self.prev_stabilization_stamp_s
            if 0.0 < dt <= 0.25 and np.all(np.isfinite(self.current_gyro_rad_s)):
                gyro_delta = float(self.current_gyro_rad_s[2]) * dt

        if attitude_delta is not None and gyro_delta is not None:
            yaw_delta = (
                (1.0 - self.pan_gyro_weight) * attitude_delta +
                self.pan_gyro_weight * gyro_delta
            )
            source = 'attitude+gyro'
        elif attitude_delta is not None:
            yaw_delta = attitude_delta
            source = 'attitude'
        elif gyro_delta is not None:
            yaw_delta = gyro_delta
            source = 'gyro'
        else:
            yaw_delta = 0.0
            source = 'none'

        yaw_delta = max(-self.pan_max_delta_rad, min(self.pan_max_delta_rad, yaw_delta))
        pan_compensado = -yaw_delta * self.pan_yaw_gain

        self.prev_stabilization_stamp_s = frame_stamp_s
        self.prev_stabilization_yaw = self.current_yaw_attitude
        self.last_pan_delta_rad = pan_compensado
        self.last_pan_delta_source = source

        return pan_compensado, source

    def montar_homografia_compensacao_imu(self, msg):
        """
        ==================================================================================
        Monta a homografia usada para estabilizar a imagem com base na IMU/atitude.

        A parte absoluta da compensacao usa roll e pitch para reduzir tilt e inclinacao da
        camera, preservando a ideia de gimbal virtual ja existente no projeto. A parte
        incremental usa o delta de pan/yaw entre frames para remover o giro horizontal
        aparente antes do calculo de fluxo optico.

        A homografia segue a forma H = K_saida * R * K_entrada^-1, em que R representa a
        rotacao 3D equivalente da camera. Esse modelo e adequado para compensar rotacao
        pura da camera; translacoes continuam dependentes da profundidade da cena e sao
        deixadas para o fluxo residual ou para o modelo de profundidade a ser treinado.

        Fontes:
        [OpenCV Homography] https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html
        [OpenCV warpPerspective] https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html
        [PX4 VehicleAttitude] https://docs.px4.io/main/en/msg_docs/VehicleAttitude
        [Artigo - Tarrio2015] https://doi.org/10.1109/iccv.2015.87
        ==================================================================================
        """

        frame_stamp_s = self.image_timestamp_s(msg)
        theta_x = self.current_pitch if self.compensar_tilt_roll else 0.0
        theta_z = self.current_roll if self.compensar_tilt_roll else 0.0
        theta_y, source = self.calcular_delta_pan_imu(frame_stamp_s)

        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(theta_x), -math.sin(theta_x)],
            [0, math.sin(theta_x), math.cos(theta_x)]
        ])

        Ry = np.array([
            [math.cos(theta_y), 0, math.sin(theta_y)],
            [0, 1, 0],
            [-math.sin(theta_y), 0, math.cos(theta_y)]
        ])

        Rz = np.array([
            [math.cos(theta_z), -math.sin(theta_z), 0],
            [math.sin(theta_z), math.cos(theta_z), 0],
            [0, 0, 1]
        ])

        R = Rx @ Ry @ Rz

        K_saida = np.array([
            [self.K[0, 0] * self.stabilization_zoom, 0, self.stabilization_output_width * 0.5],
            [0, self.K[1, 1] * self.stabilization_zoom, self.stabilization_output_height * 0.5],
            [0, 0, 1]
        ])

        H = K_saida @ R @ np.linalg.inv(self.K)
        return H, theta_y, source

    def aplicar_compensacao_imu(self, cv_image, msg):
        """
        ==================================================================================
        Aplica a compensacao visual por IMU/atitude antes da evasao por fluxo optico.

        Quando compensacao_imu_ativa esta desligado, a funcao devolve a imagem original e
        uma mascara alfa equivalente. Quando esta ligado, calcula a homografia de roll,
        pitch e pan/yaw, desenha a geometria do recorte no frame original e gera a imagem
        estabilizada por cv2.warpPerspective.

        A imagem estabilizada e usada pelo Lucas-Kanade para reduzir fluxo causado por
        ego-rotacao da camera. O fluxo que sobra tende a representar translacao, paralaxe e
        objetos proximos, que sao exatamente os sinais uteis para estimativa de risco e para
        o futuro treino supervisionado com depth ground truth do Gazebo.

        Fontes:
        [OpenCV warpPerspective] https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html
        [OpenCV Optical Flow] https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html
        [Artigo - RealTimeMonocular2022] https://doi.org/10.1109/TITS.2022.3160741
        [Artigo - Tarrio2015] https://doi.org/10.1109/iccv.2015.87
        ==================================================================================
        """

        if not self.compensacao_imu_ativa:
            self.last_pan_delta_rad = 0.0
            self.last_pan_delta_source = 'disabled'
            return cv_image, cv_image[:, :, 3], cv_image.copy()

        H, pan_delta, pan_source = self.montar_homografia_compensacao_imu(msg)
        img_geometria = self.desenhar_telemetria_geometria(
            cv_image,
            H,
            self.stabilization_output_width,
            self.stabilization_output_height
        )

        cv2.putText(
            img_geometria,
            f"IMU pan={pan_delta:.4f} rad ({pan_source})",
            (12, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1
        )

        imagem_estabilizada = cv2.warpPerspective(
            cv_image,
            H,
            (self.stabilization_output_width, self.stabilization_output_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )
        mascara_alpha = imagem_estabilizada[:, :, 3]
        return imagem_estabilizada, mascara_alpha, img_geometria

    def desenhar_telemetria_geometria(self, frame_original, H, largura_out=640, altura_out=480):
        """
        ==================================================================================
        Desenha, sobre a imagem original da câmera, a região que será utilizada pela imagem
        estabilizada após a transformação de perspectiva.
        
        A função recebe a homografia H usada no warpPerspective, calcula H inversa e projeta
        os cantos da imagem estabilizada de saída de volta para o frame original. Em seguida,
        usa funções de desenho do OpenCV para mostrar a área válida de zoom/recorte sobre a
        imagem real da câmera.
        
        Esse recurso não interfere no controle do drone; ele serve para depuração visual da
        estabilização eletrônica, permitindo verificar quais pixels da imagem original estão
        sendo aproveitados após o warping.
        
        Fontes:
        [OpenCV Homography] https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html
        [OpenCV Drawing Functions] https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html
        [OpenCV perspectiveTransform] https://docs.opencv.org/4.x/d2/de8/group__core__array.html
        ==================================================================================
        """
        
        canvas = cv2.resize(frame_original, (0, 0), fx=1, fy=1)
        cantos_saida = np.array([
            [0, 0], [largura_out, 0], [largura_out, altura_out], [0, altura_out]
        ], dtype='float32').reshape(-1, 1, 2)

        H_inv = np.linalg.inv(H)
        cantos_na_origem = cv2.perspectiveTransform(cantos_saida, H_inv).reshape(-1, 2)

        pts_canvas = cantos_na_origem.astype(np.int32)
        pts_canvas = pts_canvas.reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts_canvas], isClosed=True, color=(0, 255, 0), thickness=2)
        
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [pts_canvas], (0, 255, 0))
        cv2.addWeighted(overlay, 0.2, canvas, 0.8, 0, canvas)
        cv2.putText(canvas, "AREA DE ZOOM", (pts_canvas[0][0][0], pts_canvas[0][0][1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return canvas

    def detectar_pontos_evasao(self, gray, valid_mask):
        """
        Detecta pontos em bordas/cantos dentro da regiao valida da imagem estabilizada.

        A ideia vem de VO semi-denso/edge-based: nao reconstruimos a cena inteira,
        apenas rastreamos pontos visuais bons o suficiente para estimar risco local.

        Fontes:
        [OpenCV goodFeaturesToTrack] https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html
        [OpenCV Canny] https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
        [Artigo - Tarrio2015] https://doi.org/10.1109/iccv.2015.87
        """

        altura, largura = gray.shape
        roi_mask = np.zeros_like(valid_mask)
        roi_mask[int(altura * 0.2):int(altura * 0.9), int(largura * 0.1):int(largura * 0.9)] = 255
        roi_mask = cv2.bitwise_and(roi_mask, valid_mask)

        edges = cv2.Canny(gray, 60, 160)
        feature_mask = cv2.bitwise_and(cv2.dilate(edges, None, iterations=1), roi_mask)
        if cv2.countNonZero(feature_mask) < 150:
            feature_mask = roi_mask

        return cv2.goodFeaturesToTrack(
            gray,
            maxCorners=180,
            qualityLevel=0.01,
            minDistance=12,
            blockSize=8,
            mask=feature_mask
        )

    def suavizar_comando_evasao(self, risk, lateral_body, brake):
        """
        Aplica filtro passa-baixa aos comandos reativos gerados pela visao.

        A suavizacao reduz oscilacoes entre frames consecutivos sem alterar a direcao
        geral estimada pela evasao visual.

        Fontes:
        [OpenCV Optical Flow] https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html
        [PX4 Offboard Mode] https://docs.px4.io/main/en/flight_modes/offboard
        """

        alpha = self.velocity_smooth_alpha
        self.obstacle_risk += alpha * (risk - self.obstacle_risk)
        self.avoid_lateral_body += alpha * (lateral_body - self.avoid_lateral_body)
        self.avoid_brake += alpha * (brake - self.avoid_brake)

        if abs(self.avoid_lateral_body) > 0.05:
            self.avoid_side_memory = math.copysign(1.0, self.avoid_lateral_body)

    def calcular_evasao_visual(self, imagem_estabilizada, mascara_alpha):
        """
        Estima risco de colisao por fluxo optico e profundidade inversa relativa.

        Com camera monocular, a escala absoluta e ambigua. Por isso usamos o
        principio de profundidade inversa: durante o movimento, pontos mais
        proximos tendem a produzir fluxo radial maior na imagem. O resultado
        alimenta um campo repulsivo simples, nao um mapa 3D completo.

        Fontes:
        [OpenCV Lucas-Kanade Optical Flow] https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html
        [Artigo - Bhattacharya2024] https://doi.org/10.48550/arXiv.2411.03303
        [Artigo - RealTimeMonocular2022] https://doi.org/10.1109/TITS.2022.3160741
        [Artigo - Vyas2022] https://doi.org/10.48550/arXiv.2205.01399
        """

        frame_bgr = imagem_estabilizada[:, :, :3]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        valid_mask = (mascara_alpha > 0).astype(np.uint8) * 255
        valid_mask = cv2.erode(valid_mask, None, iterations=1)

        debug = frame_bgr.copy()
        altura, largura = gray.shape
        cx, cy = largura * 0.5, altura * 0.5
        cv2.rectangle(
            debug,
            (int(largura * 0.08), int(altura * 0.18)),
            (int(largura * 0.92), int(altura * 0.90)),
            (255, 180, 0),
            1
        )

        if self.prev_gray_avoidance is None or self.prev_points_avoidance is None:
            self.prev_gray_avoidance = gray
            self.prev_points_avoidance = self.detectar_pontos_evasao(gray, valid_mask)
            self.suavizar_comando_evasao(0.0, 0.0, 0.0)
            return debug

        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray_avoidance,
            gray,
            self.prev_points_avoidance,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
        )

        if next_points is None or status is None:
            self.prev_gray_avoidance = gray
            self.prev_points_avoidance = self.detectar_pontos_evasao(gray, valid_mask)
            self.suavizar_comando_evasao(0.0, 0.0, 0.0)
            return debug

        old = self.prev_points_avoidance[status.flatten() == 1].reshape(-1, 2)
        new = next_points[status.flatten() == 1].reshape(-1, 2)

        dentro = (
            (new[:, 0] >= 0) & (new[:, 0] < largura) &
            (new[:, 1] >= 0) & (new[:, 1] < altura)
        )
        old = old[dentro]
        new = new[dentro]

        if len(new) < 12:
            self.prev_gray_avoidance = gray
            self.prev_points_avoidance = self.detectar_pontos_evasao(gray, valid_mask)
            self.suavizar_comando_evasao(0.0, 0.0, 0.0)
            return debug

        valid_pixels = valid_mask[new[:, 1].astype(int), new[:, 0].astype(int)] > 0
        old = old[valid_pixels]
        new = new[valid_pixels]

        flow = new - old
        radial = new - np.array([[cx, cy]])
        radial_norm = np.linalg.norm(radial, axis=1) + 1e-6
        radial_unit = radial / radial_norm[:, None]
        radial_flow = np.sum(flow * radial_unit, axis=1)

        central_x = 1.0 - np.minimum(np.abs(new[:, 0] - cx) / (largura * 0.5), 1.0)
        central_y = 1.0 - np.minimum(np.abs(new[:, 1] - cy) / (altura * 0.65), 1.0)
        central_weight = np.clip(central_x * central_y, 0.0, 1.0)

        speed_xy = math.sqrt(self.smooth_vx**2 + self.smooth_vy**2)
        speed_factor = min(1.0, max(0.0, speed_xy / 2.0))

        inverse_depth_score = np.clip((radial_flow - 0.25) / 8.0, 0.0, 1.0)
        point_risk = inverse_depth_score * central_weight
        point_risk *= speed_factor

        active = point_risk > 0.03
        if np.count_nonzero(active) < 8:
            risk = 0.0
            lateral_body = 0.0
        else:
            active_risk = point_risk[active]
            active_points = new[active]
            risk = float(np.clip(np.percentile(active_risk, 75) * 1.75, 0.0, 1.0))

            left = float(np.sum(active_risk[active_points[:, 0] < cx]))
            right = float(np.sum(active_risk[active_points[:, 0] >= cx]))
            balance = (right - left) / (right + left + 1e-6)

            if abs(balance) < 0.15:
                side = self.avoid_side_memory
            else:
                side = -math.copysign(1.0, balance)

            lateral_body = side * self.max_lateral_acceleration * risk

            for p0, p1, r in zip(old[active], active_points, active_risk):
                color = (0, 0, 255) if r > 0.5 else (0, 255, 255)
                cv2.arrowedLine(debug, tuple(p0.astype(int)), tuple(p1.astype(int)), color, 1, tipLength=0.3)

        brake = min(self.avoidance_max_brake, risk * self.avoidance_max_brake)
        self.suavizar_comando_evasao(risk, lateral_body, brake)

        self.prev_gray_avoidance = gray
        if len(new) < 80:
            self.prev_points_avoidance = self.detectar_pontos_evasao(gray, valid_mask)
        else:
            self.prev_points_avoidance = new.reshape(-1, 1, 2).astype(np.float32)

        cv2.putText(
            debug,
            f"risco={self.obstacle_risk:.2f} lateral={self.avoid_lateral_body:.2f} freio={self.avoid_brake:.2f}",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            1
        )
        return debug

    def image_callback(self, msg):
        """
        ==================================================================================
        Processa cada frame recebido da camera monocular simulada.

        A imagem ROS e convertida para matriz OpenCV por meio do cv_bridge. Em seguida, a
        compensacao visual por IMU/atitude usa pitch e roll absolutos para reduzir tilt/roll
        e usa o delta de yaw/pan entre frames para reduzir o giro horizontal aparente. A
        homografia H = K_saida * R * K_entrada^-1 projeta a imagem como se houvesse um gimbal
        virtual antes do calculo de fluxo optico.

        Quando um topico de ground truth de profundidade foi configurado, a funcao tenta
        parear o frame RGB monocular com o ultimo depth sintetico do Gazebo. Esse depth pode
        ser visualizado e salvo em dataset junto com pose, atitude e IMU bruta, mas nao entra
        no calculo de evasao reativa.

        A funcao tambem atualiza as janelas de depuracao visual usadas durante os testes:
        a imagem original com a geometria do warping, a visualizacao da evasao reativa e,
        quando disponivel, o mapa de profundidade ground truth colorizado. O waitKey(1) e
        mantido para permitir que o OpenCV atualize as janelas a cada frame.

        Importante: esta compensacao reduz ego-rotacao visual, mas nao remove translacao da
        camera, porque translacao exige profundidade por pixel. A profundidade do Gazebo fica
        como ground truth para validacao/dataset e para o treino futuro do modelo.
        
        Fontes:
        [cv_bridge] https://docs.ros.org/en/jade/api/cv_bridge/html/python/
        [OpenCV Homography] https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html
        [OpenCV Camera Calibration] https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
        [OpenCV warpPerspective] https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html
        [Gazebo DepthCameraSensor] https://gazebosim.org/api/sensors/7/classgz_1_1sensors_1_1DepthCameraSensor.html
        [Artigo - Tarrio2015] https://doi.org/10.1109/iccv.2015.87
        ==================================================================================
        """
        
        resolucao_largura = msg.width
        resolucao_altura = msg.height
        # formato_ros = msg.encoding
        
        # self.get_logger().info(f'Frame Recebido - Resolução: {resolucao_largura}x{resolucao_altura} pixels | Formato: {formato_ros}')
        # Frame Recebido - Resolução: 1280x960 pixels | Formato: rgb8
        
        try:
            cv_image = np.ones((resolucao_altura,resolucao_largura, 4),dtype=np.uint8, order='F') * 255
            cv_image[:,:,:3] = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.rgb_frames_seen += 1

            depth_gt, rgb_stamp_s, depth_age_s = self.obter_depth_gt_sincronizado(msg)
            depth_gt_visual = None
            if depth_gt is not None:
                if depth_gt.shape[:2] != (resolucao_altura, resolucao_largura) and not self.depth_gt_shape_warned:
                    self.get_logger().warning(
                        f'Depth GT com resolucao {depth_gt.shape[:2]}, RGB com '
                        f'{(resolucao_altura, resolucao_largura)}. Para treino pixel-a-pixel, '
                        'configure o render de depth com a mesma resolucao/FOV da monocular.'
                    )
                    self.depth_gt_shape_warned = True

                self.salvar_par_ground_truth(cv_image[:, :, :3], depth_gt, rgb_stamp_s, depth_age_s)
                depth_gt_visual = self.criar_visualizacao_depth_gt(depth_gt)
            
            # ---- COMPENSACAO DA IMAGEM (IMU + ATITUDE) ----
            imagem_estabilizada, mascara_alpha, img_geometria = self.aplicar_compensacao_imu(
                cv_image,
                msg
            )

            # ---- VISAO COMPUTACIONAL PARA DESVIO REATIVO ----
            if self.evasao_visual_ativa:
                visao_da_evasao = self.calcular_evasao_visual(imagem_estabilizada, mascara_alpha)
            else:
                self.prev_gray_avoidance = None
                self.prev_points_avoidance = None
                visao_da_evasao = imagem_estabilizada[:, :, :3].copy()
            
            #cv2.imshow("Visão do Drone Original (Com tremor)", cv_image)
            cv2.imshow("Visao do Drone Original com a Geometria do Warping", img_geometria)
            #cv2.imshow("Visão do Drone Estabilizada (Usando IMU)", imagem_estabilizada)
            cv2.imshow("Deteccao Reativa (Fluxo Optico)", visao_da_evasao)
            if depth_gt_visual is not None:
                cv2.imshow("Ground Truth Depth Gazebo", depth_gt_visual)
            #cv2.imshow("Mascara Alpha (Branco = Pixel Valido)", mascara_alpha)
            cv2.waitKey(1) # Necessário para o OpenCV atualizar a janela
        except Exception as e:
            self.get_logger().error(f'Erro na conversão da imagem: {e}')

    def attitude_callback(self, msg):
        """
        =========================================================================================
        Recebe a atitude estimada do drone e converte a orientação de quaternion para ângulos
        de Euler roll, pitch e yaw.
        
        O PX4 publica VehicleAttitude com quaternion no formato q(w, x, y, z), seguindo a
        convenção de Hamilton. A mensagem representa a rotação do corpo do drone no referencial
        FRD para o referencial NED. O script extrai roll e pitch para compensar tilt/roll da
        camera e tambem extrai yaw para estimar o pan entre frames consecutivos.
        
        A conversão implementada segue as fórmulas usuais de quaternion para Euler, com trava
        de segurança no pitch quando o valor de asin ultrapassa o intervalo [-1, 1] por erro
        numérico. O yaw operacional usado na navegação continua vindo do campo heading da
        posição local; o yaw desta mensagem e reservado para compensacao visual.
        
        Fontes:
        [PX4 VehicleAttitude] https://docs.px4.io/main/en/msg_docs/VehicleAttitude
        [MAVLink ATTITUDE_QUATERNION] https://mavlink.io/en/messages/common.html#ATTITUDE_QUATERNION
        [Conversão Quaternion-Euler] https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
        [Artigo - Garcia2016] https://doi.org/10.1109/icarsc.2016.46
        =========================================================================================
        """
        
        w, x, y, z = msg.q[0], msg.q[1], msg.q[2], msg.q[3]
        
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        self.current_roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1:
            self.current_pitch = math.copysign(math.pi / 2.0, sinp)
        else:
            self.current_pitch = math.asin(sinp)

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        self.current_yaw_attitude = math.atan2(siny_cosp, cosy_cosp)

    def destroy_node(self):
        """
        ==================================================================================
        Fecha recursos abertos pelo no antes de delegar a destruicao para a classe base.

        Atualmente o recurso adicional e o arquivo CSV de metadados do dataset de ground
        truth, aberto apenas quando save_ground_truth_dataset esta ativo e o primeiro par
        RGB/depth e salvo. Fechar o arquivo garante que os metadados sejam gravados
        corretamente ao encerrar o processo pelo fluxo normal do ROS 2.

        Fontes:
        [ROS 2 Node] https://docs.ros.org/en/humble/Concepts/Basic/About-Nodes.html
        [Python File Objects] https://docs.python.org/3/tutorial/inputoutput.html#reading-and-writing-files
        ==================================================================================
        """

        if self.depth_gt_csv_file is not None:
            self.depth_gt_csv_file.close()
            self.depth_gt_csv_file = None
        super().destroy_node()
