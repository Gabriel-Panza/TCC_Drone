import os
import time
import json
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
            int(self.declare_parameter('ground_truth_save_every_n', 1).value)
        )
        self.ground_truth_memmap_capacity = max(
            16,
            int(self.declare_parameter('ground_truth_memmap_capacity', 4096).value)
        )
        self.ground_truth_rgb_width = max(
            16,
            int(self.declare_parameter('ground_truth_rgb_width', 160).value)
        )
        self.ground_truth_rgb_height = max(
            16,
            int(self.declare_parameter('ground_truth_rgb_height', 120).value)
        )
        self.ground_truth_depth_width = max(
            8,
            int(self.declare_parameter('ground_truth_depth_width', 40).value)
        )
        self.ground_truth_depth_height = max(
            8,
            int(self.declare_parameter('ground_truth_depth_height', 30).value)
        )
        self.ground_truth_depth_max_m = float(
            self.declare_parameter('ground_truth_depth_max_m', 50.0).value
        )
        self.ground_truth_max_flow_points = max(
            16,
            int(self.declare_parameter('ground_truth_max_flow_points', 180).value)
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
        self.visual_intervals_seen = 0
        self.depth_intervals_saved = 0
        self.depth_gt_run_dir = None
        self.depth_gt_manifest_path = None
        self.depth_gt_intervals = None
        self.depth_gt_image_delta = None
        self.depth_gt_depth_delta = None
        self.depth_gt_depth_mask = None
        self.depth_gt_flow_vectors = None
        self.depth_gt_capacity_warned = False
        self.prev_depth_interval_ref = None
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
        self.yaw_smooth_alpha = 0.7

        self.prev_gray_avoidance = None
        self.prev_points_avoidance = None
        self.obstacle_risk = 0.0
        self.avoid_lateral_body = 0.0
        self.avoid_brake = 0.0
        self.avoid_side_memory = 0.8
        self.avoidance_max_brake = 0.3
        self.raio_finalizacao = 2.0
        self.raio_desativa_evasao_final = 10.0
        self.evasao_visual_ativa = True
        self.max_lateral_acceleration = 7.0

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
        self.raio_de_aceitacao = 3.5     # Raio de aceitação para mudar de waypoint
        
        self.zona_frenagem_curva = 7.0
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

        if evasao_habilitada and self.obstacle_risk > 0.05:
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

    def dtype_intervalos_ground_truth(self):
        """
        Retorna o schema numerico salvo no memmap de intervalos visuais.

        Os campos representam variacoes entre duas atualizacoes consecutivas da
        logica de optical flow. Valores absolutos de pose/atitude nao sao gravados.
        """

        return np.dtype([
            ('sample_id', 'i4'),
            ('dt_s', 'f4'),
            ('depth_age_s', 'f4'),
            ('depth_dt_s', 'f4'),
            ('delta_x_m', 'f4'),
            ('delta_y_m', 'f4'),
            ('delta_z_m', 'f4'),
            ('delta_roll_rad', 'f4'),
            ('delta_pitch_rad', 'f4'),
            ('delta_yaw_heading_rad', 'f4'),
            ('delta_yaw_attitude_rad', 'f4'),
            ('delta_gyro_x_rad_s', 'f4'),
            ('delta_gyro_y_rad_s', 'f4'),
            ('delta_gyro_z_rad_s', 'f4'),
            ('gyro_x_integral_rad', 'f4'),
            ('gyro_y_integral_rad', 'f4'),
            ('gyro_z_integral_rad', 'f4'),
            ('delta_accel_x_m_s2', 'f4'),
            ('delta_accel_y_m_s2', 'f4'),
            ('delta_accel_z_m_s2', 'f4'),
            ('delta_smooth_vx_m_s', 'f4'),
            ('delta_smooth_vy_m_s', 'f4'),
            ('delta_obstacle_risk', 'f4'),
            ('delta_avoid_lateral_body', 'f4'),
            ('delta_avoid_brake', 'f4'),
            ('pan_comp_delta_rad', 'f4'),
            ('flow_valid_points', 'i4'),
            ('flow_active_points', 'i4'),
            ('flow_track_retention_pct', 'f4'),
            ('flow_mean_x_px', 'f4'),
            ('flow_mean_y_px', 'f4'),
            ('flow_mag_mean_px', 'f4'),
            ('flow_mag_p90_px', 'f4'),
            ('radial_flow_mean_px', 'f4'),
            ('radial_flow_p90_px', 'f4'),
            ('point_risk_mean_delta_basis', 'f4'),
            ('point_risk_p75_delta_basis', 'f4'),
            ('delta_depth_min_m', 'f4'),
            ('delta_depth_mean_m', 'f4'),
            ('delta_depth_p10_m', 'f4'),
            ('delta_depth_p50_m', 'f4'),
            ('delta_depth_p90_m', 'f4'),
            ('delta_depth_close_2m_pp', 'f4'),
            ('delta_depth_close_5m_pp', 'f4'),
            ('delta_depth_close_10m_pp', 'f4'),
            ('delta_valid_px_pct', 'f4'),
        ])

    def preparar_dataset_ground_truth(self):
        """
        Prepara os memmaps do dataset sincronizado por intervalo visual.

        Cada linha valida representa o mesmo intervalo usado pelo optical flow que desenha
        as flechas de proximidade. O dataset salva deltas de estado, deltas de depth,
        diferenca de imagem estabilizada e vetores de flow relativos ao centro da imagem.
        """

        base_dir = Path(os.path.expanduser(self.ground_truth_dataset_dir))
        self.depth_gt_run_dir = base_dir / datetime.now().strftime('run_%Y%m%d_%H%M%S')
        self.depth_gt_run_dir.mkdir(parents=True, exist_ok=True)
        self.depth_gt_manifest_path = self.depth_gt_run_dir / 'manifest.json'

        capacidade = self.ground_truth_memmap_capacity
        rgb_shape = (
            capacidade,
            self.ground_truth_rgb_height,
            self.ground_truth_rgb_width,
            3,
        )
        depth_shape = (
            capacidade,
            self.ground_truth_depth_height,
            self.ground_truth_depth_width,
        )
        flow_shape = (capacidade, self.ground_truth_max_flow_points, 6)

        self.depth_gt_intervals = np.lib.format.open_memmap(
            self.depth_gt_run_dir / 'intervals.npy',
            mode='w+',
            dtype=self.dtype_intervalos_ground_truth(),
            shape=(capacidade,),
        )
        self.depth_gt_image_delta = np.lib.format.open_memmap(
            self.depth_gt_run_dir / 'image_delta_bgr.npy',
            mode='w+',
            dtype=np.int16,
            shape=rgb_shape,
        )
        self.depth_gt_depth_delta = np.lib.format.open_memmap(
            self.depth_gt_run_dir / 'depth_delta_log.npy',
            mode='w+',
            dtype=np.float32,
            shape=depth_shape,
        )
        self.depth_gt_depth_mask = np.lib.format.open_memmap(
            self.depth_gt_run_dir / 'depth_delta_mask.npy',
            mode='w+',
            dtype=np.uint8,
            shape=depth_shape,
        )
        self.depth_gt_flow_vectors = np.lib.format.open_memmap(
            self.depth_gt_run_dir / 'flow_vectors.npy',
            mode='w+',
            dtype=np.float32,
            shape=flow_shape,
        )

        self.atualizar_manifesto_depth_gt()
        self.get_logger().info(
            f'Dataset de intervalos depth/flow sendo salvo em memmap: {self.depth_gt_run_dir}'
        )

    def atualizar_manifesto_depth_gt(self):
        """Atualiza o manifesto que descreve os memmaps validos da run."""

        if self.depth_gt_manifest_path is None:
            return

        manifesto = {
            'schema_version': 'depth_interval_memmap_v1',
            'description': (
                'Cada amostra representa a variacao entre duas atualizacoes consecutivas '
                'da logica de proximidade visual por optical flow.'
            ),
            'num_samples': int(self.depth_intervals_saved),
            'capacity': int(self.ground_truth_memmap_capacity),
            'save_every_n_visual_intervals': int(self.ground_truth_save_every_n),
            'rgb_delta_shape': [
                int(self.ground_truth_rgb_height),
                int(self.ground_truth_rgb_width),
                3,
            ],
            'depth_delta_shape': [
                int(self.ground_truth_depth_height),
                int(self.ground_truth_depth_width),
            ],
            'depth_max_m': float(self.ground_truth_depth_max_m),
            'max_flow_points': int(self.ground_truth_max_flow_points),
            'arrays': {
                'intervals': 'intervals.npy',
                'image_delta_bgr': 'image_delta_bgr.npy',
                'depth_delta_log': 'depth_delta_log.npy',
                'depth_delta_mask': 'depth_delta_mask.npy',
                'flow_vectors': 'flow_vectors.npy',
            },
            'flow_vector_columns': [
                'x_center_norm',
                'y_center_norm',
                'flow_x_px',
                'flow_y_px',
                'radial_flow_px',
                'point_risk_delta_basis',
            ],
            'interval_fields': list(self.dtype_intervalos_ground_truth().names),
        }

        with open(self.depth_gt_manifest_path, mode='w', encoding='utf-8') as fp:
            json.dump(manifesto, fp, indent=2)

    def capturar_estado_intervalo(self, frame_stamp_s, depth_age_s):
        """Captura o estado atual apenas para calcular deltas antes do salvamento."""

        def numero(valor):
            return float(valor) if valor is not None else float('nan')

        return {
            'frame_stamp_s': numero(frame_stamp_s),
            'depth_stamp_s': numero(self.latest_depth_gt_stamp_s),
            'depth_age_s': numero(depth_age_s),
            'x': numero(self.current_x),
            'y': numero(self.current_y),
            'z': numero(self.current_z),
            'roll': float(self.current_roll),
            'pitch': float(self.current_pitch),
            'yaw_heading': float(self.current_yaw),
            'yaw_attitude': float(self.current_yaw_attitude),
            'gyro': np.asarray(self.current_gyro_rad_s, dtype=float).copy(),
            'accel': np.asarray(self.current_accel_m_s2, dtype=float).copy(),
            'smooth_vx': float(self.smooth_vx),
            'smooth_vy': float(self.smooth_vy),
            'obstacle_risk': float(self.obstacle_risk),
            'avoid_lateral_body': float(self.avoid_lateral_body),
            'avoid_brake': float(self.avoid_brake),
            'pan_comp_delta_rad': float(self.last_pan_delta_rad),
        }

    def metricas_depth_memmap(self, depth_m):
        """Calcula metricas internas de depth usadas somente para gravar variacoes."""

        valid_mask = np.isfinite(depth_m) & (depth_m > 0.0)
        valores = np.asarray(depth_m[valid_mask], dtype=float)
        if valores.size == 0:
            return {
                'min': float('nan'),
                'mean': float('nan'),
                'p10': float('nan'),
                'p50': float('nan'),
                'p90': float('nan'),
                'close_2': float('nan'),
                'close_5': float('nan'),
                'close_10': float('nan'),
                'valid_pct': 0.0,
            }

        return {
            'min': float(np.min(valores)),
            'mean': float(np.mean(valores)),
            'p10': float(np.percentile(valores, 10)),
            'p50': float(np.percentile(valores, 50)),
            'p90': float(np.percentile(valores, 90)),
            'close_2': float((valores < 2.0).mean() * 100.0),
            'close_5': float((valores < 5.0).mean() * 100.0),
            'close_10': float((valores < 10.0).mean() * 100.0),
            'valid_pct': float(valid_mask.mean() * 100.0),
        }

    def preparar_delta_imagem_memmap(self, prev_bgr, curr_bgr):
        """Reduz a imagem estabilizada e salva apenas a diferenca entre frames."""

        tamanho = (self.ground_truth_rgb_width, self.ground_truth_rgb_height)
        prev_small = cv2.resize(prev_bgr, tamanho, interpolation=cv2.INTER_AREA).astype(np.int16)
        curr_small = cv2.resize(curr_bgr, tamanho, interpolation=cv2.INTER_AREA).astype(np.int16)
        return curr_small - prev_small

    def preparar_delta_depth_memmap(self, prev_depth_m, curr_depth_m):
        """Gera o alvo dense como delta de log-depth e mascara valida do intervalo."""

        tamanho = (self.ground_truth_depth_width, self.ground_truth_depth_height)
        prev_valid = np.isfinite(prev_depth_m) & (prev_depth_m > 0.0)
        curr_valid = np.isfinite(curr_depth_m) & (curr_depth_m > 0.0)

        prev_clip = np.where(
            prev_valid,
            np.clip(prev_depth_m, 0.1, self.ground_truth_depth_max_m),
            self.ground_truth_depth_max_m,
        ).astype(np.float32)
        curr_clip = np.where(
            curr_valid,
            np.clip(curr_depth_m, 0.1, self.ground_truth_depth_max_m),
            self.ground_truth_depth_max_m,
        ).astype(np.float32)

        prev_small = cv2.resize(prev_clip, tamanho, interpolation=cv2.INTER_AREA)
        curr_small = cv2.resize(curr_clip, tamanho, interpolation=cv2.INTER_AREA)
        prev_mask = cv2.resize(prev_valid.astype(np.float32), tamanho, interpolation=cv2.INTER_AREA) > 0.5
        curr_mask = cv2.resize(curr_valid.astype(np.float32), tamanho, interpolation=cv2.INTER_AREA) > 0.5

        mask = (prev_mask & curr_mask).astype(np.uint8)
        delta_log = (np.log1p(curr_small) - np.log1p(prev_small)).astype(np.float32)
        delta_log[mask == 0] = 0.0
        return delta_log, mask

    def preparar_vetores_flow_memmap(self, flow_interval, largura, altura):
        """Prepara vetores de flow com coordenadas relativas ao centro da imagem."""

        matriz = np.zeros((self.ground_truth_max_flow_points, 6), dtype=np.float32)
        if flow_interval is None:
            return matriz

        new = np.asarray(flow_interval.get('new_points', []), dtype=np.float32)
        flow = np.asarray(flow_interval.get('flow', []), dtype=np.float32)
        radial_flow = np.asarray(flow_interval.get('radial_flow', []), dtype=np.float32)
        point_risk = np.asarray(flow_interval.get('point_risk', []), dtype=np.float32)
        n = min(len(new), len(flow), len(radial_flow), len(point_risk), self.ground_truth_max_flow_points)
        if n <= 0:
            return matriz

        centro = np.array([largura * 0.5, altura * 0.5], dtype=np.float32)
        escala = np.array([max(largura * 0.5, 1.0), max(altura * 0.5, 1.0)], dtype=np.float32)
        rel = (new[:n] - centro) / escala
        matriz[:n, 0:2] = rel
        matriz[:n, 2:4] = flow[:n]
        matriz[:n, 4] = radial_flow[:n]
        matriz[:n, 5] = point_risk[:n]
        return matriz

    def registrar_intervalo_ground_truth(self, imagem_estabilizada_bgr, depth_m, estado_atual, flow_interval):
        """
        Salva uma amostra sincronizada com o intervalo usado pelo optical flow.

        A funcao nunca grava pose/atitude absolutas: usa o estado anterior apenas em memoria
        para calcular deltas e, em seguida, substitui a referencia pelo frame atual.
        """

        if not self.save_ground_truth_dataset:
            return

        referencia_atual = {
            'imagem_bgr': imagem_estabilizada_bgr.copy(),
            'depth_m': None if depth_m is None else np.asarray(depth_m, dtype=np.float32).copy(),
            'estado': estado_atual,
        }

        referencia_anterior = self.prev_depth_interval_ref
        self.prev_depth_interval_ref = referencia_atual

        if referencia_anterior is None or flow_interval is None:
            return

        self.visual_intervals_seen += 1
        if self.visual_intervals_seen % self.ground_truth_save_every_n != 0:
            return

        if referencia_anterior['depth_m'] is None or referencia_atual['depth_m'] is None:
            return

        if self.depth_gt_intervals is None:
            self.preparar_dataset_ground_truth()

        if self.depth_intervals_saved >= self.ground_truth_memmap_capacity:
            if not self.depth_gt_capacity_warned:
                self.get_logger().warning(
                    'Capacidade do memmap de depth/flow esgotada. '
                    'Aumente ground_truth_memmap_capacity para runs maiores.'
                )
                self.depth_gt_capacity_warned = True
            return

        idx = self.depth_intervals_saved
        prev_estado = referencia_anterior['estado']
        curr_estado = referencia_atual['estado']
        prev_depth_metricas = self.metricas_depth_memmap(referencia_anterior['depth_m'])
        curr_depth_metricas = self.metricas_depth_memmap(referencia_atual['depth_m'])

        dt_s = curr_estado['frame_stamp_s'] - prev_estado['frame_stamp_s']
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            dt_s = float('nan')

        def delta(campo):
            return float(curr_estado[campo] - prev_estado[campo])

        def delta_angulo(campo):
            return self.normalizar_angulo_rad(delta(campo))

        def delta_depth(campo):
            return float(curr_depth_metricas[campo] - prev_depth_metricas[campo])

        flow = np.asarray(flow_interval.get('flow', []), dtype=float)
        radial_flow = np.asarray(flow_interval.get('radial_flow', []), dtype=float)
        point_risk = np.asarray(flow_interval.get('point_risk', []), dtype=float)
        flow_mag = np.linalg.norm(flow, axis=1) if flow.size else np.array([], dtype=float)
        prev_points_total = max(1, int(flow_interval.get('prev_points_total', 0)))
        valid_points = int(flow_interval.get('valid_points', len(flow_mag)))
        active_points = int(flow_interval.get('active_points', 0))

        linha = np.zeros(1, dtype=self.dtype_intervalos_ground_truth())
        linha['sample_id'][0] = idx + 1
        linha['dt_s'][0] = dt_s
        linha['depth_age_s'][0] = curr_estado['depth_age_s']
        linha['depth_dt_s'][0] = curr_estado['depth_stamp_s'] - prev_estado['depth_stamp_s']
        linha['delta_x_m'][0] = delta('x')
        linha['delta_y_m'][0] = delta('y')
        linha['delta_z_m'][0] = delta('z')
        linha['delta_roll_rad'][0] = delta_angulo('roll')
        linha['delta_pitch_rad'][0] = delta_angulo('pitch')
        linha['delta_yaw_heading_rad'][0] = delta_angulo('yaw_heading')
        linha['delta_yaw_attitude_rad'][0] = delta_angulo('yaw_attitude')
        linha['delta_gyro_x_rad_s'][0] = curr_estado['gyro'][0] - prev_estado['gyro'][0]
        linha['delta_gyro_y_rad_s'][0] = curr_estado['gyro'][1] - prev_estado['gyro'][1]
        linha['delta_gyro_z_rad_s'][0] = curr_estado['gyro'][2] - prev_estado['gyro'][2]
        if np.isfinite(dt_s):
            linha['gyro_x_integral_rad'][0] = 0.5 * (curr_estado['gyro'][0] + prev_estado['gyro'][0]) * dt_s
            linha['gyro_y_integral_rad'][0] = 0.5 * (curr_estado['gyro'][1] + prev_estado['gyro'][1]) * dt_s
            linha['gyro_z_integral_rad'][0] = 0.5 * (curr_estado['gyro'][2] + prev_estado['gyro'][2]) * dt_s
        else:
            linha['gyro_x_integral_rad'][0] = float('nan')
            linha['gyro_y_integral_rad'][0] = float('nan')
            linha['gyro_z_integral_rad'][0] = float('nan')
        linha['delta_accel_x_m_s2'][0] = curr_estado['accel'][0] - prev_estado['accel'][0]
        linha['delta_accel_y_m_s2'][0] = curr_estado['accel'][1] - prev_estado['accel'][1]
        linha['delta_accel_z_m_s2'][0] = curr_estado['accel'][2] - prev_estado['accel'][2]
        linha['delta_smooth_vx_m_s'][0] = delta('smooth_vx')
        linha['delta_smooth_vy_m_s'][0] = delta('smooth_vy')
        linha['delta_obstacle_risk'][0] = delta('obstacle_risk')
        linha['delta_avoid_lateral_body'][0] = delta('avoid_lateral_body')
        linha['delta_avoid_brake'][0] = delta('avoid_brake')
        linha['pan_comp_delta_rad'][0] = curr_estado['pan_comp_delta_rad']
        linha['flow_valid_points'][0] = valid_points
        linha['flow_active_points'][0] = active_points
        linha['flow_track_retention_pct'][0] = valid_points / prev_points_total * 100.0
        linha['flow_mean_x_px'][0] = float(np.mean(flow[:, 0])) if len(flow_mag) else 0.0
        linha['flow_mean_y_px'][0] = float(np.mean(flow[:, 1])) if len(flow_mag) else 0.0
        linha['flow_mag_mean_px'][0] = float(np.mean(flow_mag)) if len(flow_mag) else 0.0
        linha['flow_mag_p90_px'][0] = float(np.percentile(flow_mag, 90)) if len(flow_mag) else 0.0
        linha['radial_flow_mean_px'][0] = float(np.mean(radial_flow)) if radial_flow.size else 0.0
        linha['radial_flow_p90_px'][0] = float(np.percentile(radial_flow, 90)) if radial_flow.size else 0.0
        linha['point_risk_mean_delta_basis'][0] = float(np.mean(point_risk)) if point_risk.size else 0.0
        linha['point_risk_p75_delta_basis'][0] = float(np.percentile(point_risk, 75)) if point_risk.size else 0.0
        linha['delta_depth_min_m'][0] = delta_depth('min')
        linha['delta_depth_mean_m'][0] = delta_depth('mean')
        linha['delta_depth_p10_m'][0] = delta_depth('p10')
        linha['delta_depth_p50_m'][0] = delta_depth('p50')
        linha['delta_depth_p90_m'][0] = delta_depth('p90')
        linha['delta_depth_close_2m_pp'][0] = delta_depth('close_2')
        linha['delta_depth_close_5m_pp'][0] = delta_depth('close_5')
        linha['delta_depth_close_10m_pp'][0] = delta_depth('close_10')
        linha['delta_valid_px_pct'][0] = delta_depth('valid_pct')

        delta_depth_log, depth_mask = self.preparar_delta_depth_memmap(
            referencia_anterior['depth_m'],
            referencia_atual['depth_m'],
        )
        self.depth_gt_intervals[idx] = linha[0]
        self.depth_gt_image_delta[idx] = self.preparar_delta_imagem_memmap(
            referencia_anterior['imagem_bgr'],
            referencia_atual['imagem_bgr'],
        )
        self.depth_gt_depth_delta[idx] = delta_depth_log
        self.depth_gt_depth_mask[idx] = depth_mask
        self.depth_gt_flow_vectors[idx] = self.preparar_vetores_flow_memmap(
            flow_interval,
            imagem_estabilizada_bgr.shape[1],
            imagem_estabilizada_bgr.shape[0],
        )
        self.depth_intervals_saved += 1

        self.depth_gt_intervals.flush()
        self.depth_gt_image_delta.flush()
        self.depth_gt_depth_delta.flush()
        self.depth_gt_depth_mask.flush()
        self.depth_gt_flow_vectors.flush()
        self.atualizar_manifesto_depth_gt()

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
            return debug, None

        prev_points_total = int(len(self.prev_points_avoidance))

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
            return debug, None

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
            return debug, None

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

        inverse_depth_score = np.clip((radial_flow - 0.3) / 7.5, 0.0, 1.0)
        point_risk = inverse_depth_score * central_weight
        point_risk *= speed_factor

        active = point_risk > 0.03
        if np.count_nonzero(active) < 7:
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

        flow_interval = {
            'prev_points_total': prev_points_total,
            'valid_points': int(len(new)),
            'active_points': int(np.count_nonzero(active)),
            'old_points': old.copy(),
            'new_points': new.copy(),
            'flow': flow.copy(),
            'radial_flow': radial_flow.copy(),
            'point_risk': point_risk.copy(),
        }

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
        return debug, flow_interval

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
            if rgb_stamp_s is None:
                rgb_stamp_s = self.image_timestamp_s(msg)
            depth_gt_visual = None
            if depth_gt is not None:
                if depth_gt.shape[:2] != (resolucao_altura, resolucao_largura) and not self.depth_gt_shape_warned:
                    self.get_logger().warning(
                        f'Depth GT com resolucao {depth_gt.shape[:2]}, RGB com '
                        f'{(resolucao_altura, resolucao_largura)}. Para treino pixel-a-pixel, '
                        'configure o render de depth com a mesma resolucao/FOV da monocular.'
                    )
                    self.depth_gt_shape_warned = True

                depth_gt_visual = self.criar_visualizacao_depth_gt(depth_gt)
            
            # ---- COMPENSACAO DA IMAGEM (IMU + ATITUDE) ----
            imagem_estabilizada, mascara_alpha, img_geometria = self.aplicar_compensacao_imu(
                cv_image,
                msg
            )
            estado_intervalo = self.capturar_estado_intervalo(rgb_stamp_s, depth_age_s)

            # ---- VISAO COMPUTACIONAL PARA DESVIO REATIVO ----
            if self.evasao_visual_ativa:
                visao_da_evasao, flow_interval = self.calcular_evasao_visual(
                    imagem_estabilizada,
                    mascara_alpha
                )
                self.registrar_intervalo_ground_truth(
                    imagem_estabilizada[:, :, :3],
                    depth_gt,
                    estado_intervalo,
                    flow_interval
                )
            else:
                self.prev_gray_avoidance = None
                self.prev_points_avoidance = None
                self.prev_depth_interval_ref = None
                visao_da_evasao = imagem_estabilizada[:, :, :3].copy()
            
            #cv2.imshow("Visão do Drone Original (Com tremor)", cv_image)
            #cv2.imshow("Visao do Drone Original (Com tremor) + a Geometria do Warping", img_geometria)
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

        Atualmente os recursos adicionais sao memmaps do dataset de intervalos depth/flow,
        abertos apenas quando save_ground_truth_dataset esta ativo e a primeira amostra
        sincronizada e salva. O flush garante que os buffers sejam descarregados.

        Fontes:
        [ROS 2 Node] https://docs.ros.org/en/humble/Concepts/Basic/About-Nodes.html
        [Python File Objects] https://docs.python.org/3/tutorial/inputoutput.html#reading-and-writing-files
        ==================================================================================
        """

        for memmap_array in (
            self.depth_gt_intervals,
            self.depth_gt_image_delta,
            self.depth_gt_depth_delta,
            self.depth_gt_depth_mask,
            self.depth_gt_flow_vectors,
        ):
            if memmap_array is not None:
                memmap_array.flush()
        self.atualizar_manifesto_depth_gt()
        super().destroy_node()
