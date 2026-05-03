import os
import time
import numpy as np
import math
import cv2
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleAttitude
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

        # Inicializa a ponte de conversão ROS -> OpenCV
        self.bridge = CvBridge()

        self.camera_sub = self.create_subscription(
            Image, 
            '/world/baylands/model/x500_mono_cam_0/link/camera_link/sensor/camera/image',
            self.image_callback, 
            qos_profile_sensor_data)
        
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
        self.smooth_yaw = 0.0
        self.smooth_vx = 0.0
        self.smooth_vy = 0.0
        self.velocity_smooth_alpha = 0.3
        self.yaw_smooth_alpha = 0.3

        self.start_x = None
        self.start_y = None
        self.start_z = None

        self.waypoints_relativos = [
            [54.0, -24.0, -1.5],
            [48.0, -32.0, -2.0],
            [48.0, -40.0, -2.0],
            [48.0, -48.0, -2.0],
            [36.0, -36.0, -2.0],
            [24.0, -18.0, -2.0],
            [12.0, -9.0, -3.5],
            [0.0, 0.0, -5.0]
        ]
        
        self.lista_alvos_absolutos = []
        self.wp_atual_index = 0

        self.ciclos = 0
        self.voo_iniciado = False
        self.missao_concluida = False
        self.encerrando = False
        
        self.velocidade_maxima = 12.0  # Velocidade do vetor m/s
        self.raio_de_aceitacao = 2.0   # Raio de aceitação para mudar de waypoint
        self.max_lateral_acceleration = 8.0  # m/s² limite para aceleração lateral

        self.dt = 0.04  # (25Hz)
        self.timer = self.create_timer(self.dt, self.timer_callback)

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
    # Existe um timer rodando a 25Hz (0.04s) que envia o modo de controle (OffboardControlMode). 
    # Somente após 50 ciclos (2 segundos), o script emite a ordem para armar (arm()) e decolar.
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
                self.encerrando = True
                import threading
                threading.Thread(target=self.comando_exit).start()

        self.ciclos += 1

    # ==================================================================================
    # A função calcula a distância até o waypoint alvo. Se a distância for maior que 
    # a margem de corte (distancia_corte = 1.0 metro / ou 0.5 no código atual), ele converte 
    # a distância restante em um vetor de velocidade (v) normalizado. Quando a distância cai 
    # abaixo desse limite, o script simplesmente muda o alvo para o próximo ponto da lista.
    # ==================================================================================
    def navegar_por_waypoints(self):
        alvo_atual = self.lista_alvos_absolutos[self.wp_atual_index]
        target_x, target_y, target_z = alvo_atual[0], alvo_atual[1], alvo_atual[2]
        
        # Calcula a distância Euclidiana até o alvo
        pos_x = target_x - self.current_x
        pos_y = target_y - self.current_y
        pos_z = target_z - self.current_z
        distancia = math.sqrt(pos_x**2 + pos_y**2 + pos_z**2)
        
        vx, vy = 0.0, 0.0
        if self.smooth_yaw is None or self.smooth_yaw == 0.0:
            self.smooth_yaw = self.current_yaw
        
        is_ultimo_wp = (self.wp_atual_index == len(self.lista_alvos_absolutos) - 1)
        distancia_corte = 0.3 if is_ultimo_wp else self.raio_de_aceitacao

        # --- VELOCIDADE ADAPTATIVA BASEADA EM CURVATURA ---
        velocidade_maxima_atual = self.velocidade_maxima
        if not is_ultimo_wp:
            curvatura = self.calcular_curvatura_wp_atual()
            if curvatura > 0.5:
                velocidade_maxima_atual *= (1.0 - min(curvatura * 0.3, 0.5))

        # --- LÓGICA DE VELOCIDADE DINÂMICA PARA CADA WAYPOINT ---
        if distancia > distancia_corte:
            if is_ultimo_wp:
                dist_inicio_frenagem = velocidade_maxima_atual * 1.2
            else:
                dist_inicio_frenagem = velocidade_maxima_atual * 0.6
            velocidade_minima = velocidade_maxima_atual * 0.05
            
            if distancia > dist_inicio_frenagem:
                velocidade_dinamica = velocidade_maxima_atual
            else:
                proporcao = (distancia - distancia_corte) / (dist_inicio_frenagem - distancia_corte)
                proporcao = proporcao ** 2.0
                
                if is_ultimo_wp:
                    proporcao = proporcao ** 1.5 

                velocidade_dinamica = velocidade_minima + (velocidade_maxima_atual - velocidade_minima) * proporcao

            # Normalização do vetor de velocidade
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

        # --- LIMITAÇÃO DE ACELERAÇÃO LATERAL ---
        accel_x = (vx - self.smooth_vx) / self.dt
        accel_y = (vy - self.smooth_vy) / self.dt
        accel_lateral = math.sqrt(accel_x**2 + accel_y**2)
        if accel_lateral > self.max_lateral_acceleration:
            scale = self.max_lateral_acceleration / accel_lateral
            vx = self.smooth_vx + accel_x * scale * self.dt
            vy = self.smooth_vy + accel_y * scale * self.dt

        # --- FILTRAGEM DE VELOCIDADE ---
        self.smooth_vx += self.velocity_smooth_alpha * (vx - self.smooth_vx)
        self.smooth_vy += self.velocity_smooth_alpha * (vy - self.smooth_vy)

        # --- AJUSTE DE DIREÇÃO (YAW) COM LOOK-AHEAD ---
        if self.smooth_yaw is None or self.smooth_yaw == 0.0:
            self.smooth_yaw = self.current_yaw

        yaw_alvo = self.calcular_yaw_com_look_ahead(target_x, target_y)
        erro_yaw = math.atan2(math.sin(yaw_alvo - self.smooth_yaw), math.cos(yaw_alvo - self.smooth_yaw))
        yaw_gain = self.yaw_smooth_alpha * (0.8 if abs(erro_yaw) > 0.8 else 1.2)  # Aumentado para resposta mais rápida
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

    def calcular_curvatura_wp_atual(self):
        """ Calcula a curvatura (mudança de direção) no waypoint atual """
        if self.wp_atual_index == 0 or self.wp_atual_index >= len(self.lista_alvos_absolutos) - 1:
            return 0.0
        
        # Vetor do waypoint anterior para o atual
        prev_wp = self.lista_alvos_absolutos[self.wp_atual_index - 1]
        curr_wp = self.lista_alvos_absolutos[self.wp_atual_index]
        next_wp = self.lista_alvos_absolutos[self.wp_atual_index + 1]
        
        vec1_x = curr_wp[0] - prev_wp[0]
        vec1_y = curr_wp[1] - prev_wp[1]
        vec2_x = next_wp[0] - curr_wp[0]
        vec2_y = next_wp[1] - curr_wp[1]
        
        # Ângulo entre os vetores
        dot = vec1_x * vec2_x + vec1_y * vec2_y
        mag1 = math.sqrt(vec1_x**2 + vec1_y**2)
        mag2 = math.sqrt(vec2_x**2 + vec2_y**2)
        if mag1 == 0 or mag2 == 0:
            return 0.0
        cos_angle = dot / (mag1 * mag2)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle = math.acos(cos_angle)
        return angle / math.pi

    def calcular_yaw_com_look_ahead(self, target_x, target_y):
        """ Calcula yaw alvo com look-ahead para próximo waypoint """
        distancia_atual = math.sqrt((target_x - self.current_x)**2 + (target_y - self.current_y)**2)
        
        # Se próximo waypoint existe e estamos próximos do atual, mira no próximo
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
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = True
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
        
        time.sleep(4)
        self.force_disarm()
        time.sleep(1)
        os._exit(0)

    def desenhar_telemetria_geometria(self, frame_original, H, largura_out=640, altura_out=480):
        """ Desenha a moldura de corte estabilizada sobre a imagem real do sensor """
        # Criamos uma cópia da imagem original para servir de fundo
        canvas = cv2.resize(frame_original, (0, 0), fx=1, fy=1)
        cantos_saida = np.array([
            [0, 0], [largura_out, 0], [largura_out, altura_out], [0, altura_out]
        ], dtype='float32').reshape(-1, 1, 2)

        # Aplicamos a Homografia para saber onde esses pontos de 640x480 "moram" dentro da imagem original de 1280x960
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

    # ==================================================================================
    # O image_callback é chamado exatamente a cada novo frame (quadro) que a câmera do Gazebo gera e publica no tópico.
    # No modelo x500_mono_cam, são 30 imagens por segundo, portanto o image_callback será chamado 30 vezes por segundo.
    # Como a parte de Visão Computacional vai rodar dentro desse callback, o algoritmo precisa ser executado e finalizado em menos de 0.033 segundos (30 FPS).
    #
    # --- METODOLOGIA: GIMBAL VIRTUAL E ESTABILIZAÇÃO ELETRÔNICA ---
    # Para anular a inclinação física do drone, simulamos um Gimbal mecânico através
    # de Transformação de Perspectiva (Homografia). A matemática opera nos seguintes passos:
    # 1. Puxa os dados de Atitude do IMU (Roll e Pitch).
    # 2. Gera Matrizes de Rotação 3D (Rx e Rz) aplicando força na direção contrária ao movimento.
    # 3. Calcula a Homografia: H = K_zoom * R * K_inv, que achata a imagem num plano reto.
    #
    # As Fontes:
    # [ROS/cv_bridge] https://github.com/ros-perception/vision_opencv/tree/humble/cv_bridge
    # [OpenCV Homografia] https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html
    # [OpenCV Camera Matriz] https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    # ==================================================================================
    def image_callback(self, msg):
        resolucao_largura = msg.width
        resolucao_altura = msg.height
        # formato_ros = msg.encoding
        
        # self.get_logger().info(f'Frame Recebido - Resolução: {resolucao_largura}x{resolucao_altura} pixels | Formato: {formato_ros}')
        # Frame Recebido - Resolução: 1280x960 pixels | Formato: rgb8
        
        try:
            # Convertendo a mensagem do ROS para uma imagem OpenCV (Matriz NumPy BGR)
            cv_image = np.ones((resolucao_altura,resolucao_largura, 4),dtype=np.uint8, order='F') * 255
            cv_image[:,:,:3] = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # --- ESTABILIZAÇÃO DA IMAGEM (IMU) ---
            if hasattr(self, 'current_roll') and hasattr(self, 'current_pitch'):
                theta_x = self.current_pitch 
                theta_z = self.current_roll  
                
                # Matriz de rotação em X (Compensa o nariz subindo/descendo)
                Rx = np.array([
                    [1, 0, 0],
                    [0, math.cos(theta_x), -math.sin(theta_x)],
                    [0, math.sin(theta_x), math.cos(theta_x)]
                ])
                
                # Matriz de rotação em Z (Compensa a inclinação lateral)
                Rz = np.array([
                    [math.cos(theta_z), -math.sin(theta_z), 0],
                    [math.sin(theta_z), math.cos(theta_z), 0],
                    [0, 0, 1]
                ])
                R = Rx @ Rz 
                
                zoom = 1.25
                
                K_zoom = np.array([
                    [self.K[0,0] * zoom, 0, 320.0],
                    [0, self.K[1,1] * zoom, 240.0],
                    [0, 0, 1]
                ])
                K_inv = np.linalg.inv(self.K)

                # Calcula a Homografia
                H = K_zoom @ R @ K_inv 
                
                # Gerar o Plot da visualização compensada em tempo real
                img_geometria = self.desenhar_telemetria_geometria(cv_image, H)

                imagem_estabilizada = cv2.warpPerspective(
                    cv_image, 
                    H, 
                    (640, 480), 
                    flags=cv2.INTER_LINEAR, 
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0, 0)
                )
                mascara_alpha = imagem_estabilizada[:, :, 3]
            else:
                imagem_estabilizada = cv_image
                mascara_alpha = cv_image[:, :, 3]
            
            # --- AQUI ENTRA A LÓGICA DE VISÃO COMPUTACIONAL PARA DESVIO AINDA A SER DESENVOLVIDA ---
            
            #cv2.imshow("Visão do Drone Original (Com tremor)", cv_image)
            cv2.imshow("Visão do Drone Original com a Geometria do Warping", img_geometria)
            cv2.imshow("Visão do Drone Estabilizada (Usando IMU)", imagem_estabilizada)
            cv2.imshow("Mascara Alpha (Branco = Pixel Valido)", mascara_alpha)
            cv2.waitKey(1) # Necessário para o OpenCV atualizar a janela
        except Exception as e:
            self.get_logger().error(f'Erro na conversão da imagem: {e}')

    # =========================================================================================
    # O attitude_callback é acionado de forma assíncrona e em altíssima frequência (geralmente
    # entre 50Hz e 250Hz) sempre que o PX4 atualiza os dados do IMU/Giroscópio.
    # 
    # A função recebe a orientação espacial absoluta do drone em formato de Quaternions (w, x, y, z)
    # e aplica a conversão geométrica para extrair os ângulos de Euler (Roll e Pitch) em radianos.
    #
    # As Fontes:
    # https://github.com/PX4/px4_msgs/blob/main/msg/VehicleAttitude.msg
    # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    # =========================================================================================
    def attitude_callback(self, msg):
        w, x, y, z = msg.q[0], msg.q[1], msg.q[2], msg.q[3]
        
        # Fórmula de conversão para Roll
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        self.current_roll = math.atan2(sinr_cosp, cosr_cosp)

        # Fórmula de conversão para Pitch
        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1:
            self.current_pitch = math.copysign(math.pi / 2.0, sinp) # Trava em 90 graus
        else:
            self.current_pitch = math.asin(sinp)