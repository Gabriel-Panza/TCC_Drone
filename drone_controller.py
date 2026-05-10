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
    """
    ======================================================================================
    Inicializa o nó principal de controle autônomo em ROS 2, criando os publishers,
    subscribers, parâmetros de missão, matriz intrínseca da câmera e variáveis de estado.
    
    O nó publica mensagens de controle Offboard para o PX4, envia setpoints de trajetória
    pelo tópico /fmu/in/trajectory_setpoint, envia comandos de veículo pelo tópico
    /fmu/in/vehicle_command e recebe posição local, atitude e imagem da câmera simulada.
    
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

        self.prev_gray_avoidance = None
        self.prev_points_avoidance = None
        self.obstacle_risk = 0.0
        self.avoid_lateral_body = 0.0
        self.avoid_brake = 0.0
        self.avoid_side_memory = 1.0
        self.avoid_side_lock_count = 0
        self.avoid_side_lock_frames = 10
        self.avoidance_smooth_alpha = 0.55
        self.avoidance_max_lateral_speed = 6.0
        self.avoidance_max_brake = 0.35
        self.avoidance_trigger_risk = 0.08
        self.raio_finalizacao = 1.5
        self.raio_desativa_evasao_final = 4.0
        self.max_lateral_acceleration = 8.0

        self.start_x = None
        self.start_y = None
        self.start_z = None

        self.waypoints_relativos = [
            [-25.0, 25.0, -1.75],
            [-50.0, 70.0, -1.75],
            [0.0, 0.0, -1.75]
        ]
        
        self.lista_alvos_absolutos = []
        self.wp_atual_index = 0

        self.ciclos = 0
        self.voo_iniciado = False
        self.missao_concluida = False
        self.encerrando = False
        
        self.velocidade_maxima = 12.0               # Velocidade do vetor m/s
        self.raio_de_aceitacao = 4.5                # Raio de aceitação para mudar de waypoint
        
        self.zona_frenagem_curva = 6.0
        self.angulo_curva_forte = math.radians(45)

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
        
        A Fonte: 
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
        
        Fontes:
        [PX4 Offboard Mode - TrajectorySetpoint] https://docs.px4.io/main/en/flight_modes/offboard
        [PX4 ROS 2 Offboard Control] https://docs.px4.io/main/en/ros2/offboard_control
        [PX4 VehicleLocalPosition - NED] https://docs.px4.io/main/en/msg_docs/VehicleLocalPosition
        ==================================================================================
        """
        
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
        distancia_corte = self.raio_finalizacao if is_ultimo_wp else self.raio_de_aceitacao

        # --- VELOCIDADE ADAPTATIVA BASEADA EM CURVATURA ---
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

                # Quanto mais perto do waypoint, mais reduz a velocidade
                t = (distancia - self.raio_de_aceitacao) / (
                    self.zona_frenagem_curva - self.raio_de_aceitacao
                )
                t = max(0.0, min(1.0, t))

                # Smoothstep: transição suave, sem queda brusca de velocidade
                t = t * t * (3.0 - 2.0 * t)

                velocidade_maxima_atual = (
                    velocidade_segura_curva +
                    (self.velocidade_maxima - velocidade_segura_curva) * t
                )

        # --- LÓGICA DE VELOCIDADE DINÂMICA PARA CADA WAYPOINT ---
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

        # --- EVASAO REATIVA POR VISAO ---
        evasao_habilitada = (
            not self.missao_concluida and
            not (is_ultimo_wp and distancia < self.raio_desativa_evasao_final)
        )

        if evasao_habilitada and self.obstacle_risk > self.avoidance_trigger_risk:
            brake_scale = max(0.55, 1.0 - self.avoid_brake)
            vx *= brake_scale
            vy *= brake_scale

            # avoid_lateral_body > 0 significa desvio para a direita do drone.
            right_x = -math.sin(self.current_yaw)
            right_y = math.cos(self.current_yaw)
            vx += right_x * self.avoid_lateral_body
            vy += right_y * self.avoid_lateral_body

            velocidade_cmd = math.sqrt(vx**2 + vy**2)
            if velocidade_cmd > velocidade_maxima_atual:
                escala = velocidade_maxima_atual / velocidade_cmd
                vx *= escala
                vy *= escala

        # --- LIMITAÇÃO DE ACELERAÇÃO LATERAL ---
        accel_x = (vx - self.smooth_vx) / (self.dt * 4)
        accel_y = (vy - self.smooth_vy) / (self.dt * 4)
        accel_lateral = math.sqrt(accel_x**2 + accel_y**2)
        if accel_lateral > self.max_lateral_acceleration:
            scale = self.max_lateral_acceleration / accel_lateral
            vx = self.smooth_vx + accel_x * scale * (self.dt * 4)
            vy = self.smooth_vy + accel_y * scale * (self.dt * 4)

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
        self.get_logger().info("Encerrando a missão em 2s... Iniciando pouso!")
        
        # Altera o eixo Z do waypoint alvo final para o chão
        self.lista_alvos_absolutos[self.wp_atual_index][2] = 0.0
        
        time.sleep(2)
        self.force_disarm()
        time.sleep(1)
        os._exit(0)

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

    def detectar_pontos_evasao(self, gray, valid_mask):
        """
        Detecta pontos em bordas/cantos dentro da regiao valida da imagem estabilizada.

        A ideia vem de VO semi-denso/edge-based: nao reconstruimos a cena inteira,
        apenas rastreamos pontos visuais bons o suficiente para estimar risco local.
        """

        altura, largura = gray.shape
        roi_mask = np.zeros_like(valid_mask)
        roi_mask[int(altura * 0.18):int(altura * 0.90), int(largura * 0.08):int(largura * 0.92)] = 255
        roi_mask = cv2.bitwise_and(roi_mask, valid_mask)

        edges = cv2.Canny(gray, 60, 160)
        feature_mask = cv2.bitwise_and(cv2.dilate(edges, None, iterations=1), roi_mask)
        if cv2.countNonZero(feature_mask) < 150:
            feature_mask = roi_mask

        return cv2.goodFeaturesToTrack(
            gray,
            maxCorners=180,
            qualityLevel=0.01,
            minDistance=8,
            blockSize=7,
            mask=feature_mask
        )

    def suavizar_comando_evasao(self, risk, lateral_body, brake):
        """Aplica filtro passa-baixa para evitar comandos bruscos vindos da visao."""

        alpha = self.avoidance_smooth_alpha
        self.obstacle_risk += alpha * (risk - self.obstacle_risk)
        self.avoid_lateral_body += alpha * (lateral_body - self.avoid_lateral_body)
        self.avoid_brake += alpha * (brake - self.avoid_brake)

        if abs(self.avoid_lateral_body) > 0.05:
            self.avoid_side_memory = math.copysign(1.0, self.avoid_lateral_body)

    def escolher_lado_evasao(self, preferred_side, risk):
        """
        Mantem o mesmo lado por alguns frames para evitar zigue-zague nervoso.
        """

        if risk < self.avoidance_trigger_risk:
            self.avoid_side_lock_count = max(0, self.avoid_side_lock_count - 1)
            return preferred_side

        if self.avoid_side_lock_count > 0:
            self.avoid_side_lock_count -= 1
            return self.avoid_side_memory

        self.avoid_side_memory = preferred_side
        self.avoid_side_lock_count = self.avoid_side_lock_frames
        return preferred_side

    def calcular_risco_aparente_central(self, gray, valid_mask, debug):
        """
        Detecta obstaculo visual no corredor central mesmo quando o fluxo e pequeno.

        Um obstaculo exatamente na direcao de voo pode ficar perto do foco de expansao,
        onde o deslocamento optico ainda e baixo. Esse fallback usa densidade de bordas
        no centro da imagem e escolhe o lado visualmente mais livre.
        """

        altura, largura = gray.shape
        y1, y2 = int(altura * 0.18), int(altura * 0.90)
        cx1, cx2 = int(largura * 0.34), int(largura * 0.66)
        lx1, lx2 = int(largura * 0.08), int(largura * 0.42)
        rx1, rx2 = int(largura * 0.58), int(largura * 0.92)

        edges = cv2.Canny(gray, 55, 145)
        edges = cv2.bitwise_and(cv2.dilate(edges, None, iterations=1), valid_mask)

        def densidade(x1, x2):
            roi_edges = edges[y1:y2, x1:x2]
            roi_valid = valid_mask[y1:y2, x1:x2]
            area = max(cv2.countNonZero(roi_valid), 1)
            return cv2.countNonZero(roi_edges) / area

        central_density = densidade(cx1, cx2)
        left_density = densidade(lx1, lx2)
        right_density = densidade(rx1, rx2)

        risk = float(np.clip((central_density - 0.040) / 0.060, 0.0, 1.0))
        if risk <= 0.0:
            return 0.0, 0.0

        if abs(left_density - right_density) < 0.006:
            preferred_side = self.avoid_side_memory
        else:
            preferred_side = 1.0 if left_density > right_density else -1.0

        side = self.escolher_lado_evasao(preferred_side, risk)
        lateral_body = side * max(2.2, self.avoidance_max_lateral_speed * risk)
        cv2.rectangle(debug, (cx1, y1), (cx2, y2), (0, 165, 255), 2)
        cv2.putText(
            debug,
            f"centro={risk:.2f}",
            (cx1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 165, 255),
            1
        )

        return risk, lateral_body

    def calcular_evasao_visual(self, imagem_estabilizada, mascara_alpha):
        """
        Estima risco de colisao por fluxo optico e profundidade inversa relativa.

        Com camera monocular, a escala absoluta e ambigua. Por isso usamos o
        principio de profundidade inversa: durante o movimento, pontos mais
        proximos tendem a produzir fluxo radial maior na imagem. O resultado
        alimenta um campo repulsivo simples, nao um mapa 3D completo.
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
        apparent_risk, apparent_lateral = self.calcular_risco_aparente_central(gray, valid_mask, debug)

        if self.prev_gray_avoidance is None or self.prev_points_avoidance is None:
            self.prev_gray_avoidance = gray
            self.prev_points_avoidance = self.detectar_pontos_evasao(gray, valid_mask)
            brake = min(self.avoidance_max_brake, apparent_risk * self.avoidance_max_brake)
            self.suavizar_comando_evasao(apparent_risk, apparent_lateral, brake)
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
            brake = min(self.avoidance_max_brake, apparent_risk * self.avoidance_max_brake)
            self.suavizar_comando_evasao(apparent_risk, apparent_lateral, brake)
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
            brake = min(self.avoidance_max_brake, apparent_risk * self.avoidance_max_brake)
            self.suavizar_comando_evasao(apparent_risk, apparent_lateral, brake)
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

        # Proxy de profundidade inversa: fluxo radial positivo e centralizado.
        inverse_depth_score = np.clip((radial_flow - 0.35) / 4.5, 0.0, 1.0)
        point_risk = inverse_depth_score * central_weight
        point_risk *= speed_factor

        active = point_risk > 0.04
        if np.count_nonzero(active) < 6:
            risk = 0.0
            lateral_body = 0.0
        else:
            active_risk = point_risk[active]
            active_points = new[active]
            risk = float(np.clip(np.percentile(active_risk, 80) * 2.8, 0.0, 1.0))

            left = float(np.sum(active_risk[active_points[:, 0] < cx]))
            right = float(np.sum(active_risk[active_points[:, 0] >= cx]))
            balance = (right - left) / (right + left + 1e-6)

            if abs(balance) < 0.15:
                preferred_side = self.avoid_side_memory
            else:
                preferred_side = -math.copysign(1.0, balance)

            side = self.escolher_lado_evasao(preferred_side, risk)
            lateral_body = side * max(2.2, self.avoidance_max_lateral_speed * risk)

            for p0, p1, r in zip(old[active], active_points, active_risk):
                color = (0, 0, 255) if r > 0.25 else (0, 255, 255)
                cv2.arrowedLine(debug, tuple(p0.astype(int)), tuple(p1.astype(int)), color, 1, tipLength=0.3)

        if apparent_risk > risk or abs(apparent_lateral) > abs(lateral_body):
            risk = max(risk, apparent_risk)
            lateral_body = apparent_lateral if abs(apparent_lateral) > abs(lateral_body) else lateral_body

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
            (0, 255, 255),
            1
        )
        return debug

    def image_callback(self, msg):
        """
        ==================================================================================
        Processa cada frame recebido da câmera simulada e aplica estabilização eletrônica
        baseada na atitude do drone.
        
        A imagem ROS é convertida para matriz OpenCV por meio do cv_bridge. Em seguida, os
        ângulos atuais de pitch e roll, obtidos do VehicleAttitude, são usados para montar
        matrizes de rotação 3D. A homografia H = K_zoom * R * K_inv projeta a imagem como se
        houvesse um gimbal virtual compensando a inclinação física do drone.
        
        A função exibe três janelas principais: a imagem original com a geometria do warping,
        a imagem estabilizada e a máscara alpha que indica quais pixels de saída ainda possuem
        correspondência válida na imagem de entrada.
        
        Importante: esta estabilização reduz a tremedeira visual da câmera, mas não corrige a
        dinâmica física do voo. A redução do chacoalho do drone é tratada na navegação por
        waypoints, especialmente pela frenagem por curvatura e limitação de aceleração lateral.
        
        Fontes:
        [cv_bridge] https://docs.ros.org/en/jade/api/cv_bridge/html/python/
        [OpenCV Homography] https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html
        [OpenCV Camera Calibration] https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
        [OpenCV warpPerspective] https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html
        ==================================================================================
        """
        
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
            
            # --- VISAO COMPUTACIONAL PARA DESVIO REATIVO ---
            visao_de_evasao = self.calcular_evasao_visual(imagem_estabilizada, mascara_alpha)
            
            #cv2.imshow("Visão do Drone Original (Com tremor)", cv_image)
            cv2.imshow("Visão do Drone Original com a Geometria do Warping", img_geometria)
            #cv2.imshow("Visão do Drone Estabilizada (Usando IMU)", imagem_estabilizada)
            cv2.imshow("Detecção Reativa (Fluxo Optico)", visao_de_evasao)
            cv2.imshow("Mascara Alpha (Branco = Pixel Valido)", mascara_alpha)
            cv2.waitKey(1) # Necessário para o OpenCV atualizar a janela
        except Exception as e:
            self.get_logger().error(f'Erro na conversão da imagem: {e}')

    def attitude_callback(self, msg):
        """
        =========================================================================================
        Recebe a atitude estimada do drone e converte a orientação de quaternion para ângulos
        de Euler roll e pitch.
        
        O PX4 publica VehicleAttitude com quaternion no formato q(w, x, y, z), seguindo a
        convenção de Hamilton. A mensagem representa a rotação do corpo do drone no referencial
        FRD para o referencial NED. O script extrai apenas roll e pitch porque esses ângulos
        são usados para compensar a inclinação da câmera no gimbal virtual de image_callback().
        
        A conversão implementada segue as fórmulas usuais de quaternion para Euler, com trava
        de segurança no pitch quando o valor de asin ultrapassa o intervalo [-1, 1] por erro
        numérico. O yaw operacional usado na navegação vem do campo heading da posição local.
        
        Fontes:
        [PX4 VehicleAttitude] https://docs.px4.io/main/en/msg_docs/VehicleAttitude
        [MAVLink ATTITUDE_QUATERNION] https://mavlink.io/en/messages/common.html#ATTITUDE_QUATERNION
        [Conversão Quaternion-Euler] https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
        =========================================================================================
        """
        
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
