# TCC_Drone: Simulação e Controle Autônomo com ROS 2, Gazebo e PX4

Este projeto é parte de um Trabalho de Conclusão de Curso (TCC) focado no desenvolvimento de uma arquitetura de controle autônomo para Drones. O sistema permite o voo autônomo, por meio de um sistema reativo de evasão de obstáculos usando visão computacional.

A simulação de alta fidelidade é alcançada através da integração do controlador de voo **PX4 (SITL)** com o motor físico **Gazebo Harmonic**, enquanto toda a inteligência e controle de alto nível rodam sobre o **ROS 2 (Humble)**, comunicando-se via middleware **Micro XRCE-DDS Agent**.

## Ideia do Projeto e Funcionalidades

O objetivo principal é criar uma base modular e segura para navegação de drones em ambientes simulados complexos. As principais características do projeto incluem:

*   **Controle Offboard Avançado:** O drone decola, estabiliza e se move utilizando vetores de velocidade, com interpolação suave para evitar trancos e capotamentos na simulação.
*   **Evasão Reativa de Obstáculos:** Utilizando uma câmera monocular, o drone captura e processa imagens RGB em tempo real. Se um obstáculo for detectado à frente através de técnicas de Visão Computacional, ele calcula vetores de força lateral e vertical para frear e desviar automaticamente da colisão através de Campos Potenciais.

---

## Arquitetura do Código

O código foi estruturado de forma modular, dividindo as responsabilidades de rede (ROS 2) e lógica de controle em dois arquivos distintos.

### 1. `main.py` (Ponto de Entrada / Orquestrador)
É o arquivo executável do projeto. Ele é responsável por:
*   Inicializar o ambiente do ROS 2.
*   Instanciar o nó do controlador de voo (`DroneOffboardNode`).
*   Manter o programa vivo e rodando o `rclpy.spin()`, garantindo que os callbacks de sensores e atuadores ocorram continuamente.

### 2. `drone_controller.py` (O Cérebro / Model-Controller)
Este arquivo contém toda a matemática, física e comunicação com o PX4. Ele herda a classe Node do ROS 2 e é responsável por:
*   **Comunicação Bidirecional:** Publicar mensagens (`OffboardControlMode`, `TrajectorySetpoint`, `VehicleCommand`) e assinar sensores (VehicleLocalPosition, tópicos de imagem da câmera).
*   **Movimento Suave:** Gerenciar a diferença entre a "Posição Atual" e a "Posição Alvo", aplicando passos de interpolação baseados na velocidade do drone, operando sempre a 50Hz.
*   **Visão Computacional:** Utilizar o CvBridge para converter os dados brutos de imagem do ROS 2 em matrizes OpenCV (NumPy). Isso permite aplicar filtros visuais para extrair informações do ambiente, identificar obstáculos e modificar os setpoints de trajetória instantaneamente.

---

## 💻 Como Executar a Simulação

Para executar o ecossistema completo, são necessários **3 terminais** rodando simultaneamente em um ambiente Linux (ou WSL).

**Terminal 1: Iniciar o Simulador Gazebo + PX4**
```bash
cd ~/PX4-Autopilot
PX4_GZ_WORLD=baylands make px4_sitl gz_x500_mono_cam
```

**Terminal 2: O Agente Micro XRCE-DDS + A Ponte de Visão Computacional (ros_gz_bridge) + A Ponte do Ground Truth de Profundidade (ros_gz_bridge)**
```bash
source /opt/ros/humble/setup.bash

MicroXRCEAgent udp4 -p 8888 &

ros2 run ros_gz_bridge parameter_bridge /world/baylands/model/x500_mono_cam_0/link/camera_link/sensor/camera/image@sensor_msgs/msg/Image[gz.msgs.Image &

ros2 run ros_gz_bridge parameter_bridge /sim_depth_ground_truth@sensor_msgs/msg/Image[gz.msgs.Image &

wait
```

**Terminal 3: O Nó de Controle ROS 2**
```bash
cd ~/TCC_Drone
source /opt/ros/humble/setup.bash
source ~/TCC_Drone/ws_ros2/install/setup.bash
python3 main.py --ros-args \
  -p ground_truth_depth_topic:=/sim_depth_ground_truth \
  -p save_ground_truth_dataset:=true \
  -p ground_truth_save_every_n:=5
```
