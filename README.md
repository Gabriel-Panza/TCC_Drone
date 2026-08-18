# TCC_Drone: Simulação e Controle Autônomo com ROS 2, Gazebo e PX4

> **Versão final do trabalho:** branch `reactive_approach`.

## Pontos centrais do repositório

O projeto possui dois fluxos principais:

1. **Execução do pipeline de voo e coleta:** [`main.py`](main.py). Esse arquivo inicia o nó ROS 2, executa a navegação reativa e coordena a coleta dos dados usados no estudo.
2. **Execução das métricas e dos experimentos:** [`estudos_e_analises/estudo_das_metricas.ipynb`](estudos_e_analises/estudo_das_metricas.ipynb). O notebook reúne o carregamento dos dados, as verificações de qualidade, o treinamento dos modelos, as comparações, a explicabilidade e a geração dos resultados apresentados no TCC.

Os diretórios `logs/` e `datasets/` usados nas 40 execuções finais não estão disponíveis no GitHub devido ao volume dos arquivos. Eles permanecem ignorados pelo Git e são necessários para refazer integralmente as análises a partir dos dados brutos. O notebook versionado mantém o código, as configurações e as saídas da análise final.

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

### 3. `estudos_e_analises/estudo_das_metricas.ipynb` (Análise dos Experimentos)

É o ponto central da análise feita após a coleta. O notebook:

* relaciona os diretórios correspondentes de `logs/` e `datasets/`;
* verifica sincronização, intervalos válidos, trajetórias e dados da IMU;
* treina a MLP e acompanha as curvas de perda;
* compara os marcos de crescimento do conjunto com validação e teste fixos;
* avalia várias sementes, modelos de árvores e ablação de grupos de entradas;
* executa a análise de eventos e a explicabilidade por gradientes;
* exporta os CSVs e gráficos usados no dashboard e na parte escrita.

---

## Como Executar a Simulação

Para executar o ecossistema completo, são necessários **3 terminais** rodando simultaneamente em um ambiente Linux (ou WSL).

**Terminal 1: Iniciar o Simulador Gazebo + PX4**
```bash
cd ~/PX4-Autopilot
PX4_GZ_WORLD=baylands make px4_sitl gz_x500_mono_cam
```

**Terminal 2: O Agente Micro XRCE-DDS + A Ponte de Visão Computacional (ros_gz_bridge)**
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
PYTHONNOUSERSITE=1 python3 main.py --ros-args \
  -p ground_truth_depth_topic:=/sim_depth_ground_truth \
  -p save_ground_truth_dataset:=true \
  -p ground_truth_depth_max_age_s:=0.08 \
  -p ground_truth_max_interval_s:=0.50 \
  -p use_dt_normalized_control:=false
```

O dataset descarta automaticamente frames RGB com timestamp repetido, pares que reutilizam
o mesmo frame de depth e intervalos temporais invalidos. Ao final de cada run, confira no
`manifest.json` os campos `quality_counters`: `rgb_frames_rejected_nonmonotonic` deve ser
baixo, e cada amostra salva deve ter `dt_s > 0` e `depth_dt_s > 0`.