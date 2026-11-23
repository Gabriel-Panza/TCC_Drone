# Reprodução ao vivo no Baylands

O ensaio inicia o PX4 SITL e o Gazebo para voar a rota no cenário Baylands. A câmera e a pose são produzidas durante a execução. Em ambos os modos, os objetivos relativos ao ponto inicial são `(-25, 25, -1.5)`, `(-50, 70, -1.5)` e `(-25, 25, -1.5)` em metros NED. A decolagem usa 1,6 m e os objetivos usam 1,5 m.

| Modo | Fonte do caminho | Papel da profundidade de referência |
|---|---|---|
| `ground_truth_debug` | A* no mapa construído com profundidade do Gazebo | Fornece a profundidade ao planejador |
| `monocular_topic` | A* no mapa construído com a profundidade estimada pela v21 a partir do RGB | Veta caminhos propostos pelo monocular |

O procedimento usa o código atual do repositório. Para reproduzir os números do artigo, use o [replay offline da versão congelada](../README.md#3-replay-monocular-offline).

## Preparação em uma máquina nova

O ambiente de referência foi Ubuntu 22.04 no WSL2, Python 3.10, ROS 2 Humble e Gazebo Harmonic 8.10.0. São necessários a ponte `ros_gz_bridge` compatível com Harmonic, `cv_bridge`, `colcon`, e `MicroXRCEAgent` disponível no `PATH`. Use uma sessão gráfica do WSL para abrir o Gazebo e o mapa NED.

### PX4 e cenário

Use a árvore do PX4 em `~/PX4-Autopilot` no commit `e2708705a8ca625854f1129fd9dde2ae8ae16d59`. O submódulo `Tools/simulation/gz` correspondente usa o commit `3eb05f716a81bca316ab5771f53e509a07dce3a6`.

Em uma cópia limpa do PX4 já preparada para SITL, inicialize os submódulos e aplique o patch de cenário uma vez. Ele acrescenta `/sim_depth_ground_truth`, alinha as câmeras RGB e de profundidade, e configura o vento utilizado. Se a instalação do laboratório já tiver o patch, pule a aplicação. [Patch do cenário](baylands-sitl.patch)

```bash
cd ~/PX4-Autopilot
git submodule update --init --recursive
git -C Tools/simulation/gz apply --check ~/TCC_Drone/docs/baylands-sitl.patch
git -C Tools/simulation/gz apply ~/TCC_Drone/docs/baylands-sitl.patch
```

### Interfaces ROS e Python

Se o overlay do laboratório já estiver compilado em `ws_ros2/install`, reutilize-o. Em uma cópia nova do projeto, prepare as interfaces `px4_msgs` e o ambiente Python com acesso aos pacotes ROS do sistema. O OpenCV com suporte gráfico é necessário para a janela NED.

```bash
cd ~/TCC_Drone
source /opt/ros/humble/setup.bash
mkdir -p ws_ros2/src
git clone https://github.com/PX4/px4_msgs.git ws_ros2/src/px4_msgs
git -C ws_ros2/src/px4_msgs checkout a83514f5982da4e07755c0e5e2add12f0daead33
(cd ws_ros2 && colcon build --packages-select px4_msgs)

python3.10 -m venv --system-site-packages models/runtime_env
source models/runtime_env/bin/activate
python -m pip install numpy==1.26.4 opencv-python==4.11.0.86 onnxruntime==1.23.2 'plotly>=6.1,<7.0'
```

Nos próximos terminais de execução, ative `models/runtime_env/bin/activate` se estiver usando esse ambiente para as dependências do controlador. O executor carrega o ROS 2 e o overlay antes de iniciar os nós. As dependências extras de análise estão em [requirements.txt](../requirements.txt).

### Modelo monocular v21

A v21 usa o ONNX do [pacote de reprodução da entrega](https://github.com/Gabriel-Panza/TCC_Drone/releases/download/tcc-entrega-v1.0/monocular_final_inputs.tar.gz). O modelo já está instalado na máquina do laboratório. Em uma instalação nova, salve o pacote como `downloads/monocular_final_inputs.tar.gz` na raiz do projeto, extraia somente o modelo e confira o SHA-256.

```bash
cd ~/TCC_Drone
mkdir -p models
model=models/depth_anything_v2_metric_baylands_vits_v21_native640_686x518_fp32.onnx
tar -xOzf downloads/monocular_final_inputs.tar.gz model.onnx > "$model"
printf '%s  %s\n' b11b5ae85910e9650b0b3d97bf3d857e3a097b1230948738cd69a4e0b83066a6 "$model" | sha256sum -c -
```

## Executar e conferir

Siga a [preparação dos dois terminais](../README.md#preparar-a-simulação). Deixe o agente e as pontes no Terminal 1. No Terminal 2, execute [ground truth](../README.md#1-run-ground-truth) ou [monocular v21](../README.md#2-run-monocular-na-mesma-rota). Execute um modo por vez.

O caminho `output=` identifica a pasta da tentativa. `summary.tsv` precisa indicar `mission_complete=true` e `outcome=completed`. O `events.jsonl` do dataset indicado na coluna `dataset` registra três eventos `global_goal_reached`, com índices 0, 1 e 2, e um `mission_complete`. O `controller.log` registra o planejamento e os vetos. O `px4_gazebo.log` identifica o ULog e registra o pouso.

## Registro das execuções

### Ground truth

A bateria de 31/08/2026 (`20260831_082359_ground_truth_debug`) registrou **10/10 missões completas e 30 objetivos**. O comando de uma tentativa no README usa os três objetivos da mesma rota.

### Monocular v21

A execução de 23/11/2026 (`20261123_115109_monocular_topic`) registrou **uma missão completa** no perfil ao vivo com veto de referência. O dataset foi `run_20261123_115136_854794` e o ULog foi `2026-11-23/14_51_26.ulg`.

| Verificação | Resultado |
|---|---|
| Saída do executor | `exit_code=0`, `mission_complete=true`, `outcome=completed` |
| Objetivos | 0, 1 e 2 atingidos |
| Tempo da decolagem concluída ao último objetivo | 195,3 s de relógio |
| PX4 durante a missão | Offboard, sem *failsafe* registrado no ULog |
| Maior intervalo entre heartbeats | 0,112 s no emissor e 0,172 s nos registros do PX4 |
| Finalização | Pouso e desarme registrados pelo PX4 |

O perfil [v21 ao vivo](../config/spatial_monocular_v21.yaml) usa margens horizontais de 1,00 m para o monocular e 1,25 m para a referência. O controlador usa o heartbeat a 25 Hz em uma thread própria. Esse voo demonstra a execução completa com a configuração ao vivo. Os **70,59%** da entrega correspondem aos **12 de 17 pedidos de planejamento do replay offline**. Uma missão completa ao vivo não estabelece uma taxa de sucesso.
