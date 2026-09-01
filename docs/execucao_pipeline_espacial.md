# Execucao do pipeline espacial

Este roteiro separa validacao geometrica, planejamento e voo. Nao habilite a etapa seguinte enquanto a anterior apresentar eixos invertidos, profundidade sem escala ou obstaculos fora de posicao.

## 1. Dependencias

No ambiente ROS 2 do laboratorio:

```bash
cd ~/TCC_Drone
python3 -m pip install -r requirements.txt
python3 -m unittest tests.test_spatial_mapping -v
```

Confirme tambem:

```bash
python3 - <<'PY'
import cv2
import numpy as np
print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
print("CUDA no OpenCV:", cv2.cuda.getCudaEnabledDeviceCount())
PY
```

## 2. Simulador e topicos

Use os mesmos terminais do pipeline anterior para PX4, Gazebo, MicroXRCEAgent e as duas pontes de imagem. Inicie o simulador com:

```bash
cd ~/PX4-Autopilot
PX4_GZ_WORLD=baylands make px4_sitl gz_x500_mono_cam
```

O MicroXRCEAgent deve permanecer ativo entre as runs. Ao interromper uma run,
encerre e reinicie somente PX4 e Gazebo com
`~/TCC_Drone/scripts/cleanup_spatial_sim.sh`. Confirme antes de iniciar o no:

```bash
ros2 topic hz /world/baylands/model/x500_mono_cam_0/link/camera_link/sensor/camera/image
ros2 topic hz /sim_depth_ground_truth
```

## 3. Mapa parado com profundidade do Gazebo

Este modo valida somente projecao, pose, extrinsecos e ocupacao. Ele nao arma o drone e nao representa um resultado monocular.

```bash
cd ~/TCC_Drone
source /opt/ros/humble/setup.bash
source ~/TCC_Drone/ws_ros2/install/setup.bash
PYTHONNOUSERSITE=1 python3 main.py --ros-args \
  --params-file config/spatial_debug.yaml
```

Na janela superior, confirme:

- obstaculos a frente aparecem na direcao do yaw do drone;
- obstaculos a direita e a esquerda nao estao trocados;
- o chao nao aparece na altura de voo;
- o numero de voxels livres e ocupados cresce sem explodir;
- o manifesto usa `depth_source=ground_truth_debug`.

Se a montagem descrita no SDF nao estiver alinhada ao corpo, ajuste apenas os campos `camera_offset_*_m` e `camera_mount_*_deg` no YAML e repita esta etapa.

A rota fica em `spatial_waypoints_relative_m`, como uma lista plana de grupos `x, y, z` no referencial NED relativo ao ponto inicial. Pontos intermediarios podem ser inseridos para orientar a passagem por uma regiao que precisa ser observada. Use exatamente a mesma lista nos ensaios ideal e monocular.

Encerre com `Ctrl+C`. A run fica em `datasets/spatial_mapping/`.

Gere a inspeção tridimensional:

```bash
python3 estudos_e_analises/analisar_mapeamento_espacial.py \
  ~/TCC_Drone/datasets/spatial_mapping/run_AAAAMMDD_HHMMSS
```

Abra `spatial_map_3d.html` e confira o resumo em `spatial_summary.json`.

## 4. Voo controlado pelo mapa ideal

Esta etapa valida A*, waypoints e referenciais, mas ainda usa a profundidade do Gazebo. Ela deve ser identificada como ensaio de depuracao, nunca como resultado monocular.

```bash
PYTHONNOUSERSITE=1 python3 main.py --ros-args \
  --params-file config/spatial_debug.yaml \
  -p spatial_execute_path:=true
```

O modo novo envia apenas setpoints de posicao. O PX4 controla a dinamica do voo e os comandos reativos antigos nao alteram os setpoints.

Por padrao, `spatial_collect_legacy_metrics=false` desliga a estabilizacao e o fluxo optico antigos durante esta validacao. Os frames RGB, profundidade, pose, mapas e planos continuam sendo registrados pelo gravador espacial. Isso evita que um processamento que nao participa do voo atrase a atualizacao do mapa.

O planejador remove curvas discretas quando existe linha de visada inteiramente observada e livre. O caminho resultante e dividido em pontos com no maximo 6 m. Caminhos locais terminam 2,5 m antes da fronteira observada, e um novo plano e solicitado quando restam menos de 3 m. Um subcaminho menor que 1,5 m nao e executado, pois ficaria proximo do raio de aceitacao de 0,8 m. Um replanejamento antecipado so substitui o caminho atual quando acrescenta pelo menos 1 m de progresso. Se a posicao atual entrar na regiao inflada, ou se tres microcaminhos consecutivos forem rejeitados, o drone tenta recuar 1,5 m por posicoes seguras ja voadas. Quando ainda nao existe historico suficiente para o recuo, ele permanece parado e faz uma varredura de yaw de 45 graus para cada lado em seis segundos, ampliando a area observada antes da proxima tentativa. A amostragem `spatial_depth_stride=35` preserva a grade de 0,75 m com menor custo de integracao.

O mapa integra obstaculos ate 30 m, planeja subobjetivos em um raio local de 25 m e, na configuracao final, mantem como obstaculos os pontos na banda de 0,1 m ao redor da altura do corpo. Essa faixa concentra a ocupacao no corredor de voo e reduz a influencia do solo e das partes altas das copas. A projecao superior mostra 35 m para cada lado e informa quantos voxels livres e ocupados estao visiveis. A esfera livre ao redor do drone nunca remove um voxel com ocupacao confirmada, e a margem de seguranca usa distancia euclidiana na grade.

O YAML inicial usa uma rota curta ate `[-8, 8, -1.65]` e retorna ao inicio. Ela serve apenas para validar atualizacao em movimento e estabilidade. A rota entre arvores deve ser definida depois, com pontos intermediarios registrados e mantidos iguais nos ensaios ideal e monocular.

## 5. Profundidade monocular

O arquivo ONNX deve produzir profundidade metrica ou profundidade inversa calibrada. Uma saida apenas relativa nao pode ser usada como metros sem uma calibracao independente.

Terminal do estimador:

```bash
cd ~/TCC_Drone
source /opt/ros/humble/setup.bash
PYTHONNOUSERSITE=1 python3 monocular_depth_node.py --ros-args \
  --params-file config/monocular_depth_onnx.yaml \
  -p model_path:=/CAMINHO/ABSOLUTO/modelo_metrico.onnx
```

Confira o topico:

```bash
ros2 topic hz /monocular_depth
ros2 topic echo /monocular_depth --once
```

## 6. Mapa monocular sem voo

```bash
PYTHONNOUSERSITE=1 python3 main.py --ros-args \
  --params-file config/spatial_monocular.yaml
```

Compare no `events.jsonl` os campos `mae_m`, `rmse_m` e `abs_rel`. Inspecione tambem `estimated_map.npz`, `reference_map.npz` e `false_free_rate` no evento final.

## 7. Voo monocular com A*

Esta etapa esta **bloqueada no resultado final**. O comando e apenas
documental e nao deve ser executado enquanto o relatorio mantiver
`monocular_sitl_battery=not_authorized`:

```bash
PYTHONNOUSERSITE=1 python3 main.py --ros-args \
  --params-file config/spatial_monocular.yaml \
  -p spatial_execute_path:=true
```

## Pipeline legado reativo

Depois de iniciar PX4, Gazebo, MicroXRCEAgent e a ponte RGB, execute:

```bash
cd ~/TCC_Drone
source /opt/ros/humble/setup.bash
source ~/TCC_Drone/ws_ros2/install/setup.bash
PYTHONNOUSERSITE=1 python3 main.py --ros-args \
  -p navigation_mode:=legacy_reactive \
  -p ground_truth_depth_topic:=/sim_depth_ground_truth \
  -p save_ground_truth_dataset:=true
```

O fluxo requer imagem RGB, pose e atitude do PX4 e os topicos offboard. A
profundidade e exigida quando a coleta RGB--profundidade estiver habilitada.
As saidas ficam em `logs/voo_teste_*` e `datasets/spatial_mapping/run_*`.
Encerre ao concluir a rota ou diante de oscilacao, perda de altitude ou
aproximacao insegura; em seguida reinicie PX4 e Gazebo.

## Significado de sem controle reativo

Em `spatial_astar`, os comandos do fluxo optico legado nao alteram o voo. O
A* tambem nao controla motores nem atitude: ele produz pontos de posicao em
NED. O PX4 estabiliza o veiculo, controla o baixo nivel e acompanha esses
pontos. Sem controle reativo significa sem a lei visual antiga, nao ausencia
de controle.

## Parametros verticais finais

Os YAMLs finais usam banda de obstaculos e inflacao vertical de 0,1 m em
torno do corpo. A margem horizontal permanece independente e nao foi reduzida.

## Arquivos de cada run

- `manifest.json`: configuracao e estado de conclusao;
- `events.jsonl`: frames, metricas e planos;
- `frames/*.npz`: RGB, profundidades, intrinsecos e transformacao camera-NED;
- `estimated_map.npz`: mapa usado pelo A*;
- `reference_map.npz`: mapa ideal do Gazebo;
- `logs/voo_teste_*`: odometria absoluta, deltas e diagnosticos do planejamento.
