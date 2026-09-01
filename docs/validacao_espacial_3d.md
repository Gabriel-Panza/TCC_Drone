# Validacao espacial 3D

## Papel no trabalho

A validacao espacial testa se os sinais de proximidade estudados produzem informacao util para uma decisao de navegacao. Ela nao substitui a analise das metricas e nao inclui o desenvolvimento de um controlador. O resultado esperado e um mapa de ocupacao e uma sequencia de pontos de passagem. O PX4 continua responsavel por executar esses pontos.

## Fluxo proposto

1. Receber a imagem RGB e uma estimativa densa de profundidade.
2. Sincronizar a imagem, a profundidade, a pose da camera e os parametros intrinsecos.
3. Reprojetar pixels validos para o referencial optico da camera.
4. Transformar os pontos para o referencial local do PX4.
5. Acumular raios livres e extremos ocupados em uma grade 3D.
6. Expandir os obstaculos pelo raio do drone e por uma margem de incerteza.
7. Executar A* apenas sobre o espaco observado como livre.
8. Converter o caminho discreto em pontos de passagem.
9. Comparar o resultado com um mapa ideal construido a partir da profundidade do Gazebo.

## Separacao das fontes de profundidade

- **Profundidade estimada:** entrada do mapa avaliado e unica fonte permitida na versao monocular.
- **Profundidade do Gazebo:** referencia para medir erro e construir o mapa ideal de comparacao.
- **Regra:** a referencia do Gazebo nao pode entrar no mapa estimado nem na escolha do caminho monocular.

## Dados que precisam ser registrados

- imagem RGB original;
- profundidade estimada por pixel;
- profundidade de referencia por pixel;
- matriz intrinseca da camera;
- transformacao rigida entre camera e corpo do drone;
- posicao e orientacao absolutas com timestamp;
- identificador e timestamp do frame;
- caminho planejado e tempo gasto pelo A*.

As execucoes atuais foram organizadas por deltas e nao preservam tudo que um mapa global exige. A nova avaliacao precisa registrar pose absoluta e profundidade por frame, mesmo que as analises tabulares continuem usando variacoes.

## Comparacoes

O mesmo planejador e os mesmos parametros devem ser usados em dois mapas:

1. mapa estimado a partir da percepcao monocular;
2. mapa ideal construido com a profundidade do simulador.

Essa separacao permite atribuir diferencas de caminho ao mapa, sem misturar mudancas do planejador ou do controle.

## Metricas

- MAE, RMSE e erro relativo da profundidade;
- precisao, sensibilidade e IoU das celulas ocupadas;
- taxa de espaco ocupado classificado como livre;
- erro de localizacao dos obstaculos;
- existencia de caminho e taxa de caminhos sem colisao;
- comprimento do caminho e distancia minima dos obstaculos;
- tempo de atualizacao do mapa e tempo do A*.

## Metodologia implementada

1. aquisicao sincronizada de RGB e profundidade;
2. inferencia monocular;
3. calibracao de escala e deslocamento;
4. reprojecao pinhole para a camera optica;
5. transformacao camera optica, corpo e NED;
6. integracao temporal dos raios;
7. marcacao de voxels livres e ocupados;
8. manutencao das observacoes desconhecidas;
9. faixa vertical de obstaculos em torno do corpo;
10. confirmacao temporal da evidencia;
11. expansao dos obstaculos pela margem de seguranca;
12. geracao do espaco observado e navegavel;
13. A* com conectividade tridimensional;
14. selecao de fronteiras quando o destino nao esta observado;
15. simplificacao por linha de visada conhecida e livre;
16. divisao do caminho em pontos;
17. validacao final dos segmentos;
18. publicacao de setpoints de posicao;
19. replanejamento;
20. frenagem, recuo e recuperacao;
21. registro de mapas, planos, estados e metricas.

## Estado da v1

- reprojecao, transformacao camera-NED, ocupacao e A* possuem testes sinteticos;
- o modo `ground_truth_debug` permite validar mapa e planejamento sem alegar visao monocular;
- o modo `monocular_topic` recebe profundidade metrica de um no independente;
- os mapas estimado e ideal recebem o mesmo A* em cada tentativa;
- somente o caminho estimado pode ser enviado ao PX4;
- `spatial_execute_path=false` impede o armamento durante a primeira inspecao;
- cada run salva frames sincronizados, mapas, planos e metricas;
- cada tentativa registra tempo, comprimento e sucesso do A*, e a analise verifica o caminho estimado contra o mapa ideal final;
- a referencia do Gazebo completou 10 de 10 runs na rota fixa;
- nenhuma configuracao monocular v12--v19 passou o gate espacial completo;
- candidatos v19 sem colisao no replay primario ainda reprovaram
  disponibilidade de caminho e falso espaco livre;
- a run primaria v19 foi validacao durante o treino, embora nao tenha
  participado dos gradientes, e nao constitui teste independente;
- os quatro conjuntos secundarios nao foram executados porque todos os
  candidatos falharam na etapa primaria;
- a bateria SITL monocular permaneceu bloqueada.
