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

## Ordem de implementacao

1. Validar reprojecao e transformacoes com cenas sinteticas.
2. Construir um mapa de um frame usando profundidade do Gazebo.
3. Acumular varios frames usando a pose do drone.
4. Comparar o mapa acumulado com a geometria do simulador.
5. Integrar a fonte monocular de profundidade.
6. Executar o A* nos mapas ideal e estimado.
7. Somente depois publicar os pontos de passagem para o PX4.
