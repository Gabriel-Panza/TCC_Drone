# Figuras espaciais que exigem os dados do laboratório

As figuras abaixo não podem ser reconstruídas a partir do clone público, pois os
quadros e mapas brutos não são versionados.

## Comparação de profundidade

Usar um quadro persistido do conjunto primário da v19 e apresentar, no mesmo
painel:

1. imagem RGB;
2. profundidade monocular v19, em metros;
3. profundidade alinhada do Gazebo, com a mesma escala de cores;
4. erro absoluto por pixel.

O painel deve identificar a execução e o índice do quadro. A escala espacial e
os limites de profundidade precisam ser iguais nas duas imagens.

## Exemplo de falso espaço livre

Usar o mesmo replay primário e selecionar um quadro ou snapshot no qual um voxel
ocupado no mapa de referência apareça livre no mapa monocular. Mostrar:

1. mapa de referência;
2. mapa monocular;
3. máscara da diferença, destacando apenas falso espaço livre;
4. caminho ou segmento afetado, quando houver.

A seleção deve vir diretamente da comparação entre mapas, e não de escolha
visual sem vínculo com a métrica. A legenda deve informar que o exemplo ilustra
o mecanismo da reprovação, sem representar sozinho a taxa agregada de 18,38%.

Depois da geração, copiar os dois arquivos para as pastas `figuras` do artigo
e da monografia, inserir as chamadas no resultado espacial e registrar seus
hashes junto ao relatório final.
