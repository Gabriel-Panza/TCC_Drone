# Roteiro curto para apresentacao

## Explicacao em linguagem de aluno

O problema que motivou esta etapa foi simples: um erro medio de profundidade
nao informa se o modelo declarou livre justamente o ponto ocupado por uma
arvore. Para testar a utilidade real da percepcao, cada pixel de profundidade
foi reprojetado em 3D, transformado para NED e integrado como raios livres e
extremos ocupados. Regioes sem evidencia continuam desconhecidas.

O A* foi usado como instrumento de avaliacao do mapa. Ele procura caminho
somente no componente observado e livre, depois simplifica e valida cada
segmento. O resultado e uma lista de posicoes; o PX4 continua controlando
atitude, motores e estabilidade.

O falso espaco livre e o erro mais perigoso porque autoriza um caminho por um
obstaculo que a percepcao deixou de representar. Por isso, melhorar MAE nao
basta. A profundidade ideal do Gazebo completou as dez runs e mostrou que a
geometria, o mapa e a interface A*--PX4 funcionam. O v19 passou o gate por
pixel, mas falhou disponibilidade e falso espaco livre no replay espacial. O
SITL monocular nao foi executado porque o protocolo foi definido para bloquear
o voo antes de transformar uma melhora media em risco.

Esse resultado negativo nao invalida o trabalho. Ele mostra onde esta o limite
da percepcao e sustenta a contribuicao central: liberar navegacao exige
validacao que chegue ao mapa e ao caminho, nao apenas ao pixel. As principais
limitacoes sao simulacao, um unico cenario, banda vertical estreita, conjunto
primario reutilizado como validacao de treino e ausencia de bateria monocular.
Os proximos passos sao coletar conjuntos independentes mais diversos,
quantificar incerteza e repetir o gate completo antes de qualquer voo.

## Perguntas provaveis

**O A* controla o drone?** Nao. Ele gera pontos de posicao; o PX4 executa o
controle de baixo nivel.

**Por que desconhecido nao e livre?** Porque ausencia de observacao nao e
evidencia de passagem segura.

**Por que o v19 foi rejeitado sem colisao no melhor replay?** Porque falhou
outros requisitos definidos antes do teste: disponibilidade e falso espaco
livre.

**Por que usar profundidade do Gazebo?** Para isolar e validar geometria,
mapeamento e planejamento sem misturar erro monocular.

**A banda vertical de 0,1 m e segura?** Ela e uma decisao experimental para o
corredor de voo nivelado e nao generaliza para voo 3D livre; a margem horizontal
permanece separada.

**Houve vazamento?** Nao houve sobreposicao com os gradientes, mas a run
primaria foi usada na validacao do treino e nao pode ser chamada de teste
independente. Isso foi registrado como limitacao.

**Qual e a contribuicao cientifica?** Um protocolo e uma evidencia de que
qualidade por pixel nao garante representacao espacial nem caminho seguro.
