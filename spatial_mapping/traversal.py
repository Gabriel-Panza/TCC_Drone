"""Cobertura conservadora de segmentos: nao e o raycast de integracao."""
from itertools import product
from math import ceil, floor, isfinite

import numpy as np


def segment_voxels(start, end, resolution_m):
    """Todos os voxels fechados tocados pelo segmento, inclusive extremos.

    Divide a linha nos cruzamentos com planos da grade. Em cada intervalo
    aberto o voxel e constante; nos cruzamentos, inclui todas as celulas
    adjacentes. Assim, nao ha passo de amostragem que possa saltar uma celula.
    Linhas sobre uma face/aresta verificam ambos os lados. Custo proporcional
    aos planos cruzados (ordenacao O(K log K)), nao ao volume da caixa envolvente.
    """
    if not isfinite(resolution_m) or resolution_m <= 0:
        raise ValueError("resolucao deve ser finita e positiva")
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    if start.shape != (3,) or end.shape != (3,) or not np.isfinite([start, end]).all():
        raise ValueError("extremos devem conter tres coordenadas finitas")
    a, b = start / resolution_m, end / resolution_m
    delta = b - a
    events = {0.0, 1.0}
    for axis in range(3):
        if delta[axis] == 0:
            continue
        low, high = sorted((float(a[axis]), float(b[axis])))
        for boundary in range(ceil(low), floor(high) + 1):
            t = float((boundary - a[axis]) / delta[axis])
            if 0 < t < 1:
                events.add(t)

    voxels, seen = [], set()

    def visit(point):
        choices = []
        for coordinate in point:
            boundary = round(float(coordinate))
            # Only adds conservative contacts at floating-point ties. It is
            # not a physical clearance margin and cannot remove crossed cells.
            if abs(coordinate - boundary) <= 1e-10:
                choices.append((boundary, boundary - 1))
            else:
                choices.append((floor(coordinate),))
        for voxel in product(*choices):
            if voxel not in seen:
                seen.add(voxel)
                voxels.append(voxel)

    previous = None
    for t in sorted(events):
        if previous is not None:
            visit(a + ((previous + t) * .5) * delta)
        visit(a + t * delta)
        previous = t
    return voxels
