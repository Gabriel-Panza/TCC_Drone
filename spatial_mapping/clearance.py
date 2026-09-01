"""Diagnostico geometrico offline; nao altera a grade nem aprova voo."""
import numpy as np


def _points(value):
    points = np.asarray(value, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError('pontos devem formar uma matriz Nx3 finita e nao vazia')
    return points


def point_to_path(point, points):
    """Mede a menor distancia de um ponto NED a uma polilinha NED."""
    points = _points(points)
    point = _points([point])[0]
    best = None
    for index, (a, b) in enumerate(zip(points, points[1:]) if len(points) > 1
                                    else [(points[0], points[0])]):
        delta = b - a
        square = float(np.dot(delta, delta))
        t = float(np.clip(np.dot(point - a, delta) / square, 0., 1.)) if square else 0.
        nearest = a + t * delta
        distance = float(np.linalg.norm(point - nearest))
        if best is None or distance < best['distance_m']:
            best = dict(distance_m=distance, segment_index=index, segment_fraction=t,
                        nearest_path_point_ned_m=nearest.tolist())
    return best


def path_voxel_clearance(points, voxels, resolution_m):
    """Distancia euclidiana minima da polilinha as CAIXAS fechadas dos voxels.

    Nao mede distancia aos centros. Em cada intervalo entre cruzamentos com
    faces, distancia quadratica e uma quadratica convexa: testa seu minimo
    analitico, inclusive os extremos. Sem passo de amostragem espacial.
    """
    points = _points(points)
    if not np.isfinite(resolution_m) or resolution_m <= 0:
        raise ValueError('resolucao deve ser finita e positiva')
    cells = np.asarray(sorted(tuple(v) for v in voxels), dtype=float)
    if not cells.size:
        return dict(distance_m=None, reason='no_inflated_voxels')
    if cells.ndim != 2 or cells.shape[1] != 3 or not np.isfinite(cells).all() or not np.equal(cells, np.floor(cells)).all():
        raise ValueError('voxels devem ter coordenadas inteiras Nx3')
    low, high = cells * resolution_m, (cells + 1) * resolution_m
    best = None
    segments = zip(points, points[1:]) if len(points) > 1 else [(points[0], points[0])]
    for index, (a, b) in enumerate(segments):
        delta = b - a
        # Six face crossings plus t=0,1; parallel faces add no breakpoints.
        times = [np.zeros(len(cells)), np.ones(len(cells))]
        for axis in range(3):
            if delta[axis] != 0:
                times.extend([np.clip((low[:, axis] - a[axis]) / delta[axis], 0., 1.),
                              np.clip((high[:, axis] - a[axis]) / delta[axis], 0., 1.)])
        times = np.sort(np.stack(times, axis=1), axis=1)
        best_square = np.full(len(cells), np.inf)
        best_t = np.zeros(len(cells))
        for j in range(times.shape[1] - 1):
            left, right = times[:, j], times[:, j + 1]
            mid = (left + right) * .5
            p = a + mid[:, None] * delta
            below, above = p < low, p > high
            active = below | above
            bound = np.where(below, low, high)
            linear = np.sum(np.where(active, (a - bound) * delta, 0.), axis=1)
            quadratic = np.sum(np.where(active, delta * delta, 0.), axis=1)
            t = np.clip(np.divide(-linear, quadratic, out=mid.copy(), where=quadratic > 0), left, right)
            p = a + t[:, None] * delta
            square = np.sum((p - np.clip(p, low, high)) ** 2, axis=1)
            better = square < best_square
            best_square[better], best_t[better] = square[better], t[better]
        nearest_index = int(np.argmin(best_square))
        distance = float(np.sqrt(best_square[nearest_index]))
        if best is None or distance < best['distance_m']:
            t = float(best_t[nearest_index])
            p = a + t * delta
            best = dict(distance_m=distance, segment_index=index, segment_fraction=t,
                        nearest_inflated_voxel=cells[nearest_index].astype(int).tolist(),
                        nearest_path_point_ned_m=p.tolist(),
                        nearest_box_point_ned_m=np.clip(p, low[nearest_index], high[nearest_index]).tolist())
    return best


def acceptance_entry_point(previous, waypoint, radius_m):
    """Primeira entrada na esfera ao seguir o segmento nominal (cenario, nao pose real)."""
    previous, waypoint = _points([previous, waypoint])
    if not np.isfinite(radius_m) or radius_m < 0:
        raise ValueError('raio deve ser finito e nao negativo')
    delta = previous - waypoint
    length = float(np.linalg.norm(delta))
    return (previous if length <= radius_m else waypoint + delta * radius_m / length).tolist()
