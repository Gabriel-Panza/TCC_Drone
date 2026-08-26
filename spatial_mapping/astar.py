"""Planejamento A* em uma grade tridimensional discreta."""

from heapq import heappop, heappush
from itertools import combinations, product
from math import gcd, sqrt


Voxel = tuple[int, int, int]


class PathNotFoundError(RuntimeError):
    """Indica que o destino nao pode ser alcancado na grade fornecida."""


class AStar3D:
    """Encontra caminhos em voxels livres com conectividade 6, 18 ou 26."""

    def __init__(self, traversable_voxels, blocked_voxels=(), connectivity=26):
        if connectivity not in {6, 18, 26}:
            raise ValueError("connectivity deve ser 6, 18 ou 26")
        self.traversable = {tuple(voxel) for voxel in traversable_voxels}
        self.blocked = {tuple(voxel) for voxel in blocked_voxels}
        self.connectivity = connectivity
        self._neighbors = self._build_neighbor_offsets(connectivity)

    def plan(self, start, goal):
        """Retorna uma lista de voxels entre inicio e destino, incluindo ambos."""

        start = tuple(start)
        goal = tuple(goal)
        if start not in self.traversable or goal not in self.traversable:
            raise PathNotFoundError("inicio e destino devem pertencer ao espaco livre")
        if start in self.blocked or goal in self.blocked:
            raise PathNotFoundError("inicio ou destino esta bloqueado")

        frontier = [(self._heuristic(start, goal), 0.0, start)]
        came_from: dict[Voxel, Voxel] = {}
        cost_so_far = {start: 0.0}

        while frontier:
            _, current_cost, current = heappop(frontier)
            if current == goal:
                return self._reconstruct(came_from, current)
            if current_cost > cost_so_far[current]:
                continue

            for neighbor, move_cost in self._valid_neighbors(current):
                new_cost = current_cost + move_cost
                if new_cost >= cost_so_far.get(neighbor, float("inf")):
                    continue
                cost_so_far[neighbor] = new_cost
                came_from[neighbor] = current
                priority = new_cost + self._heuristic(neighbor, goal)
                heappush(frontier, (priority, new_cost, neighbor))

        raise PathNotFoundError("nenhum caminho encontrado")

    def reachable_from(self, start):
        """Retorna o componente livre alcancavel a partir do voxel inicial."""

        start = tuple(start)
        if start not in self.traversable or start in self.blocked:
            return set()

        reachable = {start}
        frontier = [start]
        while frontier:
            current = frontier.pop()
            for neighbor, _ in self._valid_neighbors(current):
                if neighbor in reachable:
                    continue
                reachable.add(neighbor)
                frontier.append(neighbor)
        return reachable

    def _valid_neighbors(self, voxel):
        for dx, dy, dz in self._neighbors:
            neighbor = (voxel[0] + dx, voxel[1] + dy, voxel[2] + dz)
            if neighbor not in self.traversable or neighbor in self.blocked:
                continue
            if not self._transition_is_clear(voxel, (dx, dy, dz)):
                continue
            yield neighbor, sqrt(dx * dx + dy * dy + dz * dz)

    def _transition_is_clear(self, current, offset):
        """Impede que movimentos diagonais atravessem quinas ocupadas."""

        moved_axes = [index for index, value in enumerate(offset) if value]
        if len(moved_axes) == 1:
            return True

        for subset_size in range(1, len(moved_axes)):
            for axes in combinations(moved_axes, subset_size):
                intermediate = list(current)
                for axis in axes:
                    intermediate[axis] += offset[axis]
                intermediate = tuple(intermediate)
                if intermediate not in self.traversable or intermediate in self.blocked:
                    return False
        return True

    @staticmethod
    def _build_neighbor_offsets(connectivity):
        offsets = []
        for offset in product((-1, 0, 1), repeat=3):
            nonzero = sum(component != 0 for component in offset)
            if nonzero == 0:
                continue
            if connectivity == 6 and nonzero > 1:
                continue
            if connectivity == 18 and nonzero > 2:
                continue
            offsets.append(offset)
        return offsets

    @staticmethod
    def _heuristic(current, goal):
        return sqrt(sum((a - b) ** 2 for a, b in zip(current, goal)))

    @staticmethod
    def _reconstruct(came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path


def compress_collinear_path(path):
    """Remove voxels intermediarios quando a direcao discreta nao muda."""

    path = [tuple(voxel) for voxel in path]
    if len(path) <= 2:
        return path

    compressed = [path[0]]
    previous_direction = None
    for index in range(1, len(path)):
        delta = tuple(
            path[index][axis] - path[index - 1][axis]
            for axis in range(3)
        )
        divisor = 0
        for component in delta:
            divisor = gcd(divisor, abs(component))
        direction = tuple(
            component // max(1, divisor)
            for component in delta
        )
        if previous_direction is not None and direction != previous_direction:
            compressed.append(path[index - 1])
        previous_direction = direction
    compressed.append(path[-1])
    return compressed
