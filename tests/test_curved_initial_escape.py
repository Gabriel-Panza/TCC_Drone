import unittest
from types import SimpleNamespace
import numpy as np
from scripts.replay_curved_mapping import curved_entries
from spatial_mapping.occupancy import OccupancyGrid3D,OccupancyGridConfig

class CurvedEscapeTest(unittest.TestCase):
    def scene(self):
        free={(0,0,0),(1,0,0),(1,1,0),(1,2,0),(2,2,0)}
        grid=OccupancyGrid3D(OccupancyGridConfig(resolution_m=1.))
        grid._log_odds={v:-2. for v in free}
        return SimpleNamespace(grid=grid,_inflated_obstacles=lambda:free-{(2,2,0)})
    def test_recovers_bent_known_corridor_but_straight_crosses_unknown(self):
        nav=self.scene();start=np.array([.5,.5,.5]);goal=np.array([2.5,2.5,.5])
        self.assertTrue(set(nav.grid.segment_voxels(start,goal))-nav.grid.free_voxels())
        exits,_=curved_entries(nav,start,goal)
        self.assertTrue(exits)
        for _,path in exits:
            self.assertTrue(all(set(nav.grid.segment_voxels(a,b))<=nav.grid.free_voxels()
                for a,b in zip([start,*path[:-1]],path)))
    def test_unknown_or_pending_breaks_corridor_without_diagonal_shortcut(self):
        for pending in [False,True]:
            nav=self.scene()
            if pending:nav.grid._occupied_evidence_frames[(1,1,0)]={1}
            else:nav.grid._log_odds.pop((1,1,0))
            self.assertFalse(curved_entries(nav,[.5,.5,.5],[2.5,2.5,.5])[0])
    def test_search_respects_prefix_length_bound(self):
        self.assertFalse(curved_entries(self.scene(),[.5,.5,.5],[2.5,2.5,.5],max_length_m=2.)[0])
    def test_no_escape_exception_if_start_not_inflated(self):
        nav=self.scene();nav._inflated_obstacles=lambda:set()
        self.assertFalse(curved_entries(nav,[.5,.5,.5],[2.5,2.5,.5])[0])
