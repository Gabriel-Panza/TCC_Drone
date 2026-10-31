import copy
import unittest
from scripts.evaluate_v21_native import spatial_rank


class V21SelectionTest(unittest.TestCase):
    def report(self,success=.9,false_free=.05,collisions=0):
        count=round(success*10)
        return dict(planning=dict(success_rate=success,successes=count,collision_free_successes=count-collisions),
            map=dict(false_free_rate=false_free),qualification=dict(checks=dict(frames=False,
                plan_success_rate=success>=.8,false_free_rate=false_free<=.1,
                collision_free_plans=bool(count) and collisions==0)))

    def metrics(self,mae=1.,edges=.2):
        return dict(mae_m=mae,dangerous_far_edge_rate_1m=edges,dangerous_near_edge_rate_1m=edges)

    def test_lower_mae_cannot_override_spatial_failure(self):
        safe=self.report()
        unsafe=self.report(false_free=.2)
        self.assertLess(spatial_rank(safe,self.metrics(mae=2.)),spatial_rank(unsafe,self.metrics(mae=.1)))

    def test_blocking_everything_is_not_collision_free_planning(self):
        blocked=self.report(success=0.,false_free=0.)
        partial=self.report(success=.6,false_free=.05)
        self.assertLess(spatial_rank(partial,self.metrics()),spatial_rank(blocked,self.metrics()))

    def test_frame_count_is_excluded_only_from_selection_proxy(self):
        small=self.report();large=copy.deepcopy(small)
        large["qualification"]["checks"]["frames"]=True
        self.assertEqual(spatial_rank(small,self.metrics()),spatial_rank(large,self.metrics()))
        self.assertFalse(small["qualification"]["checks"]["frames"])

    def test_edge_errors_precede_mae_when_spatial_results_tie(self):
        report=self.report()
        self.assertLess(spatial_rank(report,self.metrics(mae=2.,edges=.1)),
                        spatial_rank(report,self.metrics(mae=.2,edges=.3)))
