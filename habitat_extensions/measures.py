import gzip
import json
import pickle
from typing import Any, List, Union, Optional, Tuple

import numpy as np
from dtw import dtw
from habitat.core.dataset import Episode
from habitat.core.embodied_task import Action, EmbodiedTask, Measure
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import Simulator

from habitat.tasks.nav.nav import DistanceToGoal, Success
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat.utils.visualizations import fog_of_war
from habitat.utils.visualizations import maps as habitat_maps
from numpy import ndarray
from omegaconf import DictConfig
from habitat_extensions.task import RxRVLNCEDatasetV1



def euclidean_distance(
    pos_a: Union[List[float], ndarray], pos_b: Union[List[float], ndarray]
) -> float:
    return np.linalg.norm(np.array(pos_b) - np.array(pos_a), ord=2)


@registry.register_measure
class PathLength(Measure):
    """Path Length (PL)
    PL = sum(geodesic_distance(agent_prev_position, agent_position)
            over all agent positions.
    """

    cls_uuid: str = "path_length"

    def __init__(self, sim: Simulator, *args: Any, **kwargs: Any):
        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, **kwargs: Any):
        self._previous_position = self._sim.get_agent_state().position
        self._metric = 0.0

    def update_metric(self, *args: Any, **kwargs: Any):
        current_position = self._sim.get_agent_state().position
        self._metric += euclidean_distance(
            current_position, self._previous_position
        )
        self._previous_position = current_position


@registry.register_measure
class OracleNavigationError(Measure):
    """Oracle Navigation Error (ONE)
    ONE = min(geosdesic_distance(agent_pos, goal)) over all points in the
    agent path.
    """

    cls_uuid: str = "oracle_navigation_error"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid]
        )
        self._metric = float("inf")
        self.update_metric(task=task)

    def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        distance_to_target = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        self._metric = min(self._metric, distance_to_target)


@registry.register_measure
class OracleSuccess(Measure):
    """Oracle Success Rate (OSR). OSR = I(ONE <= goal_radius)"""

    cls_uuid: str = "oracle_success"

    def __init__(self, *args: Any, config: Any, **kwargs: Any):
        self._config = config
        self.success_distance = config.get("success_distance", 3.0)
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid]
        )
        self._metric = 0.0
        self.update_metric(task=task)

    def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        d = task.measurements.measures[DistanceToGoal.cls_uuid].get_metric()
        self._metric = float(self._metric or d < self.success_distance)


@registry.register_measure
class OracleSPL(Measure):
    """OracleSPL (Oracle Success weighted by Path Length)
    OracleSPL = max(SPL) over all points in the agent path.
    """

    cls_uuid: str = "oracle_spl"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        task.measurements.check_measure_dependencies(self.uuid, ["spl"])
        self._metric = 0.0

    def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        spl = task.measurements.measures["spl"].get_metric()
        self._metric = max(self._metric, spl)



@registry.register_measure
class StepsTaken(Measure):
    """Counts the number of times update_metric() is called. This is equal to
    the number of times that the agent takes an action. STOP counts as an
    action.
    """

    cls_uuid: str = "steps_taken"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, **kwargs: Any):
        self._metric = 0.0

    def update_metric(self, *args: Any, **kwargs: Any):
        self._metric += 1.0




@registry.register_measure
class NDTW(Measure):
    """NDTW (Normalized Dynamic Time Warping)
    ref: https://arxiv.org/abs/1907.05446
    """

    cls_uuid: str = "ndtw"

    def __init__(
        self, sim: Any, config: Any, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        self.dtw_func = dtw
        
        self.split = config.get("split", "val_unseen")
        gt_path_template = config.get(
            "gt_path", "data/datasets/rxr/{split}/{split}_{role}_gt.json.gz"
        )
        self.success_distance = config.get("success_distance", 3.0)
        
        roles = config.get("roles", ["guide"])

        if "{role}" in gt_path_template:
            self.gt_json = {}
            for role in roles:
                with gzip.open(
                    gt_path_template.format(split=self.split, role=role), "rt"
                ) as f:
                    self.gt_json.update(json.load(f))
        else:
            with gzip.open(
                gt_path_template.format(split=self.split), "rt"
            ) as f:
                self.gt_json = json.load(f)

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, episode: Any, task: Any = None, **kwargs: Any):
        self.locations = []
        
        ep_id = episode.episode_id
        if ep_id not in self.gt_json and str(ep_id) in self.gt_json:
            ep_id = str(ep_id)
            
        self.gt_locations = self.gt_json[ep_id]["locations"]
        
        self.update_metric(*args, episode=episode, task=task, **kwargs)

    def update_metric(self, *args: Any, episode: Any, task: Any = None, **kwargs: Any):
        current_position = self._sim.get_agent_state().position.tolist()
        
        if len(self.locations) == 0:
            self.locations.append(current_position)
        else:
            if current_position == self.locations[-1]:
                return
            self.locations.append(current_position)

        dtw_distance = self.dtw_func(
            self.locations, self.gt_locations, dist=euclidean_distance
        )[0]

        nDTW = np.exp(
            -dtw_distance
            / (len(self.gt_locations) * self.success_distance)
        )
        self._metric = nDTW


@registry.register_measure
class SDTW(Measure):
     """SDTW (Success Weighted be nDTW)
     ref: https://arxiv.org/abs/1907.05446
     """

     cls_uuid: str = "sdtw"

     def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
         return self.cls_uuid
     def reset_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
         task.measurements.check_measure_dependencies(
             self.uuid, [NDTW.cls_uuid, Success.cls_uuid]
         )
         self.update_metric(task=task)

     def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
         ep_success = task.measurements.measures[Success.cls_uuid].get_metric()
         nDTW = task.measurements.measures[NDTW.cls_uuid].get_metric()
         self._metric = ep_success * nDTW

