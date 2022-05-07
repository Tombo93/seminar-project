from typing import List, Tuple

from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition

from pyrep.objects import ProximitySensor


class ReachTargetSimple(Task):
    def init_task(self) -> None:
        success_sensor = ProximitySensor("success")
        success_condition = DetectedCondition(
            obj=self.robot.arm.get_tip(), detector=success_sensor
        )
        self.register_success_conditions([success_condition])

    def init_episode(self, index: int) -> List[str]:
        return ["reach the red target", "reach the red sphere"]

    def variation_count(self) -> int:
        return 1

    def base_rotation_bounds(
        self,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

    def is_static_workspace(self) -> bool:
        return True
