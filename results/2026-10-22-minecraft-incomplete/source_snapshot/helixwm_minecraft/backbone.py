"""Task-locked, randomly initialized official Dreamer learner for Minecraft.

This custom visible-navigation task uses RGB64 and a continuous six-dimensional
action wrapper. It is not the official Minecraft diamond benchmark. The pinned
V2 adapter supplies unchanged Dreamer losses and updates; this adapter changes
only the task identity and checkpoint envelope. No Walker state is accepted.
"""
from __future__ import annotations

import copy

from helixwm_v2.backbone import OfficialDreamerBackbone, UPSTREAM_COMMIT


TASK_NAMESPACE = "helixwm_minecraft_visible_navigation_v1"
BACKBONE_FORMAT = "helixwm_minecraft_backbone_v1"


class MinecraftBackbone(OfficialDreamerBackbone):
    def __init__(self, config, upstream_path, run_dir):
        config = copy.deepcopy(config)
        overrides = config.setdefault("overrides", {})
        if overrides.get("task", TASK_NAMESPACE) != TASK_NAMESPACE:
            raise ValueError("A Minecraft backbone cannot claim another task")
        overrides["task"] = TASK_NAMESPACE
        # The dmc_vision preset in the pinned release changes only the task,
        # unused DMC proprio flag, and runner ratio. Explicit custom task identity
        # avoids falsely labeling this RGB/continuous-action adapter as Walker.
        super().__init__(config, upstream_path, run_dir)
        self.task_namespace = TASK_NAMESPACE

    def snapshot(self):
        return {"format": BACKBONE_FORMAT, "task_namespace": TASK_NAMESPACE,
                "initialization": "random_seed_no_transfer",
                "backbone": super().snapshot()}

    def load(self, snapshot):
        if (snapshot.get("format") != BACKBONE_FORMAT or
                snapshot.get("task_namespace") != TASK_NAMESPACE or
                snapshot.get("initialization") != "random_seed_no_transfer"):
            raise ValueError("Only a task-locked Minecraft checkpoint can be loaded; Walker transfer is forbidden")
        super().load(snapshot["backbone"])
