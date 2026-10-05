# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Keep PPO close to the pretrained policy through a sampled KL reward penalty."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.save_util import load_from_zip_file

if TYPE_CHECKING:
    from mqt.predictor.rl.gnn import GNNMaskableMultiInputActorCriticPolicy

    from .observations import PreviousActionFeaturesExtractor, PreviousActionInputLayer


class TeacherPenalty(BaseCallback):
    """Subtract beta * log(pi(action|state) / teacher(action|state)) during rollouts."""

    def __init__(self, checkpoint: str, coefficient: float) -> None:
        """Use the fixed pretrained weights as the reference for every PPO rollout."""
        super().__init__()
        self.checkpoint = checkpoint
        self.coefficient = coefficient
        self.log_ratios: list[float] = []

    def _init_callback(self) -> None:
        self.teacher = deepcopy(cast("GNNMaskableMultiInputActorCriticPolicy", self.model.policy))
        extractor = cast("PreviousActionFeaturesExtractor", self.teacher.features_extractor)
        self.quality_features = extractor.quality_features
        extractor.quality_features = False
        cast("PreviousActionInputLayer", extractor.trunk[0]).quality = None
        _, parameters, _ = load_from_zip_file(self.checkpoint, load_data=False, device=self.model.device)
        assert parameters is not None
        self.teacher.load_state_dict(cast("Any", parameters["policy"]))
        self.teacher.set_training_mode(False)
        self.teacher.requires_grad_(False)

    def _on_rollout_start(self) -> None:
        self.log_ratios.clear()

    def _on_step(self) -> bool:
        graphs = self.locals["graph_observations"]
        if self.quality_features:
            graphs = [graph.clone() for graph in graphs]
            for graph in graphs:
                graph.global_features = graph.global_features[:, :-2]
        with torch.no_grad():
            observations, _ = self.teacher.obs_to_tensor(graphs)
            distribution = self.teacher.get_distribution(observations, action_masks=self.locals["action_masks"])
            actions = torch.as_tensor(self.locals["actions"], device=self.model.device).flatten()
            log_ratio = self.locals["log_probs"] - distribution.log_prob(actions)
        sampled = log_ratio.cpu().numpy()
        self.locals["rewards"] -= self.coefficient * sampled
        self.log_ratios.extend(sampled.tolist())
        return True

    def _on_rollout_end(self) -> None:
        self.logger.record("rollout/teacher_log_ratio", float(np.mean(self.log_ratios)))
