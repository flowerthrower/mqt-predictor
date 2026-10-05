# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Graph observations and previous-action inputs for the experiments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn
from torch.nn import functional

from mqt.predictor.rl.gnn import GLOBAL_FEATURE_DIM, NODE_OPERATION_NAMES, GNNFeaturesExtractor, GNNObservationWrapper

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from gymnasium import spaces
    from numpy.typing import NDArray
    from qiskit import QuantumCircuit

    from mqt.predictor.rl.gnn import GNNMaskableMultiInputActorCriticPolicy, GNNMaskablePPO, GraphBatch

    from .environment import ExperimentEnv


class NormalizedGNNObservationWrapper(GNNObservationWrapper):
    """Keep the environment's normalized sizes in the experiment's graph input."""

    def _update_graph_observation(self, observation: dict[str, Any]) -> None:
        super()._update_graph_observation(observation)
        self.graph_observation["global_features"][0, :2] = torch.as_tensor([
            observation["num_qubits"].item(),
            observation["depth"].item(),
        ])


class PreviousActionObservationWrapper(NormalizedGNNObservationWrapper):
    """Append the previous action's one-hot vector, or all zeros after reset."""

    def __init__(self, env: ExperimentEnv) -> None:
        """Use the experiment's ordered action registry for the context vector."""
        super().__init__(env)
        self.action_count = int(cast("spaces.Discrete", env.action_space).n)
        assert list(env.action_set) == list(range(self.action_count))
        self.previous_action: int | None = None

    def reset(
        self,
        qc: Path | str | QuantumCircuit | None = None,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, NDArray[np.generic]], dict[str, Any]]:
        """Clear the previous action before refreshing the initial graph."""
        self.previous_action = None
        return super().reset(qc, seed=seed, options=options)

    def step(self, action: int) -> tuple[dict[str, NDArray[np.generic]], float, bool, bool, dict[str, Any]]:
        """Record the action before refreshing its resulting graph."""
        self.previous_action = int(action)
        return super().step(action)

    def _update_graph_observation(self, observation: dict[str, Any]) -> None:
        super()._update_graph_observation(observation)
        features = self.graph_observation["global_features"]
        extra = features.new_zeros((1, self.action_count))
        if self.previous_action is not None:
            extra[0, self.previous_action] = 1
        cast("Any", self.graph_observation).global_features = torch.cat((features, extra), dim=1)


class PreviousActionInputLayer(nn.Module):
    """Keep the original matrix operation and add a zero-initialized action contribution."""

    def __init__(self, original: nn.Linear, action_count: int) -> None:
        """Keep the existing layer and add the previous-action weights."""
        super().__init__()
        self.original = original
        self.previous_action = nn.Linear(action_count, original.out_features, bias=False)
        nn.init.zeros_(self.previous_action.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Add the previous-action contribution to the original layer output."""
        width = self.previous_action.in_features
        return self.original(features[:, :-width]) + self.previous_action(features[:, -width:])


class PreviousActionFeaturesExtractor(GNNFeaturesExtractor):
    """Use the existing graph encoder with previous-action inputs to its shared trunk."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        hidden_dim: int = 128,
        num_conv_wo_resnet: int = 3,
        num_resnet_layers: int = 5,
        dropout_p: float = 0.2,
        *,
        bidirectional: bool = True,
        action_count: int,
    ) -> None:
        """Extend the first shared layer without changing the graph encoder."""
        super().__init__(
            observation_space,
            hidden_dim,
            num_conv_wo_resnet,
            num_resnet_layers,
            dropout_p,
            bidirectional=bidirectional,
        )
        self.action_count = action_count
        self.trunk[0] = PreviousActionInputLayer(cast("nn.Linear", self.trunk[0]), action_count)

    def forward(self, observations: GraphBatch) -> torch.Tensor:
        """Extract shared actor and critic features, including the previous action."""
        scalars = observations["node_scalars"]
        one_hot = functional.one_hot(observations["gate_indices"], num_classes=len(NODE_OPERATION_NAMES)).to(
            scalars.dtype
        )
        embedding = self.encoder(
            torch.cat((one_hot, scalars), dim=1),
            observations["edge_index"],
            observations.batch,
            observations.num_graphs,
        )
        global_features = observations["global_features"].reshape(-1, GLOBAL_FEATURE_DIM + self.action_count)
        return self.trunk(torch.cat((embedding, global_features), dim=1))


def add_previous_action_inputs(model: GNNMaskablePPO, action_count: int) -> PreviousActionFeaturesExtractor:
    """Preserve saved parameters and Adam moments; add only a zero-initialized input path."""
    policy = cast("GNNMaskableMultiInputActorCriticPolicy", model.policy)
    assert action_count == cast("spaces.Discrete", model.action_space).n
    assert policy.share_features_extractor
    old = policy.features_extractor
    assert type(old) is GNNFeaturesExtractor
    kwargs: dict[str, Any] = {**model.policy_kwargs["features_extractor_kwargs"], "action_count": action_count}
    new = PreviousActionFeaturesExtractor(cast("spaces.Dict", model.observation_space), **kwargs).to(model.device)
    new.encoder = old.encoder
    new.trunk = old.trunk
    new.trunk[0] = PreviousActionInputLayer(cast("nn.Linear", new.trunk[0]), action_count).to(model.device)
    policy.features_extractor = policy.pi_features_extractor = policy.vf_features_extractor = new
    policy.features_extractor_class = PreviousActionFeaturesExtractor
    policy.features_extractor_kwargs = kwargs
    model.policy_kwargs = {
        **model.policy_kwargs,
        "features_extractor_class": PreviousActionFeaturesExtractor,
        "features_extractor_kwargs": kwargs,
    }

    # Match the parameter order used by GNNMaskableMultiInputActorCriticPolicy._build on load.
    encoder_parameters = list(new.encoder.parameters())
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    assert len(policy.optimizer.param_groups) == 2
    policy.optimizer.param_groups[0]["params"] = encoder_parameters
    policy.optimizer.param_groups[1]["params"] = [p for p in policy.parameters() if id(p) not in encoder_ids]
    return new
