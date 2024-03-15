from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from sb3_contrib.common.maskable.utils import get_action_masks

from mqt.predictor import reward, rl
from mqt.predictor.rl.torch_layers import CustomCombinedExtractor

if TYPE_CHECKING:
    from qiskit import QuantumCircuit

logger = logging.getLogger("mqt-predictor")
PATH_LENGTH = 260


class Predictor:
    """The Predictor class is used to train a reinforcement learning model for a given figure of merit and device such that it acts as a compiler."""

    def __init__(
        self,
        figure_of_merit: reward.figure_of_merit,
        device_name: str,
        logger_level: int = logging.INFO,
        features: list[str] | str = "all",
    ):
        logger.setLevel(logger_level)

        self.model = None
        self.features = features
        self.env = rl.PredictorEnv(figure_of_merit, device_name, features)
        self.device_name = device_name
        self.figure_of_merit = figure_of_merit

    def load_model(self) -> None:
        self.model = rl.helper.load_model("model_" + self.figure_of_merit + "_" + self.device_name)

    def compile_as_predicted(
        self,
        qc: QuantumCircuit,
    ) -> tuple[QuantumCircuit, list[str]]:
        """Compiles a given quantum circuit such that the given figure of merit is maximized by using the respectively trained optimized compiler.

        Args:
            qc (QuantumCircuit: The quantum circuit to be compiled or the path to a qasm file containing the quantum circuit.

        Returns:
            tuple[QuantumCircuit, list[str]] | bool: Returns a tuple containing the compiled quantum circuit and the compilation information. If compilation fails, False is returned.
        """
        if not self.model:
            try:
                self.load_model()
            except Exception as e:
                msg = "Model cannot be loaded. Try to train a local model using 'Predictor.train_model(<...>)'"
                raise RuntimeError(msg) from e

        assert self.model
        obs, _ = self.env.reset(qc)  # type: ignore[unreachable]

        used_compilation_passes = []
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action_masks = get_action_masks(self.env)
            action, _ = self.model.predict(obs, action_masks=action_masks)
            action = int(action)
            action_item = self.env.action_set[action]
            used_compilation_passes.append(action_item["name"])
            obs, reward_val, terminated, truncated, info = self.env.step(action)
            self.env.state._layout = self.env.layout  # noqa: SLF001

        if not self.env.error_occured:
            return self.env.state, used_compilation_passes

        msg = "Error occurred during compilation."
        raise RuntimeError(msg)

    def train_model(
        self,
        timesteps: int = 1000,
        model_name: str = "model",
        verbose: int = 2,
        test: bool = False,
    ) -> None:
        """Trains all models for the given reward functions and device.

        Args:
            timesteps (int, optional): The number of timesteps to train the model. Defaults to 1000.
            model_name (str, optional): The name of the model. Defaults to "model".
            verbose (int, optional): The verbosity level. Defaults to 2.
            test (bool, optional): Whether to train the model for testing purposes. Defaults to False.
        """

        if test:
            n_steps = 100
            progress_bar = True
        else:
            n_steps = 2048
            progress_bar = True

        logger.debug("Start training for: " + self.figure_of_merit + " on " + self.device_name)
        env = rl.PredictorEnv(
            reward_function=self.figure_of_merit, device_name=self.device_name, features=self.features
        )
        policy_kwargs = {
            "features_extractor_class": CustomCombinedExtractor,
            "features_extractor_kwargs": {
                "out_dim": 64,
                "normalized_image": False,
                "out_channels": 1,  # 1 == no CNN
            },
            "share_features_extractor": True,
        }
        model = MaskablePPO(
            MaskableMultiInputActorCriticPolicy,
            env,
            verbose=verbose,
            tensorboard_log="./" + model_name + "_" + self.figure_of_merit + "_" + self.device_name,
            gamma=0.98,
            n_steps=n_steps,
            policy_kwargs=policy_kwargs,
        )
        model.learn(total_timesteps=timesteps, progress_bar=progress_bar)
        model.save(
            rl.helper.get_path_trained_model() / (model_name + "_" + self.figure_of_merit + "_" + self.device_name)
        )
