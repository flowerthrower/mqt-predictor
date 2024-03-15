from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Dict, Discrete, Sequence
from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import CheckMap, GatesInBasis
from qiskit.transpiler.runningpassmanager import TranspileLayout

from mqt.bench.devices import get_device_by_name
from mqt.predictor import reward, rl

logger = logging.getLogger("mqt-predictor")


class PredictorEnv(Env):  # type: ignore[misc]
    """Predictor environment for reinforcement learning."""

    def __init__(
        self,
        reward_function: reward.figure_of_merit = "expected_fidelity",
        device_name: str = "ionq_harmony",
        features: list[str] | str = "all",
    ):
        logger.info("Init env: " + reward_function)

        self.action_set = {}
        self.actions_synthesis_indices = []
        self.actions_layout_indices = []
        self.actions_routing_indices = []
        self.actions_mapping_indices = []
        self.actions_opt_indices = []
        self.used_actions: list[str] = []
        self.device = get_device_by_name(device_name)

        index = 0

        for elem in rl.helper.get_actions_synthesis():
            self.action_set[index] = elem
            self.actions_synthesis_indices.append(index)
            index += 1
        for elem in rl.helper.get_actions_layout():
            self.action_set[index] = elem
            self.actions_layout_indices.append(index)
            index += 1
        for elem in rl.helper.get_actions_routing():
            self.action_set[index] = elem
            self.actions_routing_indices.append(index)
            index += 1
        for elem in rl.helper.get_actions_opt():
            self.action_set[index] = elem
            self.actions_opt_indices.append(index)
            index += 1
        for elem in rl.helper.get_actions_mapping():
            self.action_set[index] = elem
            self.actions_mapping_indices.append(index)
            index += 1

        self.action_set[index] = rl.helper.get_action_terminate()
        self.action_terminate_index = index

        self.reward_function = reward_function
        self.action_space = Discrete(len(self.action_set.keys()))
        self.num_steps = 0
        self.last_fig_of_mer = None
        self.init_fig_of_mer = None
        self.layout = None

        qubit_num, _max_depth = self.device.num_qubits, 10000

        spaces = {
            "num_qubits": Discrete(128),
            "depth": Discrete(1000000),
            "program_communication": Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "critical_depth": Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "entanglement_ratio": Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "parallelism": Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "liveness": Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "directed_program_communication": Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "singleQ_gates_per_layer": Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "multiQ_gates_per_layer": Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "circuit": Sequence(
                Box(
                    low=0,
                    high=1,
                    shape=(
                        1,
                        qubit_num,
                        qubit_num,
                    ),
                    dtype=np.float_,
                ),
            ),
        }
        self.observation_space = Dict({k: v for k, v in spaces.items() if ("all" in features or k in features)})
        self.features = features
        self.filename = ""

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[Any, Any]]:
        """Executes the given action and returns the new state, the reward, whether the episode is done, whether the episode is truncated and additional information."""
        self.used_actions.append(str(self.action_set[action].get("name")))
        altered_qc = self.apply_action(action)
        if not altered_qc:
            return (
                rl.helper.create_feature_dict(self.state, self.features),
                0,
                True,
                False,
                {},
            )

        self.state: QuantumCircuit = altered_qc

        self.valid_actions = self.determine_valid_actions_for_state()
        if len(self.valid_actions) == 0:
            msg = "No valid actions left."
            raise RuntimeError(msg)

        # New: Reward for all mapped states
        if self.action_terminate_index in self.valid_actions:
            fig_of_mer = self.calculate_reward()

            #if self.init_fig_of_mer is None:
            #    self.init_fig_of_mer = fig_of_mer
            #    self.last_fig_of_mer = fig_of_mer

            if action == self.action_terminate_index:
                # penalty for high number of steps
                # discount_factor = 1 - (self.num_steps / 100)
                reward_val = fig_of_mer # * discount_factor
                done = True
            else:
                reward_val = fig_of_mer - self.last_fig_of_mer
                done = False

            self.last_fig_of_mer = fig_of_mer
        # OG: Only reward for final state
        #if action == self.action_terminate_index:
        #    reward_val = self.calculate_reward()
        #    done = True
        else:
            reward_val = 0
            done = False

        self.num_steps += 1

        # in case the Qiskit.QuantumCircuit has unitary or u gates in it, decompose them (because otherwise qiskit will throw an error when applying the BasisTranslator
        if self.state.count_ops().get("unitary"):
            self.state = self.state.decompose(gates_to_decompose="unitary")

        obs = rl.helper.create_feature_dict(self.state, self.features)
        return obs, reward_val, done, False, {}

    def calculate_reward(self, init_circ: QuantumCircuit | None = None) -> Any:
        """Calculates and returns the reward for the current state."""
        state = init_circ if init_circ else self.state
        if self.reward_function == "expected_fidelity":
            return reward.expected_fidelity(state, self.device)
        if self.reward_function == "critical_depth":
            return reward.crit_depth(state)
        if self.reward_function == "expected_success_probability":
            return reward.expected_success_probability(state, self.device)
        error_msg = f"Reward function {self.reward_function} not supported."
        raise ValueError(error_msg)
    
    def calculate_final_improvement(self) -> float:
        """Calculates and returns the final improvement."""
        return self.last_fig_of_mer - self.init_fig_of_mer

    def render(self) -> None:
        """Renders the current state."""
        print(self.state.draw())

    def reset(
        self,
        qc: Path | str | QuantumCircuit | None = None,
        seed: int | None = None,
        options: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> tuple[QuantumCircuit, dict[str, Any]]:
        """Resets the environment to the given state or a random state.

        Args:
            qc (Path | str | QuantumCircuit | None, optional): The quantum circuit to be compiled or the path to a qasm file containing the quantum circuit. Defaults to None.
            seed (int | None, optional): The seed to be used for the random number generator. Defaults to None.
            options (dict[str, Any] | None, optional): Additional options. Defaults to None.

        Returns:
            tuple[QuantumCircuit, dict[str, Any]]: The initial state and additional information.
        """
        super().reset(seed=seed)
        if isinstance(qc, QuantumCircuit):
            self.state = qc
        elif qc:
            self.state = QuantumCircuit.from_qasm_file(str(qc))
        else:
            self.state, self.filename = rl.helper.get_state_sample(self.device.num_qubits)

        # Compile and map the circuit using qiskits minimum opt level and mapping method
        init_state = transpile(
            circuits=self.state,
            basis_gates=self.device.basis_gates,
            coupling_map=self.device.coupling_map,
            layout_method="trivial",
            optimization_level=0,
        )
        # To calculate an initial reward to benchmark the improvement
        self.init_fig_of_mer = self.calculate_reward(init_state)
        self.last_fig_of_mer = self.init_fig_of_mer

        self.action_space = Discrete(len(self.action_set.keys()))
        self.num_steps = 0
        self.used_actions = []

        self.layout = None

        self.valid_actions = self.actions_opt_indices + self.actions_synthesis_indices

        self.error_occured = False
        return rl.helper.create_feature_dict(self.state, self.features), {}

    def action_masks(self) -> list[bool]:
        """Returns a list of valid actions for the current state."""
        return [action in self.valid_actions for action in self.action_set]

    def apply_action(self, action_index: int) -> QuantumCircuit | None:
        """Applies the given action to the current state and returns the altered state."""
        if action_index in self.action_set:
            action = self.action_set[action_index]
            if action["name"] == "terminate":
                return self.state
            if (
                action_index
                in self.actions_layout_indices + self.actions_routing_indices + self.actions_mapping_indices
            ):
                transpile_pass = action["transpile_pass"](self.device.coupling_map)
            elif action_index in self.actions_synthesis_indices:
                transpile_pass = action["transpile_pass"](self.device.basis_gates)
            else:
                transpile_pass = action["transpile_pass"]
            if action["origin"] == "qiskit":
                try:
                    if action["name"] == "QiskitO3":
                        pm = PassManager()
                        pm.append(
                            action["transpile_pass"](self.device.basis_gates, CouplingMap(self.device.coupling_map)),
                            do_while=action["do_while"],
                        )
                    else:
                        pm = PassManager(transpile_pass)
                    altered_qc = pm.run(self.state)
                except Exception:
                    logger.exception(
                        "Error in executing Qiskit transpile pass for {action} at step {i} for {filename}".format(
                            action=action["name"], i=self.num_steps, filename=self.filename
                        )
                    )

                    self.error_occured = True
                    return None
                if action_index in self.actions_layout_indices + self.actions_mapping_indices:
                    assert pm.property_set["layout"]
                    self.layout = TranspileLayout(
                        initial_layout=pm.property_set["layout"],
                        input_qubit_mapping=pm.property_set["original_qubit_indices"],
                        final_layout=pm.property_set["final_layout"],
                    )

            elif action["origin"] == "tket":
                try:
                    tket_qc = qiskit_to_tk(self.state)
                    for elem in transpile_pass:
                        elem.apply(tket_qc)
                    altered_qc = tk_to_qiskit(tket_qc)
                except Exception:
                    logger.exception(
                        "Error in executing TKET transpile  pass for {action} at step {i} for {filename}".format(
                            action=action["name"], i=self.num_steps, filename=self.filename
                        )
                    )
                    self.error_occured = True
                    return None

            else:
                error_msg = f"Origin {action['origin']} not supported."
                raise ValueError(error_msg)

        else:
            error_msg = f"Action {action_index} not supported."
            raise ValueError(error_msg)

        return altered_qc

    def determine_valid_actions_for_state(self) -> list[int]:
        """Determines and returns the valid actions for the current state."""
        check_nat_gates = GatesInBasis(basis_gates=self.device.basis_gates)
        check_nat_gates(self.state)
        only_nat_gates = check_nat_gates.property_set["all_gates_in_basis"]

        if not only_nat_gates:
            return self.actions_synthesis_indices + self.actions_opt_indices

        check_mapping = CheckMap(coupling_map=CouplingMap(self.device.coupling_map))
        check_mapping(self.state)
        mapped = check_mapping.property_set["is_swap_mapped"]

        if mapped and self.layout is not None:
            return [self.action_terminate_index, *self.actions_opt_indices]  # type: ignore[unreachable]

        if self.state._layout is not None:  # noqa: SLF001
            return self.actions_routing_indices

        # No layout applied yet
        return self.actions_mapping_indices + self.actions_layout_indices + self.actions_opt_indices
