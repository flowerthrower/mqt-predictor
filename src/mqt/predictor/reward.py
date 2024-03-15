from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, cast

import numpy as np

from qiskit.transpiler.passes import ASAPScheduleAnalysis
from qiskit.converters import circuit_to_dag

if TYPE_CHECKING:
    from qiskit.circuit import QuantumRegister, Qubit

from mqt.bench.utils import calc_supermarq_features

if TYPE_CHECKING:
    from qiskit import QuantumCircuit

    from mqt.bench.devices import Device

logger = logging.getLogger("mqt-predictor")

figure_of_merit = Literal["expected_fidelity", "critical_depth", "fidelity"]


def calc_qubit_index(qargs: list[Qubit], qregs: list[QuantumRegister], index: int) -> int:
    offset = 0
    for reg in qregs:
        if qargs[index] not in reg:
            offset += reg.size
        else:
            return int(offset + reg.index(qargs[index]))
    error_msg = f"Global qubit index for local qubit {index} index not found."
    raise ValueError(error_msg)


def crit_depth(qc: QuantumCircuit, precision: int = 10) -> float:
    """Calculates the critical depth of a given quantum circuit."""
    supermarq_features = calc_supermarq_features(qc)
    return cast(float, np.round(1 - supermarq_features.critical_depth, precision))


def expected_fidelity(qc: QuantumCircuit, device: Device, precision: int = 10) -> float:
    """Calculates the expected fidelity of a given quantum circuit on a given device.

    Args:
        qc (QuantumCircuit): The quantum circuit to be compiled.
        device(mqt.bench.Device): The device to be used for compilation.
        precision (int, optional): The precision of the returned value. Defaults to 10.

    Returns:
        float: The expected fidelity of the given quantum circuit on the given device.
    """
    res = 1.0
    for instruction, qargs, _cargs in qc.data:
        gate_type = instruction.name

        if gate_type != "barrier":
            assert len(qargs) in [1, 2]
            first_qubit_idx = calc_qubit_index(qargs, qc.qregs, 0)
            if device.name == "rigetti_aspen_m2":
                first_qubit_idx = (first_qubit_idx + 40) % 80

            if len(qargs) == 1:
                if gate_type == "measure":
                    specific_fidelity = device.get_readout_fidelity(first_qubit_idx)
                else:
                    specific_fidelity = device.get_single_qubit_gate_fidelity(gate_type, first_qubit_idx)
            else:
                second_qubit_idx = calc_qubit_index(qargs, qc.qregs, 1)
                if device.name == "rigetti_aspen_m2":
                    second_qubit_idx = (second_qubit_idx + 40) % 80
                specific_fidelity = device.get_two_qubit_gate_fidelity(gate_type, first_qubit_idx, second_qubit_idx)

            res *= specific_fidelity

    return cast(float, np.round(res, precision))

def gate_duration(node, device, qc):
    gate_type = node.op.name
    qargs = node.qargs

    if gate_type == "barrier": return 0
    
    assert len(qargs) in [1, 2]
    first_qubit_idx = calc_qubit_index(qargs, qc.qregs, 0)
    if device.name == "rigetti_aspen_m2":
        first_qubit_idx = (first_qubit_idx + 40) % 80

    if len(qargs) == 1:
        if gate_type == "measure":
            return device.get_readout_duration(first_qubit_idx)
        else:
            return device.get_single_qubit_gate_duration(gate_type, first_qubit_idx)
    else:
        second_qubit_idx = calc_qubit_index(qargs, qc.qregs, 1)
        if device.name == "rigetti_aspen_m2":
            second_qubit_idx = (second_qubit_idx + 40) % 80
        return device.get_two_qubit_gate_duration(gate_type, first_qubit_idx, second_qubit_idx)



def expected_success_probability(qc: QuantumCircuit, device: Device, precision: int = 10) -> float:
    """Calculates the expected success probability of a given quantum circuit on a given device.

    Args:
        qc (QuantumCircuit): The quantum circuit to be compiled.
        device(mqt.bench.Device): The device to be used for compilation.
        precision (int, optional): The precision of the returned value. Defaults to 10.

    Returns:
        float: The expected success probability of the given quantum circuit on the given device.
    """
    res = 1.0

    # Use qiskits scheduling transpile pass to insert idle times (delays)
    # to the circuit to account for the device's gate times
    scheduler = ASAPScheduleAnalysis()
    scheduler._get_node_duration = lambda node, _dag: gate_duration(node, device, qc)
    dag = circuit_to_dag(qc)
    qc = scheduler.run(dag)

    for instruction, qargs, _cargs in qc.data:
        gate_type = instruction.name

        if gate_type != "barrier":
            assert len(qargs) in [1, 2]
            first_qubit_idx = calc_qubit_index(qargs, qc.qregs, 0)
            if device.name == "rigetti_aspen_m2":
                first_qubit_idx = (first_qubit_idx + 40) % 80

            if len(qargs) == 1:
                if gate_type == "measure": # measurement reliability
                    specific_fidelity = device.get_readout_fidelity(first_qubit_idx)
                else: # gate reliability
                    specific_fidelity = device.get_single_qubit_gate_fidelity(gate_type, first_qubit_idx)
            else:
                second_qubit_idx = calc_qubit_index(qargs, qc.qregs, 1)
                if device.name == "rigetti_aspen_m2":
                    second_qubit_idx = (second_qubit_idx + 40) % 80
                specific_fidelity = device.get_two_qubit_gate_fidelity(gate_type, first_qubit_idx, second_qubit_idx)

            res *= specific_fidelity

    return cast(float, np.round(res, precision))
