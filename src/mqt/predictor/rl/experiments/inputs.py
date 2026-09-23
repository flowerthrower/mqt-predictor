# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Frozen circuits and offline Boston calibration shared by all four rows."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from qiskit import QuantumCircuit
from qiskit.transpiler import Target
from qiskit_ibm_runtime.fake_provider import FakeBoston


def digest(data: bytes) -> str:
    """Return the SHA-256 identity of an input."""
    return hashlib.sha256(data).hexdigest()


class Inputs:
    """Verify the frozen inputs before loading any experiment circuits."""

    def __init__(self, path: Path) -> None:
        """Verify every archived circuit and the original calibration bytes."""
        self.path = path
        self.manifest = json.loads((path / "manifest.json").read_text())
        assert digest((path / "circuits.zip").read_bytes()) == self.manifest["archive_sha256"]
        with ZipFile(path / "circuits.zip") as archive:
            assert set(archive.namelist()) == set(self.manifest["circuits"])
            self.circuits = {name: archive.read(name) for name in archive.namelist()}
        for name, data in self.circuits.items():
            assert digest(data) == self.manifest["circuits"][name], name
        for name, expected in self.manifest["calibration"].items():
            assert digest(gzip.decompress((path / f"{name}.gz").read_bytes())) == expected, name
        assert len(self.names("train")) == 321
        assert len(self.names("test")) == 41
        train_hashes = {digest(self.circuits[name]) for name in self.names("train")}
        assert train_hashes.isdisjoint(digest(self.circuits[name]) for name in self.names("test"))

    def names(self, split: str) -> list[str]:
        """Keep the historical split and filenames."""
        return sorted(name for name in self.circuits if name.startswith(f"{split}/"))

    def circuit(self, name: str) -> QuantumCircuit:
        """Parse the original bytes without inferring qubit counts from filenames."""
        circuit = QuantumCircuit.from_qasm_str(self.circuits[name].decode())
        circuit.name = Path(name).stem
        return circuit


class FrozenBoston(FakeBoston):
    """Read the bundled snapshot instead of the installed SDK's snapshot."""

    def __init__(self, assets: Path) -> None:
        """Select the frozen snapshot directory."""
        self.assets = assets
        super().__init__()

    def _load_json(self, filename: str) -> dict[str, Any]:
        return json.loads(gzip.decompress((self.assets / f"{filename}.gz").read_bytes()))


def load_target(assets: Path) -> tuple[FrozenBoston, Target]:
    """Use the snapshot's physical gates and calibration in every compiler."""
    backend = FrozenBoston(assets)
    source = backend.target
    target = Target(
        description="ibm_boston_156_2026_04_17",
        num_qubits=source.num_qubits,
        dt=source.dt,
        qubit_properties=source.qubit_properties,
        granularity=source.granularity,
        min_length=source.min_length,
        pulse_alignment=source.pulse_alignment,
        acquire_alignment=source.acquire_alignment,
    )
    # Reset/measurement aliases and control flow are not compiler basis gates.
    for name in sorted(set(backend.configuration().basis_gates) | {"measure", "reset", "delay"}):
        target.add_instruction(source.operation_from_name(name), dict(source[name]))
    return backend, target
