from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import requests
from bqskit import MachineModel
from packaging import version
from pytket.architecture import Architecture
from pytket.circuit import Circuit, Node, Qubit
from pytket.passes import (
    CliffordSimp,
    FullPeepholeOptimise,
    PeepholeOptimise2Q,
    RemoveRedundancies,
    RoutingPass,
)
from pytket.placement import place_with_map
from qiskit import QuantumCircuit
from qiskit.circuit.equivalence_library import StandardEquivalenceLibrary
from qiskit.circuit.library import XGate, ZGate
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import CouplingMap, Layout, PassManager, TranspileLayout
from qiskit.transpiler.passes import (
    ApplyLayout,
    BasicSwap,
    BasisTranslator,
    Collect2qBlocks,
    CommutativeCancellation,
    CommutativeInverseCancellation,
    ConsolidateBlocks,
    CXCancellation,
    DenseLayout,
    Depth,
    EnlargeWithAncilla,
    FixedPoint,
    FullAncillaAllocation,
    GatesInBasis,
    InverseCancellation,
    MinimumPoint,
    Optimize1qGatesDecomposition,
    OptimizeCliffords,
    RemoveDiagonalGatesBeforeMeasure,
    SabreLayout,
    Size,
    StochasticSwap,
    TrivialLayout,
    UnitarySynthesis,
    VF2Layout,
    VF2PostLayout,
)
from qiskit.transpiler.passes.layout.vf2_layout import VF2LayoutStopReason
from sb3_contrib import MaskablePPO
from tqdm import tqdm

from mqt.bench.utils import calc_supermarq_features
from mqt.predictor import ml, reward, rl

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from qiskit.providers.models import BackendProperties

    from mqt.bench.devices import Device


if TYPE_CHECKING or sys.version_info >= (3, 10, 0):
    from importlib import metadata, resources
else:
    import importlib_metadata as metadata
    import importlib_resources as resources

import operator
import zipfile

from bqskit import compile as bqskit_compile
from bqskit.ir import gates
from qiskit import QuantumRegister
from qiskit.passmanager import ConditionalController
from qiskit.transpiler.preset_passmanagers import common
from qiskit_ibm_runtime.fake_provider import FakeGuadalupe, FakeMontreal, FakeQuito, FakeWashington

logger = logging.getLogger("mqt-predictor")


def qcompile(
    qc: QuantumCircuit | str,
    figure_of_merit: reward.figure_of_merit | None = "expected_fidelity",
    device_name: str | None = "ibm_washington",
    predictor_singleton: rl.Predictor | None = None,
) -> tuple[QuantumCircuit, list[str]]:
    """Compiles a given quantum circuit to a device optimizing for the given figure of merit.

    Args:
        qc (QuantumCircuit | str): The quantum circuit to be compiled. If a string is given, it is assumed to be a path to a qasm file.
        figure_of_merit (reward.reward_functions, optional): The figure of merit to be used for compilation. Defaults to "expected_fidelity".
        device_name (str, optional): The name of the device to compile to. Defaults to "ibm_washington".
        predictor_singleton (rl.Predictor, optional): A predictor object that is used for compilation to reduce compilation time when compiling multiple quantum circuits. If None, a new predictor object is created. Defaults to None.

    Returns:
        tuple[QuantumCircuit, list[str]] | bool: Returns a tuple containing the compiled quantum circuit and the compilation information. If compilation fails, False is returned.
    """

    if predictor_singleton is None:
        if figure_of_merit is None:
            msg = "figure_of_merit must not be None if predictor_singleton is None."
            raise ValueError(msg)
        if device_name is None:
            msg = "device_name must not be None if predictor_singleton is None."
            raise ValueError(msg)
        predictor = rl.Predictor(figure_of_merit=figure_of_merit, device_name=device_name)
    else:
        predictor = predictor_singleton

    return predictor.compile_as_predicted(qc)


def get_actions_opt() -> list[dict[str, Any]]:
    """Returns a list of dictionaries containing information about the optimization passes that are available."""
    return [
        {
            "name": "Optimize1qGatesDecomposition",
            "transpile_pass": [Optimize1qGatesDecomposition()],
            "origin": "qiskit",
        },
        {
            "name": "CXCancellation",
            "transpile_pass": [CXCancellation()],
            "origin": "qiskit",
        },
        {
            "name": "CommutativeCancellation",
            "transpile_pass": [CommutativeCancellation()],
            "origin": "qiskit",
        },
        {
            "name": "CommutativeInverseCancellation",
            "transpile_pass": [CommutativeInverseCancellation()],
            "origin": "qiskit",
        },
        {
            "name": "RemoveDiagonalGatesBeforeMeasure",
            "transpile_pass": [RemoveDiagonalGatesBeforeMeasure()],
            "origin": "qiskit",
        },
        {
            "name": "InverseCancellation",
            "transpile_pass": [InverseCancellation([XGate(), ZGate()])],
            "origin": "qiskit",
        },
        {
            "name": "OptimizeCliffords",
            "transpile_pass": [OptimizeCliffords()],
            "origin": "qiskit",
        },
        {
            "name": "Opt2qBlocks",
            "transpile_pass": [Collect2qBlocks(), ConsolidateBlocks()],
            "origin": "qiskit",
        },
        {
            "name": "PeepholeOptimise2Q",
            "transpile_pass": [PeepholeOptimise2Q()],
            "origin": "tket",
        },
        {
            "name": "CliffordSimp",
            "transpile_pass": [CliffordSimp()],
            "origin": "tket",
        },
        {
            "name": "FullPeepholeOptimiseCX",
            "transpile_pass": [FullPeepholeOptimise()],
            "origin": "tket",
        },
        {
            "name": "RemoveRedundancies",
            "transpile_pass": [RemoveRedundancies()],
            "origin": "tket",
        },
        {
            "name": "QiskitO3",
            "transpile_pass": lambda native_gate, coupling_map: [
                Collect2qBlocks(),
                ConsolidateBlocks(basis_gates=native_gate),
                UnitarySynthesis(basis_gates=native_gate, coupling_map=coupling_map),
                Optimize1qGatesDecomposition(basis=native_gate),
                CommutativeCancellation(basis_gates=native_gate),
                GatesInBasis(native_gate),
                ConditionalController(
                    common.generate_translation_passmanager(
                        target=None, basis_gates=native_gate, coupling_map=coupling_map
                    ).to_flow_controller(),
                    condition=lambda property_set: not property_set["all_gates_in_basis"],
                ),
                Depth(recurse=True),
                FixedPoint("depth"),
                Size(recurse=True),
                FixedPoint("size"),
                MinimumPoint(["depth", "size"], "optimization_loop"),
            ],
            "origin": "qiskit",
            "do_while": lambda property_set: (not property_set["optimization_loop_minimum_point"]),
        },
        {
            "name": "BQSKitO2",
            "transpile_pass": lambda circuit: bqskit_compile(circuit, optimization_level=2),
            "origin": "bqskit",
        },
    ]


def get_actions_final_optimization() -> list[dict[str, Any]]:
    """Returns a list of dictionaries containing information about the optimization passes that are available."""
    return [
        {
            "name": "VF2PostLayout",
            "transpile_pass": lambda device: VF2PostLayout(
                coupling_map=CouplingMap(device.coupling_map),
                properties=get_ibm_backend_properties_by_device_name(device.name),
            ),
            "origin": "qiskit",
        }
    ]


def get_actions_layout() -> list[dict[str, Any]]:
    """Returns a list of dictionaries containing information about the layout passes that are available."""
    return [
        {
            "name": "TrivialLayout",
            "transpile_pass": lambda device: [
                TrivialLayout(coupling_map=CouplingMap(device.coupling_map)),
                FullAncillaAllocation(coupling_map=CouplingMap(device.coupling_map)),
                EnlargeWithAncilla(),
                ApplyLayout(),
            ],
            "origin": "qiskit",
        },
        {
            "name": "DenseLayout",
            "transpile_pass": lambda device: [
                DenseLayout(coupling_map=CouplingMap(device.coupling_map)),
                FullAncillaAllocation(coupling_map=CouplingMap(device.coupling_map)),
                EnlargeWithAncilla(),
                ApplyLayout(),
            ],
            "origin": "qiskit",
        },
        {
            "name": "VF2Layout",
            "transpile_pass": lambda device: [
                VF2Layout(
                    coupling_map=CouplingMap(device.coupling_map),
                    properties=get_ibm_backend_properties_by_device_name(device.name),
                ),
                ConditionalController(
                    [
                        FullAncillaAllocation(coupling_map=CouplingMap(device.coupling_map)),
                        EnlargeWithAncilla(),
                        ApplyLayout(),
                    ],
                    condition=lambda property_set: property_set["VF2Layout_stop_reason"]
                    == VF2LayoutStopReason.SOLUTION_FOUND,
                ),
            ],
            "origin": "qiskit",
        },
    ]


def get_actions_routing() -> list[dict[str, Any]]:
    """Returns a list of dictionaries containing information about the routing passes that are available."""
    return [
        {
            "name": "BasicSwap",
            "transpile_pass": lambda device: [BasicSwap(coupling_map=CouplingMap(device.coupling_map))],
            "origin": "qiskit",
        },
        {
            "name": "RoutingPass",
            "transpile_pass": lambda device: [
                PreProcessTKETRoutingAfterQiskitLayout(),
                RoutingPass(Architecture(device.coupling_map)),
            ],
            "origin": "tket",
        },
        {
            "name": "StochasticSwap",
            "transpile_pass": lambda device: [StochasticSwap(coupling_map=CouplingMap(device.coupling_map))],
            "origin": "qiskit",
        },
    ]


def get_actions_mapping() -> list[dict[str, Any]]:
    """Returns a list of dictionaries containing information about the mapping passes that are available."""
    return [
        {
            "name": "SabreMapping",
            "transpile_pass": lambda device: [
                SabreLayout(coupling_map=CouplingMap(device.coupling_map), skip_routing=False),
            ],
            "origin": "qiskit",
        },
        {
            "name": "BQSKitMapping",
            "transpile_pass": lambda device: lambda bqskit_circuit: bqskit_compile(
                bqskit_circuit,
                model=MachineModel(
                    num_qudits=device.num_qubits,
                    gate_set=get_bqskit_native_gates(device),
                    coupling_graph=[(elem[0], elem[1]) for elem in device.coupling_map],
                ),
                with_mapping=True,
                optimization_level=2,
            ),
            "origin": "bqskit",
        },
    ]


def get_actions_synthesis() -> list[dict[str, Any]]:
    """Returns a list of dictionaries containing information about the synthesis passes that are available."""
    return [
        {
            "name": "BasisTranslator",
            "transpile_pass": lambda device: [
                BasisTranslator(StandardEquivalenceLibrary, target_basis=device.basis_gates)
            ],
            "origin": "qiskit",
        },
        {
            "name": "BQSKitSynthesis",
            "transpile_pass": lambda device: lambda bqskit_circuit: bqskit_compile(
                bqskit_circuit,
                model=MachineModel(bqskit_circuit.num_qudits, gate_set=get_bqskit_native_gates(device)),
                optimization_level=2,
            ),
            "origin": "bqskit",
        },
    ]


def get_action_terminate() -> dict[str, Any]:
    """Returns a dictionary containing information about the terminate pass that is available."""
    return {"name": "terminate"}


def get_state_sample(max_qubits: int | None = None) -> tuple[QuantumCircuit, str]:
    """Returns a random quantum circuit from the training circuits folder.

    Args:
        max_qubits (int, None): The maximum number of qubits the returned quantum circuit may have. If no limit is set, it defaults to None.

    Returns:
        tuple[QuantumCircuit, str]: A tuple containing the random quantum circuit and the path to the file from which it was read.
    """
    file_list = list(get_path_training_circuits().glob("*.qasm"))

    path_zip = get_path_training_circuits() / "training_data_compilation.zip"
    if len(file_list) == 0 and path_zip.exists():
        with zipfile.ZipFile(str(path_zip), "r") as zip_ref:
            zip_ref.extractall(get_path_training_circuits())

        file_list = list(get_path_training_circuits().glob("*.qasm"))
        assert len(file_list) > 0

    found_suitable_qc = False
    while not found_suitable_qc:
        rng = np.random.default_rng(10)
        random_index = rng.integers(len(file_list))
        num_qubits = int(str(file_list[random_index]).split("_")[-1].split(".")[0])
        if max_qubits and num_qubits > max_qubits:
            continue
        found_suitable_qc = True

    try:
        qc = QuantumCircuit.from_qasm_file(str(file_list[random_index]))
    except Exception:
        raise RuntimeError("Could not read QuantumCircuit from: " + str(file_list[random_index])) from None

    return qc, str(file_list[random_index])


def encode_circuit(qc: QuantumCircuit) -> NDArray[np.int_]:
    # Define a mapping from gate names to integers
    gate_dict = {x: i for i, x in enumerate(ml.helper.get_openqasm_gates())}
    gate_dict["ctr"] = len(gate_dict.keys())
    gate_dict["measure"] = len(gate_dict.keys())

    # Convert the circuit to a DAG and prepare the layers (wo barriers)
    dag = circuit_to_dag(qc)
    dag.remove_all_ops_named("barrier")
    layers = list(dag.multigraph_layers())

    # Create a look-up table for qubit indices (needed for multiple registers)
    q_idx_LUT = {qubit: idx for idx, qubit in enumerate(dag.qubits)}

    num_qubits, _max_depth = 11, 10000

    matrix = []  # np.zeros((num_qubits, num_qubits, max_depth), dtype=np.int_)
    for _i, tensor_op in enumerate(layers[1:-1]):
        layer = np.zeros((1, num_qubits, num_qubits), dtype=np.int_)
        for node in tensor_op:
            try:
                operation_name = node.op.name
            except Exception:
                continue
            if node.op.num_qubits == 1:  # single qubit gate
                q_idx = q_idx_LUT[node.qargs[0]]
                layer[0, q_idx, q_idx] = gate_dict[operation_name]
            else:  # multi qubit gate
                controls = []
                for qubit in node.qargs[:-1]:
                    q_idx = q_idx_LUT[qubit]  # control qubits
                    layer[0, q_idx, q_idx] = gate_dict["ctr"]
                    controls.append(q_idx)
                q_idx = q_idx_LUT[node.qargs[-1]]  # target qubit
                layer[0, q_idx, q_idx] = gate_dict[operation_name]
                for control in controls:
                    layer[0, control, q_idx] = gate_dict[operation_name]
                    layer[0, q_idx, control] = gate_dict["ctr"]
        matrix.append(layer)  # [:, :, i] = layer

    return np.array(matrix)


def create_feature_dict(qc: QuantumCircuit, features: list[str] | str = "all") -> dict[str, int | NDArray[np.float64]]:
    """Creates a feature dictionary for a given quantum circuit.

    Args:
        qc (QuantumCircuit): The quantum circuit for which the feature dictionary is created.

    Returns:
        dict[str, Any]: The feature dictionary for the given quantum circuit.
    """
    feature_dict = {
        "num_qubits": qc.num_qubits,
        "depth": qc.depth(),
    }

    supermarq_features = calc_supermarq_features(qc)
    # for all dict values, put them in a list each
    feature_dict["program_communication"] = np.array([supermarq_features.program_communication], dtype=np.float32)
    feature_dict["critical_depth"] = np.array([supermarq_features.critical_depth], dtype=np.float32)
    feature_dict["entanglement_ratio"] = np.array([supermarq_features.entanglement_ratio], dtype=np.float32)
    feature_dict["parallelism"] = np.array([supermarq_features.parallelism], dtype=np.float32)
    feature_dict["liveness"] = np.array([supermarq_features.liveness], dtype=np.float32)
    feature_dict["directed_program_communication"] = np.array(
        [supermarq_features.directed_program_communication], dtype=np.float32
    )
    feature_dict["singleQ_gates_per_layer"] = np.array([supermarq_features.singleQ_gates_per_layer], dtype=np.float32)
    feature_dict["multiQ_gates_per_layer"] = np.array([supermarq_features.multiQ_gates_per_layer], dtype=np.float32)
    feature_dict["circuit"] = encode_circuit(qc) if ("all" in features or "circuit" in features) else None

    return {k: v for k, v in feature_dict.items() if ("all" in features or k in features)}


def get_path_training_data() -> Path:
    """Returns the path to the training data folder used for RL training."""
    return Path(str(resources.files("mqt.predictor"))) / "rl" / "training_data"


def get_path_trained_model() -> Path:
    """Returns the path to the trained model folder used for RL training."""
    return get_path_training_data() / "trained_model"


def get_path_training_circuits() -> Path:
    """Returns the path to the training circuits folder used for RL training."""
    return get_path_training_data() / "training_circuits"


def load_model(model_name: str) -> MaskablePPO:
    """Loads a trained model from the trained model folder.

    Args:
        model_name (str): The name of the model to be loaded.

    Returns:
        MaskablePPO: The loaded model.
    """
    path = get_path_trained_model()

    if Path(path / (model_name + ".zip")).exists():
        return MaskablePPO.load(path / (model_name + ".zip"))
    logger.info("Model does not exist. Try to retrieve suitable Model from GitHub...")
    try:
        mqtpredictor_module_version = metadata.version("mqt.predictor")
    except ModuleNotFoundError:
        error_msg = (
            "Could not retrieve version of mqt.predictor. Please run 'pip install . or pip install mqt.predictor'."
        )
        raise RuntimeError(error_msg) from None

    headers = None
    if "GITHUB_TOKEN" in os.environ:
        headers = {"Authorization": f"token {os.environ['GITHUB_TOKEN']}"}

    version_found = False
    response = requests.get("https://api.github.com/repos/cda-tum/mqt-predictor/tags", headers=headers)

    if not response:
        error_msg = "Querying the GitHub API failed. One reasons could be that the limit of 60 API calls per hour and IP address is exceeded."
        raise RuntimeError(error_msg)

    available_versions = [elem["name"] for elem in response.json()]

    for possible_version in available_versions:
        if version.parse(mqtpredictor_module_version) >= version.parse(possible_version):
            url = "https://api.github.com/repos/cda-tum/mqt-predictor/releases/tags/" + possible_version
            response = requests.get(url, headers=headers)
            if not response:
                error_msg = "Suitable trained models cannot be downloaded since the GitHub API failed. One reasons could be that the limit of 60 API calls per hour and IP address is exceeded."
                raise RuntimeError(error_msg)

            response_json = response.json()
            if "assets" in response_json:
                assets = response_json["assets"]
            elif "asset" in response_json:
                assets = [response_json["asset"]]
            else:
                assets = []

            for asset in assets:
                if model_name in asset["name"]:
                    version_found = True
                    download_url = asset["browser_download_url"]
                    logger.info("Downloading model from: " + download_url)
                    handle_downloading_model(download_url, model_name)
                    break

        if version_found:
            break

    if not version_found:
        error_msg = "No suitable model found on GitHub. Please update your mqt.predictor package using 'pip install -U mqt.predictor'."
        raise RuntimeError(error_msg) from None

    return MaskablePPO.load(path / model_name)


def handle_downloading_model(download_url: str, model_name: str) -> None:
    """Downloads a trained model from the given URL and saves it to the trained model folder.

    Args:
        download_url (str): The URL from which the model is downloaded.
        model_name (str): The name of the model to be downloaded.
    """
    logger.info("Start downloading model...")

    r = requests.get(download_url)
    total_length = int(r.headers.get("content-length"))  # type: ignore[arg-type]
    fname = str(get_path_trained_model() / (model_name + ".zip"))

    with (
        Path(fname).open(mode="wb") as f,
        tqdm(
            desc=fname,
            total=total_length,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar,
    ):
        for data in r.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)
    logger.info(f"Download completed to {fname}. ")


class PreProcessTKETRoutingAfterQiskitLayout:
    """
        Pre-processing step to route a circuit with TKET after a Qiskit Layout pass has been applied.
        The reason why we can apply the trivial layout here is that the circuit already got assigned a layout by qiskit.
        Implicitly, Qiskit is reordering its qubits in a sequential manner, i.e., the qubit with the lowest *physical* qubit
        first.

        Assuming, the layouted circuit is given by

                       ┌───┐           ░       ┌─┐
              q_2 -> 0 ┤ H ├──■────────░───────┤M├
                       └───┘┌─┴─┐      ░    ┌─┐└╥┘
              q_1 -> 1 ─────┤ X ├──■───░────┤M├─╫─
                            └───┘┌─┴─┐ ░ ┌─┐└╥┘ ║
              q_0 -> 2 ──────────┤ X ├─░─┤M├─╫──╫─
                                 └───┘ ░ └╥┘ ║  ║
        ancilla_0 -> 3 ───────────────────╫──╫──╫─
                                          ║  ║  ║
        ancilla_1 -> 4 ───────────────────╫──╫──╫─
                                          ║  ║  ║
               meas: 3/═══════════════════╩══╩══╩═
                                          0  1  2

        Applying the trivial layout, we get the same qubit order as in the original circuit and can be respectively
        routed. This results int:
                ┌───┐           ░       ┌─┐
           q_0: ┤ H ├──■────────░───────┤M├
                └───┘┌─┴─┐      ░    ┌─┐└╥┘
           q_1: ─────┤ X ├──■───░────┤M├─╫─
                     └───┘┌─┴─┐ ░ ┌─┐└╥┘ ║
           q_2: ──────────┤ X ├─░─┤M├─╫──╫─
                          └───┘ ░ └╥┘ ║  ║
           q_3: ───────────────────╫──╫──╫─
                                   ║  ║  ║
           q_4: ───────────────────╫──╫──╫─
                                   ║  ║  ║
        meas: 3/═══════════════════╩══╩══╩═
                                   0  1  2


        If we would not apply the trivial layout, no layout would be considered resulting, e.g., in the followiong circuit:
                 ┌───┐         ░    ┌─┐
       q_0: ─────┤ X ├─────■───░────┤M├───
            ┌───┐└─┬─┘   ┌─┴─┐ ░ ┌─┐└╥┘
       q_1: ┤ H ├──■───X─┤ X ├─░─┤M├─╫────
            └───┘      │ └───┘ ░ └╥┘ ║ ┌─┐
       q_2: ───────────X───────░──╫──╫─┤M├
                               ░  ║  ║ └╥┘
       q_3: ──────────────────────╫──╫──╫─
                                  ║  ║  ║
       q_4: ──────────────────────╫──╫──╫─
                                  ║  ║  ║
    meas: 3/══════════════════════╩══╩══╩═
                                  0  1  2

    """

    def apply(self, circuit: Circuit) -> None:
        """Applies the pre-processing step to route a circuit with tket after a Qiskit Layout pass has been applied."""
        mapping = {Qubit(i): Node(i) for i in range(circuit.n_qubits)}
        place_with_map(circuit=circuit, qmap=mapping)


def get_bqskit_native_gates(device: Device) -> list[gates.Gate] | None:
    """Returns the native gates of the given device.

    Args:
        device: The device for which the native gates are returned.

    Returns:
        list[gates.Gate]: The native gates of the given provider.
    """
    provider = device.name.split("_")[0]

    native_gatesets = {
        "ibm": [gates.RZGate(), gates.SXGate(), gates.XGate(), gates.CNOTGate()],
        "rigetti": [gates.RXGate(), gates.RZGate(), gates.CZGate()],
        "ionq": [gates.RXXGate(), gates.RZGate(), gates.RYGate(), gates.RXGate()],
        "quantinuum": [gates.RZZGate(), gates.RZGate(), gates.RYGate(), gates.RXGate()],
    }

    if provider not in native_gatesets:
        logger.warning("No native gateset for provider " + provider + " found. No native gateset is used.")
        return None

    return native_gatesets[provider]


def final_layout_pytket_to_qiskit(pytket_circuit: Circuit, qiskit_ciruit: QuantumCircuit) -> Layout:
    pytket_layout = pytket_circuit.qubit_readout
    size_circuit = pytket_circuit.n_qubits
    qiskit_layout = {}
    qiskit_qreg = qiskit_ciruit.qregs[0]

    pytket_layout = dict(sorted(pytket_layout.items(), key=operator.itemgetter(1)))

    for node, qubit_index in pytket_layout.items():
        qiskit_layout[node.index[0]] = qiskit_qreg[qubit_index]

    for i in range(size_circuit):
        if i not in set(pytket_layout.values()):
            qiskit_layout[i] = qiskit_qreg[i]

    return Layout(input_dict=qiskit_layout)


def final_layout_bqskit_to_qiskit(
    bqskit_initial_layout: list[int],
    bqskit_final_layout: list[int],
    compiled_qc: QuantumCircuit,
    initial_qc: QuantumCircuit,
) -> TranspileLayout:
    # BQSKit provides an initial layout as a list[int] where each virtual qubit is mapped to a physical qubit
    # similarly, it provides a final layout as a list[int] representing where each virtual qubit is mapped to at the end
    # of the circuit

    ancilla = QuantumRegister(compiled_qc.num_qubits - initial_qc.num_qubits, "ancilla")
    qiskit_initial_layout = {}
    for i in range(compiled_qc.num_qubits):
        if i in bqskit_initial_layout:
            qiskit_initial_layout[i] = initial_qc.qubits[bqskit_initial_layout.index(i)]
        else:
            qiskit_initial_layout[i] = ancilla[i - initial_qc.num_qubits]

    initial_qubit_mapping = {bit: index for index, bit in enumerate(compiled_qc.qubits)}

    if bqskit_initial_layout == bqskit_final_layout:
        qiskit_final_layout = None
    else:
        qiskit_final_layout = {}
        for i in range(compiled_qc.num_qubits):
            if i in bqskit_final_layout:
                qiskit_final_layout[i] = compiled_qc.qubits[bqskit_initial_layout[bqskit_final_layout.index(i)]]
            else:
                qiskit_final_layout[i] = compiled_qc.qubits[i]

    return TranspileLayout(
        initial_layout=Layout(input_dict=qiskit_initial_layout),
        input_qubit_mapping=initial_qubit_mapping,
        final_layout=Layout(input_dict=qiskit_final_layout) if qiskit_final_layout else None,
        _output_qubit_list=compiled_qc.qubits,
        _input_qubit_count=initial_qc.num_qubits,
    )


def get_ibm_backend_properties_by_device_name(device_name: str) -> BackendProperties | None:
    """Returns the IBM backend name for the given device name.

    Args:
        device_name (str): The name of the device for which the IBM backend name is returned.

    Returns:
        str: The IBM backend name for the given device name.
    """
    if "ibm" not in device_name:
        return None
    if device_name == "ibm_washington":
        return FakeWashington().properties()
    if device_name == "ibm_montreal":
        return FakeMontreal().properties()
    if device_name == "ibm_guadalupe":
        return FakeGuadalupe().properties()
    if device_name == "ibm_quito":
        return FakeQuito().properties()
    return None


def postprocess_vf2postlayout(
    qc: QuantumCircuit, post_layout: Layout, layout_before: TranspileLayout
) -> tuple[QuantumCircuit, PassManager]:
    """Postprocesses the given quantum circuit with the post_layout and returns the altered quantum circuit and the respective PassManager."""
    apply_layout = ApplyLayout()
    assert layout_before is not None
    apply_layout.property_set["layout"] = layout_before.initial_layout
    apply_layout.property_set["original_qubit_indices"] = layout_before.input_qubit_mapping
    apply_layout.property_set["final_layout"] = layout_before.final_layout
    apply_layout.property_set["post_layout"] = post_layout

    altered_qc = apply_layout(qc)
    return altered_qc, apply_layout
