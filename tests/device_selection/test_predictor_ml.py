from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from qiskit.qasm2 import dump, dumps

from mqt.bench import benchmark_generator
from mqt.bench.devices import get_available_device_names, get_available_devices
from mqt.predictor import ml, reward


def test_train_random_forest_classifier() -> None:
    """Test the training of a random forest classifier. This test must be executed prior to any prediction to make sure
    the model is trained using the latest scikit-learn version."""
    predictor = ml.Predictor()
    assert predictor.clf is None
    predictor.train_random_forest_classifier(visualize_results=False)

    assert predictor.clf is not None


def test_predict_device_for_figure_of_merit() -> None:
    qc = benchmark_generator.get_benchmark("ghz", 1, 5)
    assert ml.helper.predict_device_for_figure_of_merit(qc, "expected_fidelity").name in get_available_device_names()

    with pytest.raises(FileNotFoundError, match="Classifier is neither trained nor saved."):
        ml.helper.predict_device_for_figure_of_merit(qc, "false_input")  # type: ignore[arg-type]


def test_predict() -> None:
    path = ml.helper.get_path_trained_model(figure_of_merit="expected_fidelity")
    assert path.is_file()
    filename = "test_qasm.qasm"
    figure_of_merit: reward.figure_of_merit = "expected_fidelity"
    qc = benchmark_generator.get_benchmark("dj", 1, 8)
    with Path(filename).open("w") as f:
        dump(qc, f)
    predictor = ml.Predictor()
    predictions = predictor.predict_probs(filename, figure_of_merit=figure_of_merit)
    assert predictor.clf is not None
    classes = predictor.clf.classes_  # type: ignore[unreachable]
    predicted_device_indices = classes[np.argsort(predictions)[::-1]]
    devices = get_available_devices()
    assert all(0 <= i < len(devices) for i in predicted_device_indices)
    predictions = predictor.predict_probs(dumps(qc), figure_of_merit=figure_of_merit)
    predicted_device_indices = classes[np.argsort(predictions)[::-1]]
    assert all(0 <= i < len(devices) for i in predicted_device_indices)
    Path(filename).unlink()


def test_performance_measures() -> None:
    predictor = ml.Predictor()
    figure_of_merit: Literal["expected_fidelity"] = "expected_fidelity"

    training_data = predictor.get_prepared_training_data(figure_of_merit=figure_of_merit, save_non_zero_indices=True)

    y_test = training_data.y_test
    indices_test = training_data.indices_test
    names_list = training_data.names_list
    scores_list = training_data.scores_list

    assert len(y_test) > 0
    assert len(indices_test) > 0
    assert len(names_list) > 0
    assert len(scores_list) > 0

    scores_filtered = [scores_list[i] for i in indices_test]
    names_filtered = [names_list[i] for i in indices_test]

    # Test calc_performance_measures
    res, relative_scores = predictor.calc_performance_measures(scores_filtered, y_test, y_test)
    assert all(res)
    assert not any(relative_scores)

    # Test generate_eval_histogram
    predictor.generate_eval_histogram(res, show_plot=False)
    histogram_path = Path("results/histogram.pdf")
    assert histogram_path.is_file(), "File does not exist"
    histogram_path.unlink()

    # Test generate_eval_all_datapoints
    predictor.generate_eval_all_datapoints(names_filtered, scores_filtered, y_test, y_test)
    result_path = Path("results/y_pred_eval_normed.pdf")
    assert result_path.is_file(), "File does not exist"
    result_path.unlink()


def test_compile_all_circuits_for_dev_and_fom() -> None:
    predictor = ml.Predictor()
    source_path = Path()
    target_path = Path("test_compiled_circuits")
    if not target_path.exists():
        target_path.mkdir()
    figure_of_merit: reward.figure_of_merit = "expected_fidelity"

    qc = benchmark_generator.get_benchmark("dj", 1, 3)
    qasm_path = Path("test.qasm")
    with Path(qasm_path).open("w") as f:
        dump(qc, f)
    predictor.compile_all_circuits_devicewise(
        device_name="ibm_montreal",
        timeout=100,
        figure_of_merit=figure_of_merit,
        source_path=source_path,
        target_path=target_path,
    )
    assert any(file.suffix == ".qasm" for file in target_path.iterdir())

    training_sample, circuit_name, scores = predictor.generate_training_sample(
        file=qasm_path,
        figure_of_merit=figure_of_merit,
        path_uncompiled_circuit=source_path,
        path_compiled_circuits=target_path,
    )
    assert training_sample
    assert circuit_name is not None
    assert any(score != -1 for score in scores)

    (
        training_data,
        name_list,
        scores_list,
    ) = predictor.generate_trainingdata_from_qasm_files(figure_of_merit, source_path, target_path)
    assert len(training_data) > 0
    assert len(name_list) > 0
    assert len(scores_list) > 0

    if target_path.exists():
        for file in target_path.iterdir():
            file.unlink()
        target_path.rmdir()

    if qasm_path.exists():
        qasm_path.unlink()
