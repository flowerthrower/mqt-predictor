"""Tests for the machine learning device selection predictor module."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Literal

import pytest
from qiskit.qasm2 import dump

from mqt.bench import benchmark_generator
from mqt.predictor import ml, reward, rl


def test_train_and_predictor_random_forest_classifier() -> None:
    """Test the training of a random forest classifier.

    This test must be executed prior to any prediction to make sure the model is trained using the latest scikit-learn version.
    """
    predictor = ml.Predictor()
    assert predictor.clf is None
    predictor.train_random_forest_classifier(visualize_results=False, save_classifier=False)

    assert predictor.clf is not None

    qc = benchmark_generator.get_benchmark("ghz", 1, 3)
    prediction = predictor.predict_probs(qc, "expected_fidelity")
    for elem in prediction:
        assert 0 <= elem <= 1

    file = Path("test_qasm.qasm")
    qc = benchmark_generator.get_benchmark("dj", 1, 3)
    with file.open("w", encoding="utf-8") as f:
        dump(qc, f)

    prediction = predictor.predict_probs(file, "expected_fidelity")
    for elem in prediction:
        assert 0 <= elem <= 1

    with pytest.raises(
        FileNotFoundError, match=re.escape("The ML model is not trained yet. Please train the model before using it.")
    ):
        ml.helper.predict_device_for_figure_of_merit(qc, "false_input")  # type: ignore[arg-type]

    (ml.helper.get_path_trained_model("expected_fidelity").parent / "non_zero_indices_expected_fidelity.npy").unlink()


def test_performance_measures() -> None:
    """Test the calculation of the performance measures for a given set of scores and labels."""
    predictor = ml.Predictor()
    figure_of_merit: Literal["expected_fidelity"] = "expected_fidelity"

    training_data = predictor.get_prepared_training_data(figure_of_merit=figure_of_merit, save_non_zero_indices=False)

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
    """Test the compilation of all circuits for a given device and figure of merit."""
    predictor = ml.Predictor()
    source_path = Path()
    target_path = Path("test_compiled_circuits")
    if not target_path.exists():
        target_path.mkdir()
    figure_of_merit: reward.figure_of_merit = "expected_fidelity"

    qc = benchmark_generator.get_benchmark("dj", 1, 3)
    qasm_path = Path("test.qasm")
    with Path(qasm_path).open("w", encoding="utf-8") as f:
        dump(qc, f)

    if sys.platform == "win32":
        with pytest.warns(RuntimeWarning, match=re.escape("Timeout is not supported on Windows.")):
            predictor.compile_all_circuits_devicewise(
                device_name="ionq_harmony",
                timeout=100,
                figure_of_merit=figure_of_merit,
                source_path=source_path,
                target_path=target_path,
            )
    else:
        predictor.compile_all_circuits_devicewise(
            device_name="ionq_harmony",
            timeout=100,
            figure_of_merit=figure_of_merit,
            source_path=source_path,
            target_path=target_path,
        )

    model_path = Path(rl.helper.get_path_trained_model() / ("model_" + figure_of_merit + "_ionq_harmony.zip"))
    model_path.unlink()
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

    delete_path(target_path)

    if qasm_path.exists():
        qasm_path.unlink()


def delete_path(path: Path) -> None:
    """Delete a path recursively."""
    if path.exists():
        for item in path.iterdir():
            if item.is_dir():
                delete_path(item)  # Recursive deletion
            else:
                item.unlink()  # Delete file
        path.rmdir()  # Remove empty directory
