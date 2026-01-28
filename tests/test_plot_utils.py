import matplotlib.pyplot as plt
import pytest

from post_processing.utils.plot_utils import (
    _wrap_xtick_labels,
    get_legend,
    overview,
    set_y_axis_to_percentage,
)


def test_overview_runs_without_error(sample_df) -> None:
    try:
        overview(sample_df)
    except ValueError:
        pytest.fail("test_detection_perf raised ValueError unexpectedly.")


def test_wrap_xtick_labels_short_labels():
    fig, ax = plt.subplots()
    labels = ["A", "B", "C"]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)

    _wrap_xtick_labels(ax, max_chars=2)

    wrapped_labels = [label.get_text() for label in ax.get_xticklabels()]
    assert wrapped_labels == labels


def test_wrap_xtick_labels_long_label():
    fig, ax = plt.subplots()
    labels = ["This is a long label"]
    ax.set_xticks([0])
    ax.set_xticklabels(labels)

    _wrap_xtick_labels(ax, max_chars=5)

    wrapped_labels = [label.get_text() for label in ax.get_xticklabels()]
    expected = "This\nis a\nlong\nlabel"
    assert wrapped_labels[0] == expected


def test_wrap_xtick_labels_no_spaces():
    fig, ax = plt.subplots()
    labels = ["abcdefghijk"]
    ax.set_xticks([0])
    ax.set_xticklabels(labels)

    _wrap_xtick_labels(ax, max_chars=4)

    wrapped_labels = [label.get_text() for label in ax.get_xticklabels()]

    expected = "abcd\nefgh\nijk"
    assert wrapped_labels[0] == expected


def test_set_y_axis_to_percentage():
    fig, ax = plt.subplots()
    ax.set_ylabel("Accuracy")
    set_y_axis_to_percentage(ax, max_val=200)
    formatter = ax.yaxis.get_major_formatter()
    assert formatter(100, None) == "50%"
    assert formatter(200, None) == "100%"
    assert ax.get_ylabel() == "Accuracy (%)"
    plt.close(fig)


def test_single_annotator_multiple_labels():
    annotators = ["Alice"]
    labels = ["Label1", "Label2", "Label3"]
    result = get_legend(annotators, labels)
    assert result == labels


def test_multiple_annotators_single_label():
    annotators = ["Alice", "Bob", "Charlie"]
    labels = ["CommonLabel"]
    result = get_legend(annotators, labels)
    assert result == annotators


def test_multiple_annotators_multiple_labels():
    annotators = ["Alice", "Bob", "Charlie"]
    labels = ["Label1", "Label2", "Label3"]
    result = get_legend(annotators, labels)
    expected = ["Alice\nLabel1", "Bob\nLabel2", "Charlie\nLabel3"]
    assert result == expected


def test_single_annotator_single_label():
    annotators = ["Alice"]
    labels = ["Label1"]
    result = get_legend(annotators, labels)
    assert result == ["Alice\nLabel1"]


def test_lists_and_strings_combined():
    annotators = ["Alice", "Bob"]
    labels = ["Label1", "Label2"]
    result = get_legend(annotators, labels)
    expected = ["Alice\nLabel1", "Bob\nLabel2"]
    assert result == expected
