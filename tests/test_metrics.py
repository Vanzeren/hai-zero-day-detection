import numpy as np
import pytest

from hai_zd.data import Stream
from hai_zd.metrics import ScoreRecord, binary_events, evaluate_records


def test_binary_events_are_inclusive_and_keep_tail():
    assert binary_events(np.array([0, 1, 1, 0, 1])) == [(1, 2), (4, 4)]
    assert binary_events(np.ones(3, dtype=np.uint8)) == [(0, 2)]
    assert binary_events(np.zeros(3, dtype=np.uint8)) == []


def test_official_etapr_stream_example_without_point_adjustment():
    labels = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1], dtype=np.uint8)
    predictions = np.array([0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0], dtype=np.uint8)
    stream = Stream("21.03", "test1", np.zeros((17, 1), dtype=np.float32), labels, np.arange(17))
    record = ScoreRecord(stream, np.arange(17), predictions.astype(float), np.zeros((17, 1)), predictions)
    attack_rows = [{
        "delay_seconds": 0.0,
        "detected": True,
        "target_retained": False,
        "attribution_hit": np.nan,
    }]
    result = evaluate_records([record], eval_stride=1, attack_rows=attack_rows)
    assert result["eta_recall"] == pytest.approx(0.3875)
    assert result["eta_precision"] == pytest.approx(0.46476766302377037)
    assert result["eta_f1"] == pytest.approx(0.4226312395393011)
    assert result["point_f1"] == pytest.approx(0.5)
