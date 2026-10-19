import numpy as np

from hai_zd.attacks import ATTACK_TARGETS, FILE_ATTACKS
from hai_zd.data import Stream, evaluation_ends, extract_windows, training_ends


def test_causal_windows_and_partition_boundaries():
    stream = Stream(
        "21.03",
        "train1",
        x=np.arange(200, dtype=np.float32).reshape(100, 2),
        y=np.zeros(100, dtype=np.uint8),
        timestamps=np.arange(100, dtype=np.int64),
        train_end=70,
    )
    train = training_ends(stream, window=5, stride=10)
    calibration = evaluation_ends(stream, window=5, stride=10, calibration=True)
    assert train.tolist() == [4, 14, 24, 34, 44, 54, 64]
    assert calibration.tolist() == [70, 80, 90]
    windows = extract_windows(stream.x, np.array([14]), window=5)
    np.testing.assert_array_equal(windows[0], stream.x[10:15].T)


def test_attack_metadata_covers_every_observed_event():
    assert set(ATTACK_TARGETS) == {"21.03", "23.05"}
    assert sum(map(len, FILE_ATTACKS["21.03"].values())) == 50
    assert sum(map(len, FILE_ATTACKS["23.05"].values())) == 52
    for version, files in FILE_ATTACKS.items():
        for attack_ids in files.values():
            assert len(attack_ids) == len(set(attack_ids))
            assert all(attack_id in ATTACK_TARGETS[version] for attack_id in attack_ids)
