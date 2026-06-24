from rdagent.scenarios.qlib.experiment.workspace import _extract_training_log_summary


def test_extract_training_log_summary_supports_positive_and_negative_qlib_general_ptnn_logs() -> None:
    qlib_log = "\n".join(
        [
            "[5771:MainThread](2026-06-24 22:23:14,404) INFO - qlib.GeneralPTNN - "
            "[pytorch_general_nn.py:299] - Epoch0:",
            "[5771:MainThread](2026-06-24 22:23:18,565) INFO - qlib.GeneralPTNN - "
            "[pytorch_general_nn.py:305] - Epoch0: train 0.995449, valid 0.996051",
            "[5771:MainThread](2026-06-24 22:23:19,565) INFO - qlib.GeneralPTNN - "
            "[pytorch_general_nn.py:305] - Epoch1: train -0.123000, valid -0.456000",
            "[5771:MainThread](2026-06-24 22:23:39,509) INFO - qlib.GeneralPTNN - "
            "[pytorch_general_nn.py:327] - best score: 0.995410 @ 2 epoch",
        ]
    )

    assert _extract_training_log_summary(qlib_log) == "\n".join(
        [
            "Epoch0: train 0.995449, valid 0.996051",
            "Epoch1: train -0.123000, valid -0.456000",
            "best score: 0.995410 @ 2 epoch",
        ]
    )


def test_extract_training_log_summary_preserves_original_valid_negative_output() -> None:
    qlib_log = "\n".join(
        [
            "Epoch0: train -0.123000, valid -0.456000",
            "Epoch1: train -0.222000, valid -0.333000",
            "best score: -0.456000 @ 2 epoch",
        ]
    )

    assert _extract_training_log_summary(qlib_log) == qlib_log


def test_extract_training_log_summary_rejects_substring_and_malformed_matches() -> None:
    qlib_log = "\n".join(
        [
            "The phrase Epoch0: train 1.234000, valid 0.995410 appears in a note.",
            "[5771:MainThread](2026-06-24 22:23:18,565) INFO - other.Logger - "
            "[pytorch_general_nn.py:305] - Epoch0: train 0.995449, valid 0.996051",
            "[5771:MainThread](2026-06-24 22:23:18,565) INFO - qlib.GeneralPTNN - "
            "[other.py:305] - Epoch0: train 0.995449, valid 0.996051",
            "Epoch0: train 1.2.3, valid 0.995410",
            "Epoch0: train 1.234000, valid 0.995410 extra",
            "Epoch0: train nan, valid 0.995410",
        ]
    )

    assert _extract_training_log_summary(qlib_log) == ""


def test_extract_training_log_summary_accepts_scientific_notation() -> None:
    qlib_log = "Epoch12: train 1.2e-03, valid +4.5E-02"

    assert _extract_training_log_summary(qlib_log) == qlib_log
