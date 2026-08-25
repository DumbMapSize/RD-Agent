from rdagent.scenarios.qlib.experiment.workspace import _extract_training_log_summary


def test_extract_training_log_summary_supports_positive_and_negative_qlib_general_ptnn_logs() -> None:
    qlib_log = "\n".join(
        [
            "[5771:MainThread](2026-06-24 22:23:14,404) INFO - qlib.GeneralPTNN - "
            "[pytorch_general_nn.py:299] - Epoch0:",
            "[5771:MainThread](2026-06-24 22:23:18,565) INFO - qlib.GeneralPTNN - "
            "[pytorch_general_nn.py:305] - Epoch0: train 0.995449, valid 0.996051",
            "[5771:MainThread](2026-06-24 22:23:19,565) INFO - qlib.GeneralPTNN - "
            "[rdagent_general_ptnn.py:612] - Epoch1: train -0.123000, valid -0.456000",
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


def test_extract_training_log_summary_preserves_structured_checkpoint_evidence() -> None:
    expected = [
        "RD-Agent training context (authoritative): optimizer=adamw; loss=mse; batch_mode=date; "
        "checkpoint=topk_precision@20; direction=maximize; scheduler=plateau; "
        "scheduler_monitor=valid_loss; epochs=6; early_stop_patience=2",
        "Epoch0: train_loss=0.001200; valid_loss=0.001300; "
        "train_topk_precision@20=0.095489; valid_topk_precision@20=0.085744; lr=0.00015",
        "Epoch1: train_loss=0.001100; valid_loss=0.001400; "
        "train_topk_precision@20=unavailable; valid_topk_precision@20=0.088223; lr=7.5e-05",
        "early stop: checkpoint=topk_precision@20; patience=2; epoch=5",
        "best checkpoint: metric=topk_precision@20; direction=maximize; value=0.088223; epoch=3",
    ]
    qlib_log = "\n".join(
        f"[16902:MainThread](2026-08-22 17:29:{20 + index:02d},495) INFO - qlib.GeneralPTNN - "
        f"[rdagent_general_ptnn.py:{610 + index}] - {line}"
        for index, line in enumerate(expected)
    )

    assert _extract_training_log_summary(qlib_log) == "\n".join(expected)


def test_extract_training_log_summary_rejects_mismatched_structured_checkpoint_metrics() -> None:
    qlib_log = (
        "Epoch0: train_loss=0.001200; valid_loss=0.001300; "
        "train_rank_ic=0.095489; valid_ic=0.085744; lr=0.00015"
    )

    assert _extract_training_log_summary(qlib_log) == ""
