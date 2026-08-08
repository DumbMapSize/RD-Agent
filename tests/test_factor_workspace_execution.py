import os
import sys
import time
from pathlib import Path

import pytest

from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
from rdagent.components.coder.factor_coder.factor import FactorFBWorkspace, FactorTask
from rdagent.core.conf import RD_AGENT_SETTINGS


@pytest.fixture
def factor_workspace(monkeypatch, tmp_path: Path) -> FactorFBWorkspace:
    data_path = tmp_path / "factor_data"
    data_path.mkdir()
    monkeypatch.setattr(RD_AGENT_SETTINGS, "workspace_path", tmp_path / "workspaces")
    monkeypatch.setattr(RD_AGENT_SETTINGS, "cache_with_pickle", False)
    monkeypatch.setattr(FACTOR_COSTEER_SETTINGS, "data_folder_debug", str(data_path))
    monkeypatch.setattr(FACTOR_COSTEER_SETTINGS, "python_bin", sys.executable)
    task = FactorTask("test_factor", "test factor", "test formulation")
    return FactorFBWorkspace(target_task=task)


def test_execute_does_not_reuse_previous_revision_output(factor_workspace: FactorFBWorkspace) -> None:
    factor_workspace.inject_files(
        **{
            "factor.py": """\
import pandas as pd

pd.DataFrame({"factor": [7.0]}).to_hdf("result.h5", key="factor")
"""
        }
    )
    first_feedback, first_result = factor_workspace.execute()

    assert "Expected output file found" in first_feedback
    assert first_result["factor"].tolist() == [7.0]

    factor_workspace.inject_files(**{"factor.py": "pass\n"})
    second_feedback, second_result = factor_workspace.execute()

    assert second_result is None
    assert "Expected output file not found" in second_feedback
    assert not (factor_workspace.workspace_path / "result.h5").exists()


@pytest.mark.skipif(os.name != "posix", reason="Process-group cleanup is POSIX-specific")
def test_timeout_terminates_descendants_and_prevents_late_output(
    factor_workspace: FactorFBWorkspace,
    monkeypatch,
) -> None:
    monkeypatch.setattr(FACTOR_COSTEER_SETTINGS, "file_based_execution_timeout", 0.75)
    factor_workspace.inject_files(
        **{
            "factor.py": """\
import subprocess
import sys
import time
from pathlib import Path

child = subprocess.Popen([
    sys.executable,
    "-c",
    "import signal, time; from pathlib import Path; "
    "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
    "time.sleep(2); Path('late-output').write_text('late')",
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
Path("child.pid").write_text(str(child.pid))
time.sleep(60)
"""
        }
    )

    feedback, result = factor_workspace.execute()

    assert result is None
    assert "Execution timeout error" in feedback
    child_pid = int((factor_workspace.workspace_path / "child.pid").read_text())
    time.sleep(2.25)
    assert not (factor_workspace.workspace_path / "late-output").exists()
    assert not Path(f"/proc/{child_pid}").exists()
