from pathlib import Path

from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
from rdagent.scenarios.qlib.experiment import utils


def test_generate_data_folder_from_qlib_uses_conda_env(monkeypatch, tmp_path):
    template_root = tmp_path / "experiment"
    template_path = template_root / "factor_data_template"
    template_path.mkdir(parents=True)
    (template_root / "utils.py").write_text("# test module placeholder\n")
    (template_path / "daily_pv_all.h5").write_bytes(b"full")
    (template_path / "daily_pv_debug.h5").write_bytes(b"debug")
    (template_path / "README.md").write_text("readme\n")

    full_output = tmp_path / "factor_implementation_source_data"
    debug_output = tmp_path / "factor_implementation_source_data_debug"
    monkeypatch.setattr(utils, "__file__", str(template_root / "utils.py"))
    monkeypatch.setattr(FACTOR_COSTEER_SETTINGS, "data_folder", str(full_output))
    monkeypatch.setattr(FACTOR_COSTEER_SETTINGS, "data_folder_debug", str(debug_output))

    calls = []

    class FakeCondaConf:
        pass

    class FakeCondaEnv:
        def __init__(self, conf):
            assert isinstance(conf, FakeCondaConf)
            calls.append(("conda_init",))

        def prepare(self):
            calls.append(("conda_prepare",))

        def check_output(self, local_path, entry):
            calls.append(("conda_check_output", Path(local_path).name, entry))
            return "conda log"

    class FakeDockerEnv:
        def __init__(self, *args, **kwargs):
            raise AssertionError("generate_data_folder_from_qlib should not instantiate QTDockerEnv")

    monkeypatch.setattr(utils, "QlibCondaConf", FakeCondaConf)
    monkeypatch.setattr(utils, "QlibCondaEnv", FakeCondaEnv)
    monkeypatch.setattr(utils, "QTDockerEnv", FakeDockerEnv, raising=False)

    utils.generate_data_folder_from_qlib()

    assert calls == [
        ("conda_init",),
        ("conda_prepare",),
        ("conda_check_output", "factor_data_template", "python generate.py"),
    ]
    assert (full_output / "daily_pv.h5").read_bytes() == b"full"
    assert (debug_output / "daily_pv.h5").read_bytes() == b"debug"
    assert (full_output / "README.md").read_text() == "readme\n"
    assert (debug_output / "README.md").read_text() == "readme\n"
