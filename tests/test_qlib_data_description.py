import pickle
from types import SimpleNamespace

import pandas as pd

from rdagent.app.qlib_rd_loop import quant as quant_module
from rdagent.scenarios.qlib.experiment import quant_experiment, utils
from rdagent.scenarios.qlib.experiment.quant_experiment import QlibQuantScenario


def test_refresh_source_data_reads_current_docs_without_rebuilding_data(tmp_path, monkeypatch):
    full = tmp_path / "full"
    debug = tmp_path / "debug"
    full.mkdir()
    debug.mkdir()
    monkeypatch.setattr(utils.FACTOR_COSTEER_SETTINGS, "data_folder", str(full))
    monkeypatch.setattr(utils.FACTOR_COSTEER_SETTINGS, "data_folder_debug", str(debug))

    def unexpected_generation():
        raise AssertionError("Refreshing documentation must not regenerate existing data")

    monkeypatch.setattr(utils, "generate_data_folder_from_qlib", unexpected_generation)
    path = debug / "daily_pv.h5"
    frame = pd.DataFrame(
        {"$close": [20.0], "$factor": [2.0]},
        index=pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2020-01-02"), "SH600000")], names=["datetime", "instrument"]
        ),
    )
    frame.to_hdf(path, key="data")
    before = path.read_bytes()
    readme = debug / "README.md"
    readme.write_text("Old field description.")
    scenario = QlibQuantScenario()
    settings = scenario.experiment_setting
    scenario = pickle.loads(pickle.dumps(scenario))
    readme.write_text("Updated field description: raw_close = $close / $factor.")

    scenario.refresh_source_data()

    description = scenario.get_source_data_desc()
    assert "Updated field description" in description
    assert "Old field description" not in description
    assert "$close: float64" in description
    assert scenario.experiment_setting == settings
    assert path.read_bytes() == before


def test_quant_resume_refreshes_shared_scenario_before_running(monkeypatch):
    scenario = object.__new__(QlibQuantScenario)
    scenario._source_data = "checkpoint data description"
    saved = SimpleNamespace(
        trace=SimpleNamespace(scen=scenario, hist=["accepted experiment"]),
        factor_coder=SimpleNamespace(scen=scenario),
        hypothesis_gen=SimpleNamespace(scen=scenario),
        plan={},
    )
    checkpoint = pickle.dumps(saved)
    loop = pickle.loads(checkpoint)
    reads = []

    def current_description():
        reads.append(True)
        return "current data description"

    monkeypatch.setattr(quant_experiment, "get_data_folder_intro", current_description)
    monkeypatch.setattr(quant_module.QuantRDLoop, "load", lambda *args, **kwargs: loop)
    observed = []

    async def run(**kwargs):
        observed.append(loop.factor_coder.scen.get_source_data_desc())
        assert loop.trace.scen is loop.hypothesis_gen.scen is loop.factor_coder.scen

    loop.run = run
    quant_module.main(path="checkpoint", step_n=0)

    assert observed == ["current data description"]
    assert reads == [True]
    assert loop.trace.hist == ["accepted experiment"]
    assert pickle.loads(checkpoint).trace.scen.get_source_data_desc() == "checkpoint data description"
