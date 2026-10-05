"""Fine-tuning must continue from the network as it stands.

The flow is carried over between retrainings, reinitialised only when the run is
configured to do so. A fine-tune that reloaded an older checkpoint on the way in
would silently undo the intervening stages.
"""

import torch

from nsflows.samplers.DL_samplers import nflows_propagator


class _Trainer:
    """Stands in for the real trainer: records the calls and nudges a parameter so
    a test can tell whether training was carried over or thrown away."""

    def __init__(self, flow):
        self.flow = flow
        self.stages = []

    def training_routine(self, train_dataset, stage=None, counter=None, **kwargs):
        self.stages.append((counter, stage))
        with torch.no_grad():
            self.flow.weight += 1.0
        return torch.zeros(1, 1)


class _Flow(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))


def _propagator(tmp_path):
    p = object.__new__(nflows_propagator)
    p.flow = _Flow()
    p.flow_trainer = _Trainer(p.flow)
    p.training_counter = 0
    p.loaded = []
    p.load_parameters = lambda params_path=None, **kw: p.loaded.append(params_path)
    return p


STAGE = {"total_steps": 4, "batch_size": 2, "optimizer": None, "scheduler": None,
         "w_xz": 1, "w_zx": 0, "conds_per_batch": -1, "save_best": False,
         "start_lr": 1e-3, "end_lr": 1e-3, "max_lr": 1e-3}
PROTOCOL = [dict(STAGE), dict(STAGE)]          # [OneCycle-like, anneal-like]


class _Dataset:
    def __len__(self):
        return 4


def test_fine_tune_runs_only_the_last_stage(tmp_path):
    p = _propagator(tmp_path)
    p.train(_Dataset(), PROTOCOL, fine_tune=False, outputdir=str(tmp_path))
    p.train(_Dataset(), PROTOCOL, fine_tune=True, outputdir=str(tmp_path))

    full = [s for c, s in p.flow_trainer.stages[:2]]
    tuned = [s for c, s in p.flow_trainer.stages[2:]]
    assert full == [0, 1], full
    assert tuned == [1], tuned


def test_fine_tune_does_not_reload_an_older_checkpoint(tmp_path):
    """Three fine-tunes after one full training must accumulate, not reset.

    The defect reloaded the same file at the start of every fine-tune -- the
    second-to-last stage of the last full training -- so the three fine-tunes
    each started from the same state instead of continuing from one another.
    """
    p = _propagator(tmp_path)
    p.train(_Dataset(), PROTOCOL, fine_tune=False, outputdir=str(tmp_path))
    for _ in range(3):
        p.train(_Dataset(), PROTOCOL, fine_tune=True, outputdir=str(tmp_path))

    assert p.loaded == [], f"fine-tuning reloaded {p.loaded}"
    # two stages in the full training, then one per fine-tune
    assert float(p.flow.weight) == 5.0, float(p.flow.weight)


def test_training_counter_advances_once_per_call(tmp_path):
    p = _propagator(tmp_path)
    for _ in range(4):
        p.train(_Dataset(), PROTOCOL, fine_tune=False, outputdir=str(tmp_path))
    assert p.training_counter == 4
    assert [c for c, _ in p.flow_trainer.stages] == [0, 0, 1, 1, 2, 2, 3, 3]
