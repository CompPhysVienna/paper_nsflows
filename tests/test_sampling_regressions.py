"""Regressions for defects that were silent: they produced plausible runs.

Each test below pins one of them. They use stand-ins for the flow rather than a
trained network, because every defect lives in the bookkeeping around the flow
and none of them depends on what the flow actually generates.

Run directly (``python tests/test_sampling_regressions.py``) or under pytest.
"""

import os
import sys
import tempfile
from collections import deque

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import nsflows.nested_sampling as ns_module
from nsflows.nested_sampling import nested_sampling
from nsflows.samplers.DL_samplers import nflows_propagator
from nsflows.samplers.monte_carlo import rejection_monte_carlo
from nsflows.systems.lennard_jones import lennard_jones


DOFS = 4
DEVICE = torch.device("cpu")


# --------------------------------------------------------------------------
# stand-ins
# --------------------------------------------------------------------------

class _Posterior:
    """Energy is the first coordinate, so a test can place a configuration
    above or below any bound it likes."""

    PBC = False
    n_particles, dimensions, dofs = DOFS // 2, 2, DOFS

    def energy(self, x):
        return x[:, :1].clone()


class _Prior:
    def sample(self, n_samples, transform=True):
        return torch.zeros(n_samples, DOFS)

    def energy(self, z):
        return torch.zeros(z.shape[0], 1)


class _Flow:
    """Maps the base samples onto configurations whose energies are dictated by
    ``energies``, cycling through it one batch at a time."""

    def __init__(self, energies):
        self.device = DEVICE
        self.prior, self.posterior = _Prior(), _Posterior()
        self._energies = energies

    def F_zx(self, z, condition=None):
        n = z.shape[0]
        x = torch.zeros(n, DOFS)
        x[:, 0] = self._energies[:n]
        return x, torch.zeros(n, 1)


def _propagator(flow):
    """A propagator with the given flow, bypassing __init__ (which would build a
    Trainer and a base sampler that these tests do not exercise)."""
    p = object.__new__(nflows_propagator)
    p.flow = flow
    p.device = DEVICE
    p.transform = False
    p.max_sample_size = 1000
    p.max_generation_attempts = None
    p.generating_counter = 0
    p.empty_pool = True
    p.pool_size = None
    p.pool_cursor = 0
    p.average_attempts = 0
    p.n_sample_space = 0
    return p


# --------------------------------------------------------------------------
# generate(): the all-rejected batch
# --------------------------------------------------------------------------

def test_batch_entirely_above_the_bound_is_rejected():
    """Every proposal above the bound gets the same -inf weight. Comparing each
    weight with the batch maximum then made the whole batch look acceptable."""
    bound = 1.0
    flow = _Flow(torch.full((1000,), 5.0))          # everything above the bound
    p = _propagator(flow)
    p.max_generation_attempts = 3                   # or generate() never returns

    with tempfile.TemporaryDirectory() as d:
        p.generate(n_pool=100, energy_bound=bound, outputdir=d, disable_pbar=True)

    # generate() bailed out, so the pool is still the list it starts as.
    pooled = sum(t.shape[0] for t in p.pool)
    assert pooled == 0, f"{pooled} configurations above the bound entered the pool"


def test_mixed_batch_pools_only_configurations_below_the_bound():
    bound = 1.0
    energies = torch.full((1000,), 5.0)
    energies[::4] = 0.0                              # a quarter are admissible
    p = _propagator(_Flow(energies))

    with tempfile.TemporaryDirectory() as d:
        p.generate(n_pool=50, energy_bound=bound, outputdir=d, disable_pbar=True)

    assert p.pool.shape[0] == 50
    assert bool((p.pool[:, 0] < bound).all()), "pooled a configuration above the bound"


# --------------------------------------------------------------------------
# sample_space(): shapes, and consumption without replacement
# --------------------------------------------------------------------------

def test_sample_space_returns_a_batch_axis_for_n_equal_one():
    """N = 1 is the only case the nested sampler uses, and it is the case where
    squeeze() collapsed the batch axis and the mask then added a spurious one."""
    p = _propagator(_Flow(torch.zeros(10)))
    p.pool = torch.zeros(4, DOFS)                    # energy 0, below the bound
    p.pool_cursor, p.empty_pool = 0, False

    x, u, _ = p.sample_space(N=1, energy_bound=1.0)
    assert x.shape == (1, DOFS), x.shape
    assert u.shape == (1,), u.shape


def test_exhausted_pool_reports_no_replacement():
    """An empty pool must say so rather than hand back a copy of a live point:
    the caller drops the iteration instead of spending a step on a walker that
    was never sampled."""
    p = _propagator(_Flow(torch.zeros(10)))
    p.pool = torch.zeros(0, DOFS)
    p.pool_cursor, p.n_sample_space, p.empty_pool = 0, 1, False

    x, u, _ = p.sample_space(N=1, energy_bound=1.0)
    assert x is None and u is None, "the exhausted pool invented a replacement"
    assert p.empty_pool is True
    assert p.pool_size is None


def test_pool_is_consumed_without_replacement():
    p = _propagator(_Flow(torch.zeros(10)))
    p.x0 = torch.zeros(1, DOFS)
    n = 40
    p.pool = torch.zeros(n, DOFS)
    p.pool[:, 1] = torch.arange(n, dtype=torch.float32)   # a tag per configuration
    p.pool_cursor, p.empty_pool = 0, False

    seen = []
    while not p.empty_pool:
        x, _, _ = p.sample_space(N=1, energy_bound=1.0)
        if not p.empty_pool:
            seen.append(float(x[0, 1]))

    assert len(seen) == n, f"drew {len(seen)} of {n}"
    assert len(set(seen)) == n, "a configuration was drawn twice"
    assert sorted(seen) == list(range(n))


# --------------------------------------------------------------------------
# nested_sampling(): the training window
# --------------------------------------------------------------------------

class _WindowRecorder:
    """Stands in for the dataset so the test can see what the window holds."""

    seen = []

    def __init__(self, flow, data_tensor, conditions_tensor, **kwargs):
        type(self).seen.append((data_tensor.clone(), conditions_tensor.clone()))
        self.transform = kwargs.get("transform", False)

    def __len__(self):
        return 1


class _StubSystem:
    """Energy is the first coordinate, so the test dictates which walker sits at
    the bound and therefore which slot the next replacement goes into."""

    PBC = False
    n_particles, dimensions, dofs = DOFS // 2, 2, DOFS
    device = DEVICE
    box_length = torch.ones(2)

    def energy(self, x):
        return x[:, :1].clone()

    def init_conf(self, random=True):
        x = torch.zeros(DOFS)
        x[0] = float(torch.randint(1000, 2000, (1,)))   # distinct, high energies
        x[1] = float(torch.randint(0, 10 ** 6, (1,)))   # a tag, to tell copies apart
        return x


class _StubPropagator:
    """Hands out a fixed number of replacements per pool, then reports empty.

    Each replacement is strictly lower in energy than anything alive, so the
    live set turns over slot by slot, and carries a unique tag, so that copies
    of one configuration are distinguishable from genuinely new ones.

    ``failures_per_pool`` makes generate() give up without filling the pool, the
    way it does when max_generation_attempts is exceeded, so the retraining loop
    runs more than once for a single live set.
    """

    def __init__(self, system, draws_per_pool, failures_per_pool=0):
        self.system = system
        self.draws_per_pool = draws_per_pool
        self.failures_per_pool = failures_per_pool
        self.device = DEVICE
        self.transform = False
        self.flow = type("F", (), {"posterior": type("P", (), {"PBC": False})()})()
        self.empty_pool = True
        self.training_counter = 0
        self.pool_size = None
        self.average_attempts = 0.0
        self._left = 0
        self._failed = 0
        self._served = 0
        self.exhaustions = 0
        self.x0 = None

    def train(self, train_dataset, training_protocol, fine_tune, outputdir="./"):
        self.training_counter += 1

    def generate(self, n_pool, energy_bound, outputdir="./", disable_pbar=False,
                 save_biased_pool=False):
        if self._failed < self.failures_per_pool:
            self._failed += 1            # gave up early: empty_pool stays True
            return
        self._failed = 0
        self._left = self.draws_per_pool
        self.empty_pool = False

    def sample_space(self, N, energy_bound):
        if self._left == 0:
            self.empty_pool = True
            self.pool_size = None
            self.exhaustions += 1
            return None, None, 0          # no replacement: not an iteration
        self._left -= 1
        self._served += 1
        self.pool_size = self._left
        x = torch.zeros(1, DOFS)
        x[0, 0] = -float(self._served)          # below every live walker
        x[0, 1] = 10.0 ** 6 + self._served      # unique tag
        return x, self.system.energy(x).squeeze(-1), 1


def _run_window_test(failures_per_pool, tmpdir, iprint=0, max_iters=None,
                     alternate_std_ns_iters=0):
    K, draws = 12, 20
    system = _StubSystem()
    std = rejection_monte_carlo(system=system, n_cycles=2, step_size=0.05, transform=False)
    prop = _StubPropagator(system, draws_per_pool=draws,
                           failures_per_pool=failures_per_pool)

    _WindowRecorder.seen = []
    original = ns_module.ConditionedDataset
    ns_module.ConditionedDataset = _WindowRecorder
    try:
        nested_sampling(K=K, system=system, std_propagator=std, nf_propagator=prop,
                        max_iters=max_iters or 4 * draws + 6, n_propagate=1, turn_on_nf=1,
                        alternate_std_ns_iters=alternate_std_ns_iters, n_pool=10, itrain=0,
                        reinitialize_nf_parameters=False,   # as the paper's runs do
                        cumulate_n_dataset=3, training_protocol=[{}],
                        outputdir=tmpdir, iprint=iprint, isavesamp=0,
                        disable_pbar=True, update_step=False)
    finally:
        ns_module.ConditionedDataset = original
    return K, _WindowRecorder.seen, prop


def _assert_window_is_distinct_live_sets(data, conds, K, n_sets, where=""):
    """The window must hold n_sets *different* live sets, every walker distinct.

    Two separate defects showed up here. The window collapsed onto the latest
    live set, repeated, because the deque held aliases of one tensor; and an
    exhausted pool used to replace a walker with an unevolved copy of another,
    so a live set carried duplicates of its own members. With both fixed the
    stand-in propagator hands out a distinct configuration every time, and every
    row in the window should be unique.
    """
    assert data.shape[0] == n_sets * K, f"{where}: {data.shape} rows"
    blocks = [data[i * K:(i + 1) * K] for i in range(n_sets)]
    for i in range(n_sets):
        for j in range(i + 1, n_sets):
            assert not torch.equal(blocks[i], blocks[j]), (
                f"{where}: live sets {i} and {j} are the same tensor")
    distinct = torch.unique(data, dim=0).shape[0]
    assert distinct == n_sets * K, (
        f"{where}: {distinct} distinct configurations of {n_sets * K} rows "
        f"({n_sets} live sets of {K})")
    assert torch.unique(conds).numel() == n_sets, (
        f"{where}: {torch.unique(conds).numel()} energy bounds for {n_sets} live sets")


def test_training_window_holds_distinct_live_sets():
    """The window is the last three live sets, not three copies of the latest."""
    with tempfile.TemporaryDirectory() as d:
        K, seen, _ = _run_window_test(failures_per_pool=0, tmpdir=d)

    assert len(seen) >= 3, f"only {len(seen)} retrainings; the test needs three"
    data, conds = seen[2]                       # the first full window
    _assert_window_is_distinct_live_sets(data, conds, K, 3, "full window")


def test_retrying_generation_does_not_duplicate_the_live_set():
    """A pool that fails to fill must not push the live set in a second time.

    Every failed attempt rebuilds the dataset, so one live set is recorded once
    per attempt. Those recordings must be identical: under the defect each retry
    appended the live set again, so the window grew by a copy of itself every
    time the pool failed to fill.
    """
    with tempfile.TemporaryDirectory() as d:
        K, seen, _ = _run_window_test(failures_per_pool=2, tmpdir=d)

    assert len(seen) >= 3
    for i, (data, conds) in enumerate(seen):
        assert data.shape[0] % K == 0, (i, data.shape)
        n_sets = data.shape[0] // K
        assert n_sets <= 3, f"retraining {i}: window holds {n_sets} live sets, max 3"
        _assert_window_is_distinct_live_sets(data, conds, K, n_sets, f"retraining {i}")

    # The retries must have actually happened, or the test proves nothing.
    repeats = sum(1 for a, b in zip(seen, seen[1:]) if torch.equal(a[0], b[0]))
    assert repeats > 0, "generation never retried; the retry path was not exercised"
    assert seen[-1][0].shape[0] == 3 * K, seen[-1][0].shape


def test_exhausted_pool_does_not_consume_an_iteration():
    """An exhausted pool replaces no walker, so it must not advance the counter.

    Before, it wrote an unevolved copy of another live point into the slot at the
    bound and counted the step anyway, contracting the prior volume for a step
    that never happened. The check is that the iterations logged are exactly
    1..max_iters with none missing and none repeated, even though the pool ran
    out several times along the way.
    """
    max_iters = 90
    with tempfile.TemporaryDirectory() as d:
        _, _, prop = _run_window_test(failures_per_pool=0, tmpdir=d, iprint=1,
                                      max_iters=max_iters, alternate_std_ns_iters=3)
        logged = [int(line.split()[0])
                  for line in open(os.path.join(d, "output.txt")) if line.strip()]

    assert prop.exhaustions > 0, "the pool never ran out; the test proves nothing"
    assert logged[0] == 0, logged[:3]
    steps = logged[1:]
    assert steps == list(range(1, max_iters + 1)), (
        f"{len(steps)} iterations logged, {len(set(steps))} distinct, "
        f"expected {max_iters} consecutive")


# --------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
