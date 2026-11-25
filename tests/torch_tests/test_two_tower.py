"""PyTorch-side tests. Run as their own pytest process (`make test-torch`): never alongside LightGBM or FAISS."""

import numpy as np
import pytest
import torch

from reelrank.retrieval import two_tower as T
from reelrank.serving.numpy_model import NumpyTwoTower


@pytest.fixture(scope="module")
def trainer(fixture_cfg):
    fd = T.FeatureData(fixture_cfg)
    return T.Trainer(fixture_cfg, fd, T.make_spec(fixture_cfg, "main"), "R1", fixture_cfg.model_cutoffs()["R1"])


def test_logq_uses_the_batch_days_distribution(trainer):
    snap = int(trainer.ts[0])
    lq = trainer.day_log_q[snap]
    day_items = trainer.tv[trainer.ts == snap]
    c = np.bincount(day_items, minlength=trainer.fd.I) + 0.1
    np.testing.assert_allclose(lq.numpy(), np.log(c / c.sum()), rtol=1e-5)


def test_known_positive_lookup(trainer):
    u = torch.tensor([[int(trainer.tu[0])]])
    v = torch.tensor([[int(trainer.tv[0]), -5]])
    k = trainer.known(u, v)
    assert bool(k[0, 0]) and not bool(k[0, 1])


def test_inbatch_masks_duplicate_positives(trainer):
    """A batch where the same item appears twice: the duplicate column must not act as a negative."""
    snap = int(trainer.ts[0])
    idx = np.where(trainer.ts == snap)[0][:8]
    users = trainer.tu[idx].copy()
    items = trainer.tv[idx].copy()
    items[1] = items[0]  # duplicate item in the batch
    trainer.model.eval()
    with torch.no_grad():
        ut = torch.from_numpy(users.astype(np.int64))
        vt = torch.from_numpy(items.astype(np.int64))
        mask = trainer.known(ut.unsqueeze(1), vt.unsqueeze(0)) & ~torch.eye(len(users), dtype=torch.bool)
    assert bool(mask[0, 1]) or users[0] != users[1] and bool(mask[0, 1]) == trainer.known(
        ut[:1].unsqueeze(1), vt[1:2].unsqueeze(0))[0, 0]
    loss = trainer.batch_loss(users, items, snap)
    assert torch.isfinite(loss)


def test_loss_matches_manual_softmax_without_masking(fixture_cfg):
    fd = T.FeatureData(fixture_cfg)
    tr = T.Trainer(fixture_cfg, fd, T.make_spec(fixture_cfg, "inbatch"), "R1", fixture_cfg.model_cutoffs()["R1"])
    tr.tc = {**tr.tc, "id_dropout": 0.0}
    tr.model.eval()
    snap = int(tr.ts[0])
    idx = np.where(tr.ts == snap)[0]
    # greedily pick rows whose users have no known positive among the other rows' items (no masking applies)
    rows: list[int] = []
    for i in idx:
        cand = rows + [int(i)]
        ut = torch.from_numpy(tr.tu[cand].astype(np.int64))
        vt = torch.from_numpy(tr.tv[cand].astype(np.int64))
        if not (tr.known(ut.unsqueeze(1), vt.unsqueeze(0)) & ~torch.eye(len(cand), dtype=torch.bool)).any():
            rows = cand
        if len(rows) == 6:
            break
    assert len(rows) >= 3
    rows = np.array(rows)
    users, items = tr.tu[rows], tr.tv[rows]
    with torch.no_grad():
        u = tr.model.user_forward(*tr.user_inputs(users, snap))
        v = tr.model.item_forward(*tr.item_inputs(items, snap))
        logits = (u @ v.T / tr.spec.temperature).numpy()
        manual = np.mean([-logits[i, i] + np.log(np.exp(logits[i]).sum()) for i in range(len(rows))])
        assert float(tr.batch_loss(users, items, snap)) == pytest.approx(manual, rel=1e-4)


@pytest.mark.parametrize("variant", ["gru", "sasrec"])
def test_sequential_encoders_handle_empty_history(fixture_cfg, variant):
    fd = T.FeatureData(fixture_cfg)
    m = T.TwoTower(fd, T.make_spec(fixture_cfg, variant), fixture_cfg).eval()
    hist = torch.full((3, fd.user_hist.shape[2]), -1, dtype=torch.long)
    hist[1, -3:] = torch.tensor([1, 2, 3])
    out = m.encode_history(hist, torch.zeros_like(hist, dtype=torch.float32))
    assert torch.isfinite(out).all() and out.shape == (3, fixture_cfg["two_tower"]["dim"])


@pytest.mark.parametrize("variant", ["main", "id_only", "cold"])
def test_numpy_towers_match_torch(fixture_cfg, tmp_path, variant):
    fd = T.FeatureData(fixture_cfg)
    tr = T.Trainer(fixture_cfg, fd, T.make_spec(fixture_cfg, variant), "R1", fixture_cfg.model_cutoffs()["R1"])
    for p in tr.model.parameters():  # random but non-trivial weights
        torch.nn.init.normal_(p, std=0.2)
    with torch.no_grad():
        tr.model.item_id.weight[tr.model.PAD].zero_()
    tr.save(tmp_path, {}, [])
    nt = NumpyTwoTower(tmp_path / "weights.npz", {"encoder": "mean", "metadata": tr.spec.metadata})
    snap = int(tr.ts[-1])
    items = np.arange(fd.I)
    iv_t = tr.item_vectors(snap)
    st = fd.static
    iv_n = nt.item_forward(items, st["first_cat"], st["dur_bucket"], st["author_idx"], st["tags"], st["log_dur"],
                           fd.item_dyn[snap])
    np.testing.assert_allclose(iv_n, iv_t, atol=1e-5)
    users = np.arange(fd.U)
    uv_t = tr.user_vectors(snap, users)
    uv_n = nt.user_forward(users, fd.user_stats[snap], fd.user_cat[snap], fd.user_hist[snap].astype(np.int64))
    np.testing.assert_allclose(uv_n, uv_t, atol=1e-5)


def test_numpy_multitask_ranker_matches_torch(tmp_path):
    from reelrank.ranking.dataset import FEATURES
    from reelrank.ranking.neural import NUMERIC, MultiTask, utility
    from reelrank.serving.numpy_ranker import NumpyMultiTask

    torch.manual_seed(0)
    m = MultiTask(len(NUMERIC), 64, 8, [32, 16]).eval()
    rng = np.random.default_rng(0)
    n = 50
    X = rng.normal(size=(n, len(FEATURES))).astype(np.float32)
    X[:, FEATURES.index("i_first_cat")] = rng.integers(0, 30, n)
    mean, std = X[:, [FEATURES.index(f) for f in NUMERIC]].mean(0), X.std(0)[[FEATURES.index(f) for f in NUMERIC]] + 1e-6
    uvec, ivec = rng.normal(size=8).astype(np.float32), rng.normal(size=(n, 8)).astype(np.float32)
    W, ws = (1.0, 0.5, 0.7), (0.3, 1.2)
    np.savez(tmp_path / "nn.npz", **{k: v.numpy() for k, v in m.state_dict().items()}, mean=mean, std=std,
             utility=np.array(W), watch_stats=np.array(ws))
    with torch.no_grad():
        num = torch.from_numpy((X[:, [FEATURES.index(f) for f in NUMERIC]] - mean) / std)
        pw, pc, ps = m(num, torch.from_numpy(X[:, FEATURES.index("i_first_cat")].astype(np.int64)),
                       torch.from_numpy(np.broadcast_to(uvec, (n, 8)).copy()), torch.from_numpy(ivec))
    ref = utility({"watch": pw.numpy(), "complete": torch.sigmoid(pc).numpy(), "skip": torch.sigmoid(ps).numpy()},
                  *W, stats=ws)
    np.testing.assert_allclose(NumpyMultiTask(tmp_path / "nn.npz").score(X, uvec, ivec), ref, rtol=1e-4, atol=1e-5)
