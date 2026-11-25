"""PyTorch two-tower retrieval with negative-sampling, sequential-history and cold-start variants.

    python -m reelrank.retrieval.two_tower --tags R1 R2 --variants main inbatch uniform ...

Runs in its own process: PyTorch must never share a process with FAISS or LightGBM (see reelrank.runtime).

User tower: concat of the user-ID embedding, encoded positive history (mean-pool, GRU or SASRec over the last 50 positively
engaged videos, sharing the item-ID table), standardised activity/watch stats and category affinity.
Item tower: concat of the item-ID embedding, category/duration-bucket/creator/tag embeddings and lagged popularity /
engagement / platform statistics. Both: MLP 256 -> 128 -> 64 with ReLU + dropout, then L2-normalised.
Item inputs never depend on the user, so item vectors can be indexed per snapshot.

score(u, v) = <u, v> / temperature, trained with softmax cross-entropy over the positive and a set of negatives.
Batches are drawn from a single snapshot day so in-batch and sampled negatives see the same item features.
"""

from __future__ import annotations

from reelrank.runtime import claim

claim("torch")

import resource  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict, dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from reelrank.config import Config, cli, day_to_date, read_json, write_json  # noqa: E402
from reelrank.evaluation.protocol import load_protocol, recommend  # noqa: E402
from reelrank.tracking import mlflow_run  # noqa: E402

# ------------------------------------------------------------------------------------------- variants
VARIANTS: dict[str, dict] = {
    # name: negatives, logq correction, metadata features, history encoder, extras
    # in-batch negatives are drawn from the batch's own day, so logQ must use that day's item frequency
    "main": {"negatives": "inbatch", "logq": True, "metadata": True, "encoder": "mean"},
    "inbatch": {"negatives": "inbatch", "logq": False, "metadata": True, "encoder": "mean"},
    "inbatch_globalq": {"negatives": "inbatch", "logq": True, "metadata": True, "encoder": "mean",
                        "logq_scope": "global"},
    "uniform": {"negatives": "uniform", "logq": False, "metadata": True, "encoder": "mean"},
    "freq": {"negatives": "freq", "logq": False, "metadata": True, "encoder": "mean"},
    "freq_logq": {"negatives": "freq", "logq": True, "metadata": True, "encoder": "mean"},
    "hard": {"negatives": "hard", "logq": True, "metadata": True, "encoder": "mean"},
    "id_only": {"negatives": "inbatch", "logq": True, "metadata": False, "encoder": "mean"},
    "temp01": {"negatives": "inbatch", "logq": True, "metadata": True, "encoder": "mean", "temperature": 0.1},
    "gru": {"negatives": "inbatch", "logq": True, "metadata": True, "encoder": "gru"},
    "sasrec": {"negatives": "inbatch", "logq": True, "metadata": True, "encoder": "sasrec"},
    "cold": {"negatives": "inbatch", "logq": True, "metadata": True, "encoder": "mean", "cold_items": True},
    "cold_id_only": {"negatives": "inbatch", "logq": True, "metadata": False, "encoder": "mean",
                     "cold_items": True},
}


@dataclass
class Spec:
    variant: str
    negatives: str
    logq: bool
    metadata: bool
    encoder: str
    temperature: float
    cold_items: bool = False
    logq_scope: str = "day"   # "day": frequency within the batch's snapshot day (the actual sampler); "global"
    extra: dict = field(default_factory=dict)


def make_spec(cfg: Config, variant: str) -> Spec:
    v = dict(VARIANTS[variant])
    return Spec(variant=variant, negatives=v["negatives"], logq=v["logq"], metadata=v["metadata"],
                encoder=v["encoder"], temperature=float(v.get("temperature", cfg["two_tower"]["temperature"])),
                cold_items=bool(v.get("cold_items", False)), logq_scope=v.get("logq_scope", "day"))


# ----------------------------------------------------------------------------------------------- data
class FeatureData:
    """In-memory snapshot features (all arrays are indexed [snapshot, entity, ...])."""

    def __init__(self, cfg: Config):
        fdir = cfg.processed / "features"
        self.man = read_json(fdir / "manifest.json")
        self.U, self.I = self.man["users"], self.man["items"]
        self.snap_days = np.load(fdir / "snap_days.npy")
        self.user_stats = np.load(fdir / "user_stats.npy")
        self.user_cat = np.load(fdir / "user_cat.npy")
        self.user_hist = np.load(fdir / "user_hist.npy")
        self.user_hist_wr = np.load(fdir / "user_hist_wr.npy").astype(np.float32)
        item_stats = np.load(fdir / "item_stats.npy")
        item_platform = np.load(fdir / "item_platform.npy")
        st = dict(np.load(fdir / "item_static.npz"))
        self.static = st
        pop = np.load(fdir / "item_pop_before.npy")
        # item age in days at each snapshot, using the earlier of upload day and first observed view
        first_seen = np.where(pop > 0, 1, 0).argmax(0)
        first_day = np.where(pop.max(0) > 0, self.snap_days[np.maximum(first_seen - 1, 0)], 10**6)
        born = np.minimum(np.where(st["upload_day"] >= 0, st["upload_day"], 10**6), first_day)
        age = np.log1p(np.clip(self.snap_days[:, None] - born[None, :], 0, 400)).astype(np.float32)
        self.item_dyn = np.concatenate([item_stats, item_platform, age[..., None]], axis=-1)
        self.pop_before = pop
        self.n_cat = self.man["categories"]
        self.n_tags = self.man["tags"]
        self.n_authors = self.man["authors"]
        self.n_dur = int(st["dur_bucket"].max()) + 1

    def snap_of_day(self, day: int) -> int:
        s = int(np.searchsorted(self.snap_days, day))
        assert self.snap_days[s] == day, day
        return s


class Norm:
    """Feature standardisation fitted on training snapshots only (saved with the checkpoint)."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean, self.std = mean.astype(np.float32), np.maximum(std, 1e-6).astype(np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


# ---------------------------------------------------------------------------------------------- model
def mlp(d_in: int, hidden: list[int], d_out: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = d_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
        d = h
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


class TwoTower(nn.Module):
    def __init__(self, fd: FeatureData, spec: Spec, cfg: Config):
        super().__init__()
        tc = cfg["two_tower"]
        dim, hidden, drop = int(tc["dim"]), list(tc["hidden"]), float(tc["dropout"])
        self.spec, self.I, self.U = spec, fd.I, fd.U
        self.L = fd.user_hist.shape[2]
        self.UNK, self.PAD = fd.I, fd.I + 1
        self.item_id = nn.Embedding(fd.I + 2, dim, padding_idx=self.PAD)
        self.user_id = nn.Embedding(fd.U + 1, dim)
        nn.init.normal_(self.item_id.weight, std=0.05)
        nn.init.normal_(self.user_id.weight, std=0.05)
        with torch.no_grad():
            self.item_id.weight[self.PAD].zero_()
        d_user = dim + dim
        d_item = dim
        if spec.metadata:
            self.cat = nn.Embedding(fd.n_cat, 16)
            self.dur = nn.Embedding(fd.n_dur, 8)
            self.author = nn.Embedding(fd.n_authors, 16)
            self.tags = nn.Linear(fd.n_tags, 16, bias=False)
            d_user += fd.user_stats.shape[2] + fd.n_cat
            d_item += 16 + 8 + 16 + 16 + 1 + fd.item_dyn.shape[2]
        if spec.encoder == "gru":
            self.gru = nn.GRU(dim + 1, dim, batch_first=True)
        elif spec.encoder == "sasrec":
            sc = cfg["sequential"]
            self.pos = nn.Embedding(self.L + 1, dim)
            self.wr_proj = nn.Linear(1, dim)
            self.empty = nn.Parameter(torch.zeros(dim))
            layer = nn.TransformerEncoderLayer(dim, int(sc["heads"]), dim * 2, dropout=drop, batch_first=True,
                                               norm_first=True)
            self.sas = nn.TransformerEncoder(layer, int(sc["layers"]), enable_nested_tensor=False)
            self.register_buffer("causal", torch.triu(torch.ones(self.L + 1, self.L + 1, dtype=torch.bool), 1))
        self.user_mlp = mlp(d_user, hidden, dim, drop)
        self.item_mlp = mlp(d_item, hidden, dim, drop)

    # history ids use -1 padding on disk; map to PAD (and held-out items to PAD for the cold experiment)
    def encode_history(self, hist: torch.Tensor, hist_wr: torch.Tensor) -> torch.Tensor:
        valid = hist >= 0
        ids = torch.where(valid, hist, torch.full_like(hist, self.PAD))
        emb = self.item_id(ids)
        if self.spec.encoder == "mean":
            n = valid.sum(1, keepdim=True).clamp(min=1)
            return emb.sum(1) / n
        if self.spec.encoder == "gru":
            lengths = valid.sum(1)
            # left-padded (oldest -> newest); left-align so packing sees the real items first
            order = torch.argsort((~valid).to(torch.int8), dim=1, stable=True)
            x = torch.cat([emb, hist_wr.unsqueeze(-1)], -1)
            x = torch.gather(x, 1, order.unsqueeze(-1).expand_as(x))
            out = torch.zeros(len(hist), emb.shape[-1], device=emb.device)
            nz = lengths > 0
            if nz.any():
                packed = nn.utils.rnn.pack_padded_sequence(x[nz], lengths[nz].cpu(), batch_first=True,
                                                           enforce_sorted=False)
                _, h = self.gru(packed)
                out[nz] = h[-1]
            return out
        # SASRec-style causal self-attention; an always-valid "empty" token at the end reads the sequence
        B = len(hist)
        x = emb + self.wr_proj(hist_wr.unsqueeze(-1)) * valid.unsqueeze(-1)
        x = torch.cat([x, self.empty.expand(B, 1, -1)], 1)
        x = x + self.pos(torch.arange(self.L + 1, device=x.device)).unsqueeze(0)
        pad = torch.cat([~valid, torch.zeros(B, 1, dtype=torch.bool, device=x.device)], 1)
        h = self.sas(x, mask=self.causal, src_key_padding_mask=pad)
        return h[:, -1]

    def user_forward(self, uid, stats, cat, hist, hist_wr) -> torch.Tensor:
        parts = [self.user_id(uid), self.encode_history(hist, hist_wr)]
        if self.spec.metadata:
            parts += [stats, cat]
        return F.normalize(self.user_mlp(torch.cat(parts, -1)), dim=-1)

    def item_forward(self, iid, cat, dur, author, tags, log_dur, dyn) -> torch.Tensor:
        parts = [self.item_id(iid)]
        if self.spec.metadata:
            parts += [self.cat(cat), self.dur(dur), self.author(author), self.tags(tags), log_dur.unsqueeze(-1), dyn]
        return F.normalize(self.item_mlp(torch.cat(parts, -1)), dim=-1)


# -------------------------------------------------------------------------------------------- trainer
class Trainer:
    def __init__(self, cfg: Config, fd: FeatureData, spec: Spec, tag: str, cutoff: int):
        self.cfg, self.fd, self.spec, self.tag, self.cutoff = cfg, fd, spec, tag, cutoff
        tc = cfg["two_tower"]
        self.tc = tc
        torch.manual_seed(cfg.seed)
        torch.set_num_threads(int(tc["threads"]))
        self.rng = np.random.default_rng(cfg.seed)
        ev = pl.read_parquet(cfg.processed / "features" / "events.parquet",
                             columns=["user_id", "video_id", "day", "snap", "pos"])
        train = ev.filter((pl.col("day") < cutoff) & pl.col("pos"))
        self.cut_snap = fd.snap_of_day(cutoff)
        self.held_out = np.zeros(fd.I, dtype=bool)
        if spec.cold_items:
            seen = np.unique(train["video_id"].to_numpy())
            n = int(round(float(tc["cold_item_frac"]) * len(seen)))
            self.held_out[np.random.default_rng(cfg.seed + 1).choice(seen, n, replace=False)] = True
            train = train.filter(~pl.col("video_id").is_in(np.where(self.held_out)[0]))
        self.train = train
        self.tu = train["user_id"].to_numpy()
        self.tv = train["video_id"].to_numpy()
        self.ts = train["snap"].to_numpy()
        keys = np.unique(self.tu.astype(np.int64) * fd.I + self.tv)
        self.pos_keys = torch.from_numpy(keys)
        cnt = np.bincount(self.tv, minlength=fd.I).astype(np.float64)
        self.item_count = cnt
        q = (cnt + 1.0) / (cnt + 1.0).sum()
        self.log_q = torch.from_numpy(np.log(q).astype(np.float32))
        # per-day sampling distribution of in-batch negatives (batches never mix days)
        self.day_log_q: dict[int, torch.Tensor] = {}
        for snap in np.unique(self.ts):
            c = np.bincount(self.tv[self.ts == snap], minlength=fd.I).astype(np.float64)
            self.day_log_q[int(snap)] = torch.from_numpy(np.log((c + 0.1) / (c + 0.1).sum()).astype(np.float32))
        train_items = np.where(cnt > 0)[0]
        self.neg_pool = train_items
        w = cnt[train_items] ** float(tc["freq_power"])
        self.freq_p = w / w.sum()
        # feature normalisation from training snapshots only
        snaps = np.unique(self.ts)
        us = fd.user_stats[snaps].reshape(-1, fd.user_stats.shape[2])
        idyn = fd.item_dyn[snaps].reshape(-1, fd.item_dyn.shape[2])
        self.unorm = Norm(us.mean(0), us.std(0))
        self.inorm = Norm(idyn.mean(0), idyn.std(0))
        self.model = TwoTower(fd, spec, cfg)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=float(tc["lr"]),
                                     weight_decay=float(tc["weight_decay"]))
        self.hard = self._hard_candidates() if spec.negatives == "hard" else None
        self.st = {k: torch.from_numpy(np.asarray(v)) for k, v in fd.static.items()
                   if k in ("first_cat", "dur_bucket", "author_idx", "log_dur", "tags")}

    def _hard_candidates(self) -> np.ndarray:
        """Per user: top-k ALS items (same training cutoff) that are not known positives."""
        a = np.load(self.cfg.artifacts / "als" / f"{self.tag}.npz")
        uf, vf = a["user"], a["item"]
        k = int(self.tc["hard_k"])
        out = np.empty((self.fd.U, k), dtype=np.int64)
        keys = self.pos_keys.numpy()
        for lo in range(0, self.fd.U, 1024):
            s = uf[lo: lo + 1024] @ vf.T
            s[:, self.item_count == 0] = -np.inf
            idx = np.argpartition(-s, 2 * k, axis=1)[:, : 2 * k]
            for r in range(len(idx)):
                u = lo + r
                cand = idx[r][np.argsort(-s[r, idx[r]])]
                kk = u * self.fd.I + cand
                known = np.isin(kk, keys[np.searchsorted(keys, u * self.fd.I): np.searchsorted(keys, (u + 1) * self.fd.I)])
                cand = cand[~known][:k]
                out[u] = np.resize(cand, k) if len(cand) else self.rng.choice(self.neg_pool, k)
        return out

    # feature gathering
    def user_inputs(self, users: np.ndarray, snap: int):
        fd = self.fd
        hist = fd.user_hist[snap, users].astype(np.int64)
        if self.held_out.any():
            hist = np.where((hist >= 0) & self.held_out[np.maximum(hist, 0)], -1, hist)
        return (torch.from_numpy(users.astype(np.int64)),
                torch.from_numpy(self.unorm(fd.user_stats[snap, users])),
                torch.from_numpy(fd.user_cat[snap, users]),
                torch.from_numpy(hist), torch.from_numpy(fd.user_hist_wr[snap, users]))

    def item_inputs(self, items: np.ndarray, snap: int, id_dropout: float = 0.0):
        items = np.asarray(items, dtype=np.int64)
        ids = items.copy()
        dyn = self.fd.item_dyn[snap, items].copy()
        if self.held_out.any():
            cold = self.held_out[items]
            ids[cold] = self.model.UNK
            dyn[cold] = 0.0  # a held-out item has no in-dataset or platform history
        if id_dropout > 0:
            ids[self.rng.random(len(ids)) < id_dropout] = self.model.UNK
        t = torch.from_numpy(items)
        return (torch.from_numpy(ids), self.st["first_cat"][t], self.st["dur_bucket"][t], self.st["author_idx"][t],
                self.st["tags"][t], self.st["log_dur"][t], torch.from_numpy(self.inorm(dyn)))

    def known(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Boolean mask: is (u[i], v[j]) a training positive?  u: [B, 1], v: [1, M] or [B, M]."""
        q = u * self.fd.I + v
        idx = torch.searchsorted(self.pos_keys, q.reshape(-1)).clamp(max=len(self.pos_keys) - 1)
        return (self.pos_keys[idx] == q.reshape(-1)).reshape(q.shape)

    # loss
    def batch_loss(self, users: np.ndarray, items: np.ndarray, snap: int) -> torch.Tensor:
        sp_, tau = self.spec, self.spec.temperature
        u = self.model.user_forward(*self.user_inputs(users, snap))
        v = self.model.item_forward(*self.item_inputs(items, snap, float(self.tc["id_dropout"])))
        ut = torch.from_numpy(users.astype(np.int64))
        vt = torch.from_numpy(items.astype(np.int64))
        B = len(users)
        eye = torch.eye(B, dtype=torch.bool)
        if sp_.negatives in ("inbatch", "hard"):
            logits = u @ v.T / tau
            if sp_.logq:
                lq = self.day_log_q[snap] if sp_.logq_scope == "day" else self.log_q
                logits = logits - lq[vt].unsqueeze(0)
            # mask duplicate items and other known positives of the row user (multi-positive collisions)
            mask = self.known(ut.unsqueeze(1), vt.unsqueeze(0)) & ~eye
            logits = logits.masked_fill(mask, float("-inf"))
            if sp_.negatives == "hard":
                h = self.hard[users, self.rng.integers(0, self.hard.shape[1], B)]
                hv = self.model.item_forward(*self.item_inputs(h, snap))
                hl = (u * hv).sum(-1, keepdim=True) / tau
                hl = hl.masked_fill(torch.from_numpy(h == items).unsqueeze(1), float("-inf"))
                logits = torch.cat([logits, hl], 1)
            return F.cross_entropy(logits, torch.arange(B))
        # shared sampled negatives
        M = int(self.tc["n_sampled"])
        if sp_.negatives == "uniform":
            neg = self.rng.choice(self.neg_pool, M)
            logq_neg = torch.full((M,), float(-np.log(len(self.neg_pool))))
            logq_pos = torch.full((B,), float(-np.log(len(self.neg_pool))))
        else:
            neg = self.rng.choice(self.neg_pool, M, p=self.freq_p)
            full = np.zeros(self.fd.I)
            full[self.neg_pool] = self.freq_p
            logq_neg = torch.from_numpy(np.log(full[neg] + 1e-12).astype(np.float32))
            logq_pos = torch.from_numpy(np.log(full[items] + 1e-12).astype(np.float32))
        nv = self.model.item_forward(*self.item_inputs(neg, snap))
        pos_l = (u * v).sum(-1, keepdim=True) / tau
        neg_l = u @ nv.T / tau
        if sp_.logq:
            pos_l = pos_l - logq_pos.unsqueeze(1)
            neg_l = neg_l - logq_neg.unsqueeze(0)
        neg_l = neg_l.masked_fill(self.known(ut.unsqueeze(1), torch.from_numpy(neg).unsqueeze(0)), float("-inf"))
        return F.cross_entropy(torch.cat([pos_l, neg_l], 1), torch.zeros(B, dtype=torch.long))

    def epoch_batches(self) -> list[tuple[int, np.ndarray]]:
        """Sample an epoch of positives (recency-weighted if configured), grouped into same-day batches."""
        n = len(self.tu)
        m = min(n, int(self.tc["examples_per_epoch"]))
        hl = self.tc.get("sample_half_life_days")
        if hl:
            age = self.cutoff - self.fd.snap_days[self.ts]
            w = 0.5 ** (age / float(hl))
            pick = self.rng.choice(n, m, replace=False, p=w / w.sum())
        else:
            pick = self.rng.choice(n, m, replace=False) if m < n else self.rng.permutation(n)
        pick = pick[np.argsort(self.ts[pick], kind="stable")]
        B = int(self.tc["batch_size"])
        out = []
        bounds = np.flatnonzero(np.diff(self.ts[pick])) + 1
        for grp in np.split(pick, bounds):
            grp = self.rng.permutation(grp)
            for i in range(0, len(grp) - B // 4, B):  # drop tiny tails (too few in-batch negatives)
                out.append((int(self.ts[grp[0]]), grp[i: i + B]))
        order = self.rng.permutation(len(out))
        return [out[i] for i in order]

    # embeddings
    @torch.no_grad()
    def item_vectors(self, snap: int) -> np.ndarray:
        self.model.eval()
        out = [self.model.item_forward(*self.item_inputs(np.arange(lo, min(lo + 4096, self.fd.I)), snap)).numpy()
               for lo in range(0, self.fd.I, 4096)]
        return np.concatenate(out).astype(np.float32)

    @torch.no_grad()
    def user_vectors(self, snap: int, users: np.ndarray | None = None) -> np.ndarray:
        self.model.eval()
        users = np.arange(self.fd.U) if users is None else users
        out = [self.model.user_forward(*self.user_inputs(users[lo: lo + 2048], snap)).numpy()
               for lo in range(0, len(users), 2048)]
        return np.concatenate(out).astype(np.float32)

    def fit(self, epochs: int, val_protocol=None, log=None) -> dict:
        hist = []
        best, best_state, best_epoch, bad = -1.0, None, 0, 0
        t_all = time.time()
        n_ex = 0
        for ep in range(1, epochs + 1):
            self.model.train()
            t = time.time()
            losses = []
            for snap, idx in self.epoch_batches():
                loss = self.batch_loss(self.tu[idx], self.tv[idx], snap)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                losses.append(loss.item())
                n_ex += len(idx)
            row = {"epoch": ep, "loss": float(np.mean(losses)), "seconds": time.time() - t}
            if val_protocol is not None:
                row.update(self.evaluate(val_protocol))
                score = row["recall@100"]
                if score > best:
                    best, best_epoch, bad = score, ep, 0
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                else:
                    bad += 1
            hist.append(row)
            if log:
                log({k: v for k, v in row.items() if k != "epoch"}, step=ep)
            print(f"    [{self.tag}/{self.spec.variant}] epoch {ep} loss={row['loss']:.4f}"
                  + (f" val recall@100={row['recall@100']:.4f}" if "recall@100" in row else "")
                  + f" ({row['seconds']:.0f}s)", flush=True)
            if val_protocol is not None and bad >= int(self.tc["patience"]):
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        secs = time.time() - t_all
        return {"history": hist, "best_epoch": best_epoch or epochs, "train_seconds": secs,
                "examples_per_second": n_ex / max(secs, 1e-9),
                "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024),
                "train_positives": int(len(self.tu)), "parameters": int(sum(p.numel() for p in self.model.parameters()))}

    def evaluate(self, p) -> dict:
        iv = self.item_vectors(p.snap)
        uv = self.user_vectors(p.snap, p.users)
        pos = {u: i for i, u in enumerate(p.users)}
        rec = recommend(p, lambda us: uv[[pos[x] for x in us]] @ iv.T, 300)
        m = p.evaluate(rec, [50, 100, 300], [20])
        return {k: float(np.nanmean(m[k])) for k in ("recall@50", "recall@100", "recall@300", "ndcg@20")}

    # persistence
    def save(self, out: Path, info: dict, export_days: list[int]) -> None:
        out.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.model.state_dict(), "optimizer": self.opt.state_dict(),
                    "spec": asdict(self.spec), "seed": self.cfg.seed, "tag": self.tag, "cutoff": self.cutoff,
                    "vocab": {"users": self.fd.U, "items": self.fd.I, "categories": self.fd.n_cat,
                              "tags": self.fd.n_tags, "authors": self.fd.n_authors, "dur_buckets": self.fd.n_dur},
                    "user_norm": (self.unorm.mean, self.unorm.std), "item_norm": (self.inorm.mean, self.inorm.std),
                    "held_out": np.where(self.held_out)[0]}, out / "model.pt")
        weights = {k: v.detach().numpy() for k, v in self.model.state_dict().items()}
        np.savez(out / "weights.npz", **weights, user_norm_mean=self.unorm.mean, user_norm_std=self.unorm.std,
                 item_norm_mean=self.inorm.mean, item_norm_std=self.inorm.std, held_out=self.held_out)
        vecs = {}
        for d in export_days:
            s = self.fd.snap_of_day(d)
            np.save(out / f"items_{day_to_date(d)}.npy", self.item_vectors(s))
            np.save(out / f"users_{day_to_date(d)}.npy", self.user_vectors(s))
            vecs[day_to_date(d)] = s
        info = {**info, "spec": asdict(self.spec), "cutoff": day_to_date(self.cutoff), "vectors": vecs,
                "held_out_items": int(self.held_out.sum())}
        write_json(out / "info.json", info)


def export_days(cfg: Config, tag: str) -> list[int]:
    days: list[int] = []
    for start, end, model in cfg.windows().values():
        if model == tag:
            days += list(range(start, end + 1))
    return sorted(set(days))


def train_one(cfg: Config, fd: FeatureData, tag: str, variant: str, val_protocol=None) -> dict:
    cutoff = cfg.model_cutoffs()[tag]
    spec = make_spec(cfg, variant)
    out = cfg.artifacts / "two_tower" / tag / variant
    tr = Trainer(cfg, fd, spec, tag, cutoff)
    tc = cfg["two_tower"]
    epochs = int(tc["epochs"]) if spec.encoder == "mean" else int(cfg["sequential"]["epochs"])
    if val_protocol is None:  # fixed epoch count chosen on validation (R1) for the same variant
        sel = cfg.artifacts / "two_tower" / "R1" / variant / "info.json"
        if sel.exists():
            epochs = int(read_json(sel)["best_epoch"])
    print(f"  training {tag}/{variant}: {len(tr.tu):,} positives before {day_to_date(cutoff)}, {epochs} epochs",
          flush=True)
    with mlflow_run(cfg, "two_tower", f"{tag}/{variant}", {**asdict(spec), "tag": tag, "epochs": epochs,
                                                            **{k: v for k, v in tc.items()}}) as log:
        info = tr.fit(epochs, val_protocol, log)
        log({"train_seconds": info["train_seconds"], "examples_per_second": info["examples_per_second"],
             "peak_rss_mb": info["peak_rss_mb"]})
    tr.save(out, info, export_days(cfg, tag) + ([cfg.protocol_cutoff("dense")] if tag == cfg.protocol_model("dense") else []))
    return info


def main() -> None:
    cfg, ns = cli(__doc__, [
        (("--tags",), {"nargs": "+", "default": ["R1", "R2", "R0", "RG"]}),
        (("--variants",), {"nargs": "+", "default": ["main"]}),
    ])
    if ns.variants == ["all"]:
        ns.variants = list(VARIANTS)
    sel = cfg.reports / "metrics" / "selection.json"
    ns.variants = [read_json(sel)["retrieval_variant"] if v == "selected" and sel.exists() else
                   ("main" if v == "selected" else v) for v in ns.variants]
    ns.variants = list(dict.fromkeys(ns.variants))
    fd = FeatureData(cfg)
    for tag in ns.tags:
        val = load_protocol(cfg, "val") if tag == "R1" else None
        for variant in ns.variants:
            train_one(cfg, fd, tag, variant, val)  # per-run info.json; the report aggregates them


if __name__ == "__main__":
    main()
