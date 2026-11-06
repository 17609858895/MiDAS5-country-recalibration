"""Shared loaders and compositional transforms."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "processed"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
MIN_DEPTH = 2000


def load_meta():
    m = pd.read_csv(PROC / "metadata_clean.csv", parse_dates=["Date"])
    return m.set_index("SampleID")


def load_counts(amp, rank="g"):
    return pd.read_parquet(PROC / f"{amp}_{rank}_counts.parquet")


def rel_abund(counts):
    return counts.div(counts.sum(1), axis=0)


def clr(counts, pseudo=0.5):
    x = np.log(counts + pseudo)
    return x.sub(x.mean(1), axis=0)


def filter_taxa(rel, min_prev=0.1, min_mean=1e-4):
    keep = ((rel > 0).mean() >= min_prev) & (rel.mean() >= min_mean)
    return keep[keep].index


def amp_features(amp, rank="g", samples=None, min_prev=0.1, min_mean=1e-4, taxa=None):
    c = load_counts(amp, rank)
    c = c[c.sum(1) >= MIN_DEPTH]
    if samples is not None:
        c = c.reindex([s for s in samples if s in c.index])
    rel = rel_abund(c)
    if taxa is None:
        taxa = filter_taxa(rel, min_prev, min_mean)
    rel_k = rel.reindex(columns=taxa, fill_value=0.0)
    other = (1 - rel_k.sum(1)).clip(lower=0)
    comp = rel_k.copy()
    comp["_other"] = other
    # CLR on proportions with multiplicative pseudo-count
    x = np.log(comp + 1e-5)
    x = x.sub(x.mean(1), axis=0).drop(columns="_other")
    x.columns = [f"{amp}|{t}" for t in x.columns]
    return x, list(taxa), rel
