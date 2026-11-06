"""Build sample x taxon count matrices for the three MiDAS5 amplicons.

Inputs (extracted from the official Figshare v2 data.zip into data_extracted/):
  DataS1_AD_metadata_231220.txt, {V13,V4,V35}_ASVtab, *_metadata.txt (seqID -> SampleID, no header),
  *_ASV_vs_MiDAS_5.2.sintax (column 4 = taxonomy passing the SINTAX cutoff).
Outputs (processed/): metadata_clean.csv, {amp}_{rank}_counts.parquet, asv_taxonomy_{amp}.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_extracted"
OUT = ROOT / "processed"
OUT.mkdir(exist_ok=True)

RANKS = ["d", "p", "c", "o", "f", "g", "s"]
AMPS = {
    "V13": ("V13_ASVtab.txt", "V13_metadata.txt", "V13_ASV_vs_MiDAS_5.2.sintax"),
    "V4": ("V4_ASVtab.txt", "V4c_metadata.txt", "V4_ASV_vs_MiDAS_5.2.sintax"),
    "V35": ("V35_reverse_250bp_ASVtab.txt", "V35_metadata.txt", "V35_ASV_vs_MiDAS_5.2.sintax"),
}


def parse_sintax(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            asv = parts[0]
            tax = parts[3] if len(parts) > 3 else ""
            d = {r: None for r in RANKS}
            for tok in filter(None, tax.split(",")):
                k, v = tok.split(":", 1)
                d[k] = v
            d["ASV"] = asv
            rows.append(d)
    tax = pd.DataFrame(rows).set_index("ASV")
    # lowest-rank label: unassigned ranks inherit "<rank>__<parent>"
    for i, r in enumerate(RANKS[1:], start=1):
        parent = RANKS[i - 1]
        miss = tax[r].isna()
        tax.loc[miss, r] = "uc_" + tax.loc[miss, parent].astype(str).str.replace("^uc_", "", regex=True)
    return tax


def parse_date(s):
    if pd.isna(s):
        return pd.NaT
    s = str(s)
    if "/" in s:
        return pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    try:
        return pd.Timestamp("1899-12-30") + pd.Timedelta(days=float(s))
    except ValueError:
        return pd.NaT


meta = pd.read_csv(RAW / "DataS1_AD_metadata_231220.txt", sep="\t")
meta["Digester"] = meta.SampleID.str.replace(r"-[AB]$", "", regex=True)
meta["Date"] = meta.Sampling_date.map(parse_date)
for c in ["Digester_location", "Digester_technology", "Primary_substrate", "Temperature_range"]:
    meta[c] = meta[c].astype("string").str.strip()
meta.to_csv(OUT / "metadata_clean.csv", index=False)

for amp, (tab, mapf, sx) in AMPS.items():
    asv = pd.read_csv(RAW / tab, sep="\t", index_col=0)
    mp = pd.read_csv(RAW / mapf, sep="\t", header=None, names=["seqID", "SampleID"])
    mp = mp[mp.seqID.isin(asv.columns)]
    asv = asv[mp.seqID].copy()
    asv.columns = mp.SampleID.values
    asv = asv.T.groupby(level=0).sum()  # samples x ASV
    tax = parse_sintax(RAW / sx).reindex(asv.columns)
    tax.to_csv(OUT / f"asv_taxonomy_{amp}.csv")
    depth = asv.sum(1)
    print(amp, "samples", asv.shape[0], "ASVs", asv.shape[1],
          "depth median", int(depth.median()), "min", int(depth.min()),
          "ASVs w/o genus", int(tax.g.str.startswith("uc_").sum()))
    asv.astype("int32").to_parquet(OUT / f"{amp}_asv_counts.parquet")
    for r in ["p", "f", "g"]:
        agg = asv.T.groupby(tax[r].values).sum().T
        agg.astype("int64").to_parquet(OUT / f"{amp}_{r}_counts.parquet")
        print("   rank", r, agg.shape)
