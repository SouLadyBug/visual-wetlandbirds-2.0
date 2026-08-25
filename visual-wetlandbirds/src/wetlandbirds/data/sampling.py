from __future__ import annotations

import pandas as pd


def stratified_sample(df: pd.DataFrame, group_col: str, n_per_group: int, max_total: int | None, seed: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    parts = [g.sample(min(len(g), n_per_group), random_state=seed) for _, g in df.groupby(group_col, sort=True)]
    out = pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0].copy()
    if max_total is not None and len(out) > max_total:
        out = out.sample(max_total, random_state=seed).reset_index(drop=True)
    return out
