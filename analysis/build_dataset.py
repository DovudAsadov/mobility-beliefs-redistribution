"""
Build analysis dataset from LiTS IV (raw .dta).

Variables (verified against LiTS IV 2022-23 codebook):
- q401b: I have done better in life than my parents (1-5) [retrospective mobility]
- q401c: HH lives better nowadays than 4 years ago     (1-5) [short-term retrospective]
- q401e: Children born now will have a better life     (1-5) [prospective mobility]
- q401f: The gap between rich and poor should be reduced (1-5) [PRIMARY DV]
- q405e: Willingness to pay more to reduce inequality   (1=Yes, 2=No) [SECONDARY DV]
- q235:  Current 10-step wealth ladder
- q236:  Ladder 4 years ago
- q237:  Expected ladder 4 years from now
- q406:  Most important factor for success
        1=effort/hard work, 2=intelligence/skills,
        3=political connections, 4=breaking the law, 5=other  [moderator base]
- q225:  Household monthly income (continuous)
- q107b: Marital status (1-5)
- q109b: Education (1-8)
- q310:  Employment status (categorical)
- q1031..: gender of household roster member 1..N
- q1051..: age of household roster member 1..N
- rand_resp_code: roster index (1..N) of the primary respondent
- urbanity: 1 urban / 2 rural
- country: country code
- weight_pop: final weight scaled to 18+ population
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyreadstat


RAW_PATH = "lits_iv.dta"
OUT_PATH = "analysis/lits_iv_clean.parquet"


KEEP = [
    "hhid", "country", "urbanity", "weight", "weight_pop",
    "rand_resp_code", "know_resp_code",
    # outcomes
    "q401f", "q405e",
    # mobility
    "q401b", "q401c", "q401e", "q235", "q236", "q237",
    # fairness moderator
    "q406",
    # demographics / controls
    "q107b", "q109b", "q108b", "q310", "q225",
]
# roster gender/age columns (members 1..20)
KEEP += [f"q103{i}" for i in range(1, 10)] + [f"q103{i}" for i in range(10, 21)]
KEEP += [f"q105{i}" for i in range(1, 10)] + [f"q105{i}" for i in range(10, 21)]


def main() -> None:
    df, meta = pyreadstat.read_dta(RAW_PATH, usecols=KEEP)
    print(f"Read {len(df):,} rows x {df.shape[1]} cols")

    miss = {-99, -98, -97}

    def clean(s: pd.Series) -> pd.Series:
        return s.where(~s.isin(miss))

    for c in df.columns:
        if c.startswith("q"):
            df[c] = clean(df[c])

    # Pull PR's gender and age from roster using rand_resp_code
    rc = df["rand_resp_code"].astype("Int64")
    gender = pd.Series(np.nan, index=df.index, dtype="float64")
    age = pd.Series(np.nan, index=df.index, dtype="float64")
    for i in range(1, 21):
        mask = (rc == i).fillna(False)
        if not mask.any():
            continue
        g_vals = pd.to_numeric(df.loc[mask, f"q103{i}"], errors="coerce").to_numpy(dtype="float64")
        a_vals = pd.to_numeric(df.loc[mask, f"q105{i}"], errors="coerce").to_numpy(dtype="float64")
        gender.loc[mask] = g_vals
        age.loc[mask] = a_vals
    # gender encoding: 1=male, 2=female -> female dummy
    df["female"] = (gender == 2).astype("float64")
    df["female"] = df["female"].where(gender.notna())
    df["age"] = age
    df["age2"] = age ** 2

    # ---- Outcomes ------------------------------------------------------
    # q401f already 1=strongly disagree to 5=strongly agree on
    # "gap rich-poor should be reduced" -> higher = stronger pro-redistribution.
    df["redist_5"] = df["q401f"]
    # standardised version
    df["redist_z"] = (df["redist_5"] - df["redist_5"].mean()) / df["redist_5"].std()

    # binary willingness to pay more for poverty/inequality (1=Yes, 2=No)
    df["wtp_redist"] = (df["q405e"] == 1).astype("float64")
    df.loc[df["q405e"].isna(), "wtp_redist"] = np.nan

    # ---- Mobility belief variables -------------------------------------
    df["mob_retro_parents"] = df["q401b"]    # done better than parents
    df["mob_retro_4y"] = df["q401c"]         # HH lives better than 4y ago
    df["mob_prospect_kids"] = df["q401e"]    # kids better life
    df["ladder_now"] = df["q235"]
    df["ladder_past"] = df["q236"]
    df["ladder_fut"] = df["q237"]
    df["ladder_diff"] = df["ladder_fut"] - df["ladder_now"]   # expected mobility (-9..+9)
    df["ladder_retro"] = df["ladder_now"] - df["ladder_past"]  # realised mobility

    # ---- Fairness belief moderator -------------------------------------
    # q406: 1=effort/hard work, 2=intelligence/skills (merit),
    #       3=political connections, 4=breaking the law (injustice),
    #       5=other
    df["effort_belief"] = (df["q406"] == 1).astype("float64")
    df.loc[df["q406"].isna(), "effort_belief"] = np.nan
    df["injustice_belief"] = df["q406"].isin([3, 4]).astype("float64")
    df.loc[df["q406"].isna(), "injustice_belief"] = np.nan
    df["merit_belief"] = df["q406"].isin([1, 2]).astype("float64")
    df.loc[df["q406"].isna(), "merit_belief"] = np.nan

    # ---- Controls ------------------------------------------------------
    df["urban"] = (df["urbanity"] == 1).astype("float64")
    df["married"] = (df["q107b"] == 2).astype("float64")
    df.loc[df["q107b"].isna(), "married"] = np.nan
    df["edu"] = df["q109b"]
    df["tertiary"] = (df["q109b"] >= 6).astype("float64")
    df.loc[df["q109b"].isna(), "tertiary"] = np.nan
    df["employed"] = df["q310"].notna().astype("float64")  # has a main job
    df["log_inc"] = np.log(df["q225"].where(df["q225"] > 0))

    # ---- Standardised analysis variables -------------------------------
    for v in [
        "mob_retro_parents", "mob_retro_4y", "mob_prospect_kids",
        "ladder_now", "ladder_diff", "ladder_retro",
    ]:
        df[v + "_z"] = (df[v] - df[v].mean()) / df[v].std()

    keep_final = [
        "hhid", "country", "weight", "weight_pop",
        "redist_5", "redist_z", "wtp_redist",
        "mob_retro_parents", "mob_retro_parents_z",
        "mob_retro_4y", "mob_retro_4y_z",
        "mob_prospect_kids", "mob_prospect_kids_z",
        "ladder_now", "ladder_now_z",
        "ladder_diff", "ladder_diff_z",
        "ladder_retro", "ladder_retro_z",
        "effort_belief", "merit_belief", "injustice_belief",
        "female", "age", "age2", "married", "edu", "tertiary",
        "urban", "employed", "log_inc",
    ]
    out = df[keep_final].copy()
    print(out.describe(include="all").T.head(40))
    try:
        out.to_parquet(OUT_PATH, index=False)
    except Exception:
        out.to_csv(OUT_PATH.replace(".parquet", ".csv"), index=False)
    print(f"Wrote {OUT_PATH} ({len(out):,} rows)")


if __name__ == "__main__":
    main()
