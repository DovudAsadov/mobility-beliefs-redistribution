"""
OLS regressions of redistribution preference on mobility beliefs,
with country fixed effects, country-clustered SEs, and fairness-belief
interactions.

Outputs:
- output/tab1_descriptives.csv
- output/tab2_main_ols.csv
- output/tab3_interactions.csv
- output/tab4_robust_wtp.csv
- output/fig_country_means.png
- output/fig_pred_interaction.png
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns


warnings.filterwarnings("ignore")
sns.set_theme(context="paper", style="whitegrid")

DATA = "analysis/lits_iv_clean.parquet"


# ---------------------------------------------------------------------------
def cluster_ols(y: pd.Series, X: pd.DataFrame, cluster: pd.Series) -> sm.regression.linear_model.RegressionResultsWrapper:
    """OLS with country-clustered standard errors."""
    return sm.OLS(y, X, missing="drop").fit(
        cov_type="cluster", cov_kwds={"groups": cluster.loc[y.index].values}
    )


def make_design(df: pd.DataFrame, mob, controls: list[str], country_fe: bool = True) -> pd.DataFrame:
    """Design matrix; `mob` may be a single column name or a list of names."""
    if isinstance(mob, str):
        mob_cols = [mob]
    else:
        mob_cols = list(mob)
    cols = mob_cols + controls
    X = df[cols].copy()
    if country_fe:
        fe = pd.get_dummies(df["country"].astype("Int64"), prefix="c", drop_first=True, dtype=float)
        X = pd.concat([X, fe], axis=1)
    X = sm.add_constant(X, has_constant="add")
    return X


# ---- LiTS IV region groupings (country codes per .dta value labels) ---------
REGIONS: dict[str, list[int]] = {
    "EU-CEE":          [6, 7, 8, 10, 12, 13, 14, 19, 21, 27, 28, 31, 32],
    "Western Balkans": [1, 5, 17, 24, 26, 30],
    "EaP+Caucasus":    [2, 3, 4, 11, 22, 29],
    "Central Asia":    [16, 18, 23, 33, 37],
    "SEMED+Türkiye":   [15, 20, 25, 34, 35, 38, 39],
}


def region_of(c: int) -> str | None:
    for name, members in REGIONS.items():
        if c in members:
            return name
    return None


def to_dropna(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df.dropna(subset=cols).copy()


def stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""


def coef_row(res, name: str) -> dict:
    if name not in res.params.index:
        return {}
    return {
        "coef": float(res.params[name]),
        "se": float(res.bse[name]),
        "p": float(res.pvalues[name]),
        "stars": stars(float(res.pvalues[name])),
    }


# ---------------------------------------------------------------------------
def main() -> None:
    df = pd.read_parquet(DATA)
    print(f"Loaded {len(df):,} rows")

    controls = ["female", "age", "age2", "tertiary", "married", "urban", "log_inc"]
    df = df.dropna(subset=["redist_z", "country"]).copy()

    # ---- Table 1: descriptives -------------------------------------------
    desc_vars = [
        "redist_5", "wtp_redist",
        "mob_retro_parents", "mob_retro_4y", "mob_prospect_kids",
        "ladder_now", "ladder_diff", "ladder_retro",
        "effort_belief", "merit_belief", "injustice_belief",
        "female", "age", "tertiary", "married", "urban", "log_inc",
    ]
    tab1 = df[desc_vars].describe().T[["count", "mean", "std", "min", "max"]]
    tab1.to_csv("output/tab1_descriptives.csv", float_format="%.3f")
    print("\n=== Table 1: Descriptives ===")
    print(tab1.round(3))

    # ---- Table 2: main OLS – each mobility variable separately -----------
    print("\n=== Table 2: Main OLS (DV = standardised redistribution support) ===")
    rows = []
    for mob in [
        "mob_retro_parents_z", "mob_retro_4y_z", "mob_prospect_kids_z",
        "ladder_now_z", "ladder_diff_z", "ladder_retro_z",
    ]:
        sub = to_dropna(df, ["redist_z", mob] + controls + ["country"])
        X = make_design(sub, mob, controls)
        y = sub["redist_z"]
        res = cluster_ols(y, X, sub["country"])
        rows.append({
            "mobility_var": mob,
            "n": int(res.nobs),
            "coef_mobility": res.params[mob],
            "se_mobility": res.bse[mob],
            "p_mobility": res.pvalues[mob],
            "stars": stars(res.pvalues[mob]),
            "r2": res.rsquared,
        })
        print(f"{mob:25s} n={int(res.nobs):>6}  beta={res.params[mob]:+.4f}  "
              f"SE={res.bse[mob]:.4f}  p={res.pvalues[mob]:.3g} {stars(res.pvalues[mob])}")
    tab2 = pd.DataFrame(rows)
    tab2.to_csv("output/tab2_main_ols.csv", index=False, float_format="%.4f")

    # ---- Table 3: interactions with fairness beliefs ---------------------
    print("\n=== Table 3: Interaction with fairness beliefs ===")
    rows3 = []
    interaction_results = {}
    for mob in ["mob_retro_parents_z", "mob_prospect_kids_z", "ladder_diff_z"]:
        for mod in ["effort_belief", "injustice_belief"]:
            sub = to_dropna(df, ["redist_z", mob, mod] + controls + ["country"])
            sub["interaction"] = sub[mob] * sub[mod]
            X = make_design(sub, mob, controls + [mod, "interaction"])
            y = sub["redist_z"]
            res = cluster_ols(y, X, sub["country"])
            interaction_results[(mob, mod)] = res
            row = {
                "mobility": mob, "moderator": mod, "n": int(res.nobs),
                "beta_mob": res.params[mob], "se_mob": res.bse[mob], "p_mob": res.pvalues[mob],
                "beta_mod": res.params[mod], "se_mod": res.bse[mod], "p_mod": res.pvalues[mod],
                "beta_int": res.params["interaction"], "se_int": res.bse["interaction"],
                "p_int": res.pvalues["interaction"],
                "stars_int": stars(res.pvalues["interaction"]),
                "r2": res.rsquared,
            }
            rows3.append(row)
            print(f"{mob:25s} x {mod:18s} n={int(res.nobs):>6}  "
                  f"int={row['beta_int']:+.4f}  SE={row['se_int']:.4f}  "
                  f"p={row['p_int']:.3g} {row['stars_int']}")
    tab3 = pd.DataFrame(rows3)
    tab3.to_csv("output/tab3_interactions.csv", index=False, float_format="%.4f")

    # ---- Table 4: robustness with binary WTP outcome (LPM) ---------------
    print("\n=== Table 4: LPM on willingness-to-pay outcome ===")
    rows4 = []
    sub = df.dropna(subset=["wtp_redist"] + controls + ["country"]).copy()
    for mob in [
        "mob_retro_parents_z", "mob_prospect_kids_z", "ladder_diff_z",
    ]:
        s2 = sub.dropna(subset=[mob])
        X = make_design(s2, mob, controls)
        y = s2["wtp_redist"]
        res = cluster_ols(y, X, s2["country"])
        rows4.append({
            "mobility": mob, "n": int(res.nobs),
            "beta": res.params[mob], "se": res.bse[mob], "p": res.pvalues[mob],
            "stars": stars(res.pvalues[mob]),
            "r2": res.rsquared,
        })
        print(f"{mob:25s} n={int(res.nobs):>6}  beta={res.params[mob]:+.4f}  "
              f"SE={res.bse[mob]:.4f}  p={res.pvalues[mob]:.3g} {stars(res.pvalues[mob])}")
    tab4 = pd.DataFrame(rows4)
    tab4.to_csv("output/tab4_robust_wtp.csv", index=False, float_format="%.4f")

    # ---- Table 5: population-weighted re-estimates (WLS) ----------------
    print("\n=== Table 5: Population-weighted (WLS, weight_pop) ===")
    rows5 = []
    for mob in [
        "mob_retro_parents_z", "mob_retro_4y_z", "mob_prospect_kids_z",
        "ladder_now_z", "ladder_diff_z", "ladder_retro_z",
    ]:
        sub = to_dropna(df, ["redist_z", mob, "weight_pop"] + controls + ["country"])
        X = make_design(sub, mob, controls)
        y = sub["redist_z"]
        w = sub["weight_pop"].astype(float)
        res = sm.WLS(y, X, weights=w, missing="drop").fit(
            cov_type="cluster", cov_kwds={"groups": sub["country"].values}
        )
        rows5.append({
            "mobility_var": mob, "n": int(res.nobs),
            "coef_mobility": res.params[mob],
            "se_mobility": res.bse[mob],
            "p_mobility": res.pvalues[mob],
            "stars": stars(res.pvalues[mob]),
            "r2": res.rsquared,
        })
        print(f"{mob:25s} n={int(res.nobs):>6}  beta={res.params[mob]:+.4f}  "
              f"SE={res.bse[mob]:.4f}  p={res.pvalues[mob]:.3g} {stars(res.pvalues[mob])}")
    pd.DataFrame(rows5).to_csv("output/tab5_weighted_main.csv", index=False, float_format="%.4f")

    # ---- Table 6: ordered-logit robustness on the 1-5 Likert outcome ----
    # We use the unstandardised redist_5 as the ordinal DV and report
    # marginal-effect-comparable coefficients on standardised mobility variables.
    # Country FE included as dummies; SEs are model-based (clustered SEs are
    # not natively supported by OrderedModel) — see footnote in the paper.
    from statsmodels.miscmodels.ordinal_model import OrderedModel
    print("\n=== Table 6: Ordered logit (DV = redistribution Likert 1-5) ===")
    rows6 = []
    for mob in ["mob_retro_parents_z", "mob_prospect_kids_z", "ladder_diff_z", "ladder_now_z"]:
        sub = to_dropna(df, ["redist_5", mob] + controls + ["country"])
        # build design without intercept (OrderedModel handles thresholds)
        fe = pd.get_dummies(sub["country"].astype("Int64"), prefix="c", drop_first=True, dtype=float)
        Xo = pd.concat([sub[[mob] + controls].astype(float), fe], axis=1)
        try:
            mod = OrderedModel(sub["redist_5"].astype(int), Xo, distr="logit")
            ores = mod.fit(method="bfgs", disp=False, maxiter=200)
            rows6.append({
                "mobility_var": mob, "n": int(ores.nobs),
                "coef_mobility": float(ores.params[mob]),
                "se_mobility": float(ores.bse[mob]),
                "p_mobility": float(ores.pvalues[mob]),
                "stars": stars(float(ores.pvalues[mob])),
            })
            print(f"{mob:25s} n={int(ores.nobs):>6}  beta={ores.params[mob]:+.4f}  "
                  f"SE={ores.bse[mob]:.4f}  p={ores.pvalues[mob]:.3g} {stars(float(ores.pvalues[mob]))}")
        except Exception as e:
            print(f"{mob}: ordered-logit failed ({e})")
            rows6.append({"mobility_var": mob, "n": np.nan, "coef_mobility": np.nan,
                          "se_mobility": np.nan, "p_mobility": np.nan, "stars": ""})
    pd.DataFrame(rows6).to_csv("output/tab6_ologit.csv", index=False, float_format="%.4f")

    # ---- Table 7: joint horse-race regression ---------------------------
    # All four mobility variables on the right-hand side at once: which
    # channel survives controlling for the others?
    print("\n=== Table 7: Joint horse-race regression ===")
    mobs_joint = ["mob_retro_parents_z", "mob_prospect_kids_z", "ladder_now_z", "ladder_diff_z"]
    sub = to_dropna(df, ["redist_z"] + mobs_joint + controls + ["country"])
    X = make_design(sub, mobs_joint, controls)
    y = sub["redist_z"]
    res = cluster_ols(y, X, sub["country"])
    rows7 = []
    for m in mobs_joint:
        rows7.append({
            "mobility_var": m,
            "coef": float(res.params[m]),
            "se": float(res.bse[m]),
            "p": float(res.pvalues[m]),
            "stars": stars(float(res.pvalues[m])),
        })
        print(f"  {m:25s} beta={res.params[m]:+.4f}  SE={res.bse[m]:.4f}  "
              f"p={res.pvalues[m]:.3g} {stars(float(res.pvalues[m]))}")
    pd.DataFrame(rows7).assign(n=int(res.nobs), r2=res.rsquared).to_csv(
        "output/tab7_joint.csv", index=False, float_format="%.4f"
    )

    # ---- Table 8: heterogeneity by current ladder position --------------
    # Bottom = ladder 1-3, Middle = 4-6, Top = 7-10. POUM predicts a
    # stronger negative effect for those with more "room to rise" (bottom).
    print("\n=== Table 8: Heterogeneity by current ladder position ===")
    rows8 = []
    bins = [("Bottom (1-3)", lambda x: x <= 3),
            ("Middle (4-6)", lambda x: (x >= 4) & (x <= 6)),
            ("Top (7-10)",   lambda x: x >= 7)]
    for label, mask_fn in bins:
        sub = to_dropna(df, ["redist_z", "mob_prospect_kids_z", "ladder_now"] + controls + ["country"])
        sub = sub[mask_fn(sub["ladder_now"])].copy()
        if len(sub) < 200:
            continue
        X = make_design(sub, "mob_prospect_kids_z", controls)
        y = sub["redist_z"]
        res = cluster_ols(y, X, sub["country"])
        rows8.append({
            "ladder_group": label, "n": int(res.nobs),
            "coef": float(res.params["mob_prospect_kids_z"]),
            "se": float(res.bse["mob_prospect_kids_z"]),
            "p": float(res.pvalues["mob_prospect_kids_z"]),
            "stars": stars(float(res.pvalues["mob_prospect_kids_z"])),
        })
        print(f"  {label:14s} n={int(res.nobs):>6}  beta={res.params['mob_prospect_kids_z']:+.4f}  "
              f"SE={res.bse['mob_prospect_kids_z']:.4f}  "
              f"p={res.pvalues['mob_prospect_kids_z']:.3g} {stars(float(res.pvalues['mob_prospect_kids_z']))}")
    pd.DataFrame(rows8).to_csv("output/tab8_ladder_split.csv", index=False, float_format="%.4f")

    # ---- Table 9: age-cohort heterogeneity ------------------------------
    # Approximate Soviet-socialised vs post-Soviet cohort using age in 2023.
    # LiTS IV fielded 2022-23; respondents born <1980 are >= 43 in 2023.
    print("\n=== Table 9: Heterogeneity by age cohort (born <1980 vs >=1980) ===")
    rows9 = []
    cohorts = [("Born <1980 (older)", lambda a: a >= 43),
               ("Born >=1980 (younger)", lambda a: a < 43)]
    for label, mask_fn in cohorts:
        for mob in ["mob_prospect_kids_z", "ladder_diff_z"]:
            sub = to_dropna(df, ["redist_z", mob, "age"] + controls + ["country"])
            sub = sub[mask_fn(sub["age"])].copy()
            X = make_design(sub, mob, controls)
            y = sub["redist_z"]
            res = cluster_ols(y, X, sub["country"])
            rows9.append({
                "cohort": label, "mobility_var": mob, "n": int(res.nobs),
                "coef": float(res.params[mob]), "se": float(res.bse[mob]),
                "p": float(res.pvalues[mob]),
                "stars": stars(float(res.pvalues[mob])),
            })
            print(f"  {label:24s} {mob:22s} n={int(res.nobs):>6}  "
                  f"beta={res.params[mob]:+.4f}  p={res.pvalues[mob]:.3g} "
                  f"{stars(float(res.pvalues[mob]))}")
    pd.DataFrame(rows9).to_csv("output/tab9_cohort.csv", index=False, float_format="%.4f")

    # ---- Table 10 + Figure 3: region splits -----------------------------
    print("\n=== Table 10: Region splits ===")
    df["region"] = df["country"].apply(region_of)
    rows10 = []
    for region in REGIONS.keys():
        sub = to_dropna(df[df["region"] == region], ["redist_z", "mob_prospect_kids_z"] + controls + ["country"])
        if len(sub) < 200 or sub["country"].nunique() < 2:
            continue
        X = make_design(sub, "mob_prospect_kids_z", controls)
        y = sub["redist_z"]
        res = cluster_ols(y, X, sub["country"])
        rows10.append({
            "region": region, "n_obs": int(res.nobs),
            "n_countries": int(sub["country"].nunique()),
            "coef": float(res.params["mob_prospect_kids_z"]),
            "se": float(res.bse["mob_prospect_kids_z"]),
            "p": float(res.pvalues["mob_prospect_kids_z"]),
            "stars": stars(float(res.pvalues["mob_prospect_kids_z"])),
        })
        print(f"  {region:18s} n={int(res.nobs):>6} ({sub['country'].nunique()} countries)  "
              f"beta={res.params['mob_prospect_kids_z']:+.4f}  "
              f"p={res.pvalues['mob_prospect_kids_z']:.3g} "
              f"{stars(float(res.pvalues['mob_prospect_kids_z']))}")
    tab10 = pd.DataFrame(rows10)
    tab10.to_csv("output/tab10_regions.csv", index=False, float_format="%.4f")

    # Forest plot of region coefficients
    if len(tab10):
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        order = tab10.sort_values("coef")
        y_pos = np.arange(len(order))
        ax.errorbar(order["coef"], y_pos, xerr=1.96 * order["se"],
                    fmt="o", color="tab:blue", capsize=3)
        ax.axvline(0, lw=0.6, color="grey", linestyle="--")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(order["region"])
        ax.set_xlabel("Coefficient on standardised prospective-mobility belief\n(95% CI, country-clustered SE)")
        ax.set_title("Region heterogeneity: mobility belief → redistribution support")
        plt.tight_layout()
        plt.savefig("output/fig_region_coefs.png", dpi=160)
        plt.close(fig)

    # ---- Figure 1: country means redistribution vs mobility belief --------
    cmap_data = (
        df.groupby("country", observed=True)
          .agg(redist=("redist_5", "mean"),
               mob=("mob_prospect_kids", "mean"),
               n=("redist_5", "size"))
          .reset_index()
    )
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.scatter(cmap_data["mob"], cmap_data["redist"], s=24, alpha=0.85)
    for _, r in cmap_data.iterrows():
        ax.annotate(int(r["country"]), (r["mob"], r["redist"]),
                    fontsize=7, alpha=0.7, xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel("Country mean prospective mobility belief\n(higher = more optimistic about kids' future)")
    ax.set_ylabel("Country mean support for redistribution\n('gap should be reduced', 1–5)")
    ax.set_title("Cross-country pattern of mobility optimism and redistribution support")
    plt.tight_layout()
    plt.savefig("output/fig_country_means.png", dpi=160)
    plt.close(fig)

    # ---- Figure 2: predicted redistribution by mobility, split by belief --
    sub = df.dropna(subset=["redist_z", "mob_prospect_kids_z", "effort_belief"] + controls).copy()
    sub["interaction"] = sub["mob_prospect_kids_z"] * sub["effort_belief"]
    X = make_design(sub, "mob_prospect_kids_z", controls + ["effort_belief", "interaction"])
    res = cluster_ols(sub["redist_z"], X, sub["country"])
    grid = np.linspace(-2, 2, 41)
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    for belief, lab, c in [(1, "Believes effort drives success", "tab:red"),
                          (0, "Doesn't believe effort drives success", "tab:blue")]:
        slope = res.params["mob_prospect_kids_z"] + belief * res.params["interaction"]
        intercept = (res.params["const"]
                     + belief * res.params["effort_belief"]
                     + sub[controls].mean().to_numpy() @ res.params[controls].to_numpy()
                     + 0)  # country FE collapse to mean ~ 0 in standardised outcome
        ax.plot(grid, intercept + slope * grid, label=f"{lab} (slope={slope:+.3f})", color=c)
    ax.axhline(0, lw=0.5, color="grey")
    ax.set_xlabel("Prospective mobility belief (z)")
    ax.set_ylabel("Predicted standardised redistribution support")
    ax.set_title("Mobility belief × fairness belief interaction\n(LiTS IV, 37 economies)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("output/fig_pred_interaction.png", dpi=160)
    plt.close(fig)

    print("\nAll tables/figures written to output/")


if __name__ == "__main__":
    main()
