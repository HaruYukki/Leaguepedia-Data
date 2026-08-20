"""
RQ1 Elite-Group Stability Analysis

Research question:
How stable is the composition of the elite group from year to year in each
league, and does that stability change over 2018-2024?

Inputs
------
output_data/playoffs_appearances.csv
output_data/player_master.csv

Outputs
-------
rq1_outputs/rq1_stability_metrics.csv
rq1_outputs/rq1_elite_counts.csv
rq1_outputs/rq1_identity_qc.csv
rq1_outputs/rq1_jaccard.png
rq1_outputs/rq1_retention.png

Definitions
-----------
Annual elite set:
The union of players with at least one official playoff appearance in Spring
or Summer for the same league and calendar year.

Jaccard similarity:
|E_t intersection E_t+1| / |E_t union E_t+1|

Retention:
|E_t intersection E_t+1| / |E_t|

Identity resolution follows the conservative logic used in Step 5 v2:
exact alias -> case-insensitive normalized alias -> normalized display name.
Ambiguous normalized matches are never forced.
"""

from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt

INPUT_DIR = Path("output_data")
OUTPUT_DIR = Path("rq1_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PLAYOFFS_FILE = INPUT_DIR / "playoffs_appearances.csv"
PLAYER_MASTER_FILE = INPUT_DIR / "player_master.csv"

ANALYSIS_START = 2018
ANALYSIS_END = 2024

LEAGUE_MAP = {
    "LoL Champions Korea": "LCK",
    "Tencent LoL Pro League": "LPL",
}
TOP_LEAGUES = set(LEAGUE_MAP)


def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def norm_basic(x):
    return clean_text(x).casefold()


def norm_display(x):
    s = clean_text(x)
    s = re.sub(r"\s*\([^()]*\)\s*$", "", s)
    return s.casefold().strip()


def choose_canonical(row):
    for col in ["PlayerPage", "CurrentID", "AllName"]:
        v = clean_text(row.get(col, ""))
        if v:
            return v
    return ""


def unique_mapping(df, key_col, value_col="CanonicalPlayer"):
    tmp = df[[key_col, value_col]].dropna().copy()
    tmp = tmp[
        (tmp[key_col].astype(str) != "")
        & (tmp[value_col].astype(str) != "")
    ]
    grouped = tmp.groupby(key_col)[value_col].agg(
        lambda s: sorted(set(map(str, s)))
    )
    unique = grouped[grouped.map(len) == 1].map(lambda x: x[0])
    ambiguous = grouped[grouped.map(len) > 1]
    return unique.to_dict(), ambiguous.to_dict()


def build_identity_maps(pm):
    pm = pm.copy()

    for col in ["AllName", "PlayerPage", "CurrentID"]:
        if col not in pm.columns:
            pm[col] = pd.NA
        pm[col] = pm[col].astype("string").str.strip()

    pm["CanonicalPlayer"] = pm.apply(choose_canonical, axis=1)

    exact_alias = (
        pm.dropna(subset=["AllName"])
        .drop_duplicates("AllName")
        .set_index("AllName")["CanonicalPlayer"]
        .to_dict()
    )

    long = []
    for col in ["AllName", "CurrentID", "PlayerPage"]:
        t = pm[[col, "CanonicalPlayer"]].copy()
        t = t.rename(columns={col: "RawKey"})
        t["BasicKey"] = t["RawKey"].map(norm_basic)
        t["DisplayKey"] = t["RawKey"].map(norm_display)
        long.append(t)

    keys = pd.concat(long, ignore_index=True)

    basic_map, ambiguous_basic = unique_mapping(
        keys.rename(columns={"BasicKey": "Key"}), "Key"
    )
    display_map, ambiguous_display = unique_mapping(
        keys.rename(columns={"DisplayKey": "Key"}), "Key"
    )

    return {
        "exact_alias": exact_alias,
        "basic_map": basic_map,
        "display_map": display_map,
        "ambiguous_basic": ambiguous_basic,
        "ambiguous_display": ambiguous_display,
    }


def resolve_player(name, maps):
    raw = clean_text(name)
    if not raw:
        return pd.NA, "unresolved", ""

    if raw in maps["exact_alias"]:
        return maps["exact_alias"][raw], "exact", ""

    basic = norm_basic(raw)
    if basic in maps["basic_map"]:
        return maps["basic_map"][basic], "normalized_basic", ""
    if basic in maps["ambiguous_basic"]:
        return (
            pd.NA,
            "ambiguous",
            ";".join(maps["ambiguous_basic"][basic]),
        )

    display = norm_display(raw)
    if display in maps["display_map"]:
        return maps["display_map"][display], "normalized_display", ""
    if display in maps["ambiguous_display"]:
        return (
            pd.NA,
            "ambiguous",
            ";".join(maps["ambiguous_display"][display]),
        )

    return pd.NA, "unresolved", ""


def prepare_playoffs(playoffs, maps):
    d = playoffs.copy()

    required = {"PlayerLink", "League", "Year", "Split", "Games"}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(f"Missing required playoff columns: {sorted(missing)}")

    d["Year"] = pd.to_numeric(d["Year"], errors="coerce")
    d = d[d["Year"].notna()].copy()
    d["Year"] = d["Year"].astype(int)

    d = d[
        d["League"].isin(TOP_LEAGUES)
        & d["Year"].between(ANALYSIS_START, ANALYSIS_END)
        & d["Split"].isin(["Spring", "Summer"])
    ].copy()

    resolved = d["PlayerLink"].map(lambda x: resolve_player(x, maps))
    d["CanonicalPlayer"] = resolved.map(lambda x: x[0])
    d["ResolutionMethod"] = resolved.map(lambda x: x[1])
    d["ResolutionCandidates"] = resolved.map(lambda x: x[2])

    # Keep unresolved rows visible rather than silently dropping them.
    d["CanonicalPlayer"] = d["CanonicalPlayer"].fillna(
        d["PlayerLink"].astype(str).str.strip()
    )

    d["LeagueShort"] = d["League"].map(LEAGUE_MAP)

    return d


def build_annual_sets(d):
    sets = {}
    for (league, year), g in d.groupby(["LeagueShort", "Year"]):
        players = set(g["CanonicalPlayer"].dropna().astype(str))
        sets[(league, int(year))] = players
    return sets


def calculate_stability(sets):
    rows = []

    for league in ["LCK", "LPL"]:
        for year in range(ANALYSIS_START, ANALYSIS_END):
            current = sets.get((league, year), set())
            nxt = sets.get((league, year + 1), set())

            intersection = current & nxt
            union = current | nxt

            jaccard = (
                len(intersection) / len(union)
                if len(union) > 0 else float("nan")
            )
            retention = (
                len(intersection) / len(current)
                if len(current) > 0 else float("nan")
            )

            rows.append({
                "League": league,
                "FromYear": year,
                "ToYear": year + 1,
                "Transition": f"{year}-{year + 1}",
                "Elite_N_t": len(current),
                "Elite_N_t1": len(nxt),
                "Retained_N": len(intersection),
                "Union_N": len(union),
                "Jaccard": jaccard,
                "Retention": retention,
            })

    return pd.DataFrame(rows)


def identity_qc(d):
    review = d[
        d["ResolutionMethod"].isin(["ambiguous", "unresolved"])
    ].copy()

    # Detect suspicious canonical identities that normalize to the same key.
    ids = (
        d[["CanonicalPlayer"]]
        .drop_duplicates()
        .assign(
            BasicKey=lambda x: x["CanonicalPlayer"].map(norm_basic),
            DisplayKey=lambda x: x["CanonicalPlayer"].map(norm_display),
        )
    )

    split_rows = []
    for key_col in ["BasicKey", "DisplayKey"]:
        for key, g in ids.groupby(key_col):
            vals = sorted(set(g["CanonicalPlayer"]))
            if key and len(vals) > 1:
                split_rows.append({
                    "Issue": "suspected_identity_split",
                    "KeyType": key_col,
                    "Key": key,
                    "Details": ";".join(vals),
                })

    qc_rows = [{
        "Issue": "unresolved_or_ambiguous_appearance_rows",
        "KeyType": "",
        "Key": "",
        "Details": str(len(review)),
    }]

    qc_rows.append({
        "Issue": "unique_unresolved_or_ambiguous_playerlinks",
        "KeyType": "",
        "Key": "",
        "Details": str(review["PlayerLink"].nunique()),
    })

    qc_rows.extend(split_rows)
    return pd.DataFrame(qc_rows), review


def plot_metric(metrics, metric, ylabel, filename):
    fig, ax = plt.subplots(figsize=(8, 4.8))

    for league, g in metrics.groupby("League"):
        g = g.sort_values("FromYear")
        ax.plot(
            g["Transition"],
            g[metric],
            marker="o",
            label=league,
        )

    ax.set_xlabel("Adjacent-year transition")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    playoffs = pd.read_csv(PLAYOFFS_FILE)
    pm = pd.read_csv(PLAYER_MASTER_FILE)

    maps = build_identity_maps(pm)
    d = prepare_playoffs(playoffs, maps)

    qc, review = identity_qc(d)
    qc.to_csv(OUTPUT_DIR / "rq1_identity_qc.csv", index=False)

    if not review.empty:
        review.to_csv(
            OUTPUT_DIR / "rq1_manual_identity_review.csv",
            index=False
        )

    sets = build_annual_sets(d)
    metrics = calculate_stability(sets)

    counts = (
        d.groupby(["LeagueShort", "Year"])["CanonicalPlayer"]
        .nunique()
        .reset_index(name="Elite_N")
        .rename(columns={"LeagueShort": "League"})
        .sort_values(["League", "Year"])
    )

    metrics.to_csv(
        OUTPUT_DIR / "rq1_stability_metrics.csv",
        index=False
    )
    counts.to_csv(
        OUTPUT_DIR / "rq1_elite_counts.csv",
        index=False
    )

    plot_metric(
        metrics,
        "Jaccard",
        "Jaccard similarity",
        "rq1_jaccard.png",
    )
    plot_metric(
        metrics,
        "Retention",
        "Retention rate",
        "rq1_retention.png",
    )

    print("\nIdentity QC")
    print(qc.to_string(index=False))

    print("\nAnnual elite counts")
    print(counts.to_string(index=False))

    print("\nAdjacent-year stability")
    print(
        metrics[
            [
                "League",
                "Transition",
                "Elite_N_t",
                "Elite_N_t1",
                "Retained_N",
                "Jaccard",
                "Retention",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
