"""
Appendix Reproducibility Script
===============================

Purpose
-------
This script reproduces the descriptive analysis used in RQ1 and RQ2 of the
LCK-LPL elite-player analysis.

It covers the complete workflow from canonical player identity resolution to
the final tables and figures currently used in the Findings section:

RQ1
---
1. Construct annual elite-player sets from Spring and Summer playoff appearances.
2. Calculate adjacent-year Jaccard similarity.
3. Calculate adjacent-year retention.
4. Export annual elite-group counts and stability metrics.
5. Plot Jaccard and retention trends for the LCK and LPL.

RQ2
---
6. Merge verified date-of-birth information onto annual elite-player sets.
7. Calculate annual mean and median elite age.
8. Calculate annual mean and median TrueDebut entry age.
9. Plot mean elite age and mean TrueDebut age.
10. Count annual elite players and TrueDebuts.
11. Calculate TrueDebut share = TrueDebut N / Elite N.
12. Produce a two-panel figure for elite-group size and TrueDebut counts.

Inputs
------
output_data/playoffs_appearances.csv
output_data/player_master.csv
step5_outputs/first_elite_entry_events.csv

Outputs
-------
appendix_analysis_outputs/
    q1_elite_counts.csv
    q1_stability_metrics.csv
    q1_identity_qc.csv
    q1_stability_trends.png

    q2_elite_age_summary.csv
    q2_true_debut_age_summary.csv
    q2_age_and_count_table.csv
    q2_identity_qc.csv
    q2_dob_conflicts.csv
    q2_mean_age_trends.png
    q2_elite_count_trend.png
    q2_true_debut_count_trend.png
    q2_elite_true_debut_counts_two_panel.png

Notes
-----
- The annual elite group is the union of Spring and Summer playoff participants.
- A player appears only once per league-year.
- Age is measured as of 31 December of each calendar year:
      Age = calendar year - birth year
- Missing DOB values are not imputed.
- TrueDebut classifications are taken from the final Step 5 entry-event file.
- The script is descriptive and does not estimate causal effects.
"""

from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageDraw, ImageFont


# ---------------------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------------------

INPUT_DIR = Path("output_data")
STEP5_DIR = Path("step5_outputs")
OUTPUT_DIR = Path("appendix_analysis_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PLAYOFFS_FILE = INPUT_DIR / "playoffs_appearances.csv"
PLAYER_MASTER_FILE = INPUT_DIR / "player_master.csv"
ENTRY_FILE = STEP5_DIR / "first_elite_entry_events.csv"

ANALYSIS_START = 2018
ANALYSIS_END = 2024

LEAGUE_MAP = {
    "LoL Champions Korea": "LCK",
    "Tencent LoL Pro League": "LPL",
}
TOP_LEAGUES = set(LEAGUE_MAP.keys())


# ---------------------------------------------------------------------
# 1. Identity-resolution helpers
# ---------------------------------------------------------------------

def clean_text(x):
    """Return a stripped string, treating missing values as an empty string."""
    if pd.isna(x):
        return ""
    return str(x).strip()


def norm_basic(x):
    """Case-insensitive normalisation used for conservative identifier matching."""
    return clean_text(x).casefold()


def norm_display(x):
    """
    Display-name normalisation.

    A trailing parenthetical real-name suffix is removed, e.g.
    'Tarzan (Lee Seung-yong)' -> 'tarzan'.
    """
    s = clean_text(x)
    s = re.sub(r"\s*\([^()]*\)\s*$", "", s)
    return s.casefold().strip()


def choose_canonical(row):
    """Select preferred canonical identifier: PlayerPage -> CurrentID -> AllName."""
    for col in ["PlayerPage", "CurrentID", "AllName"]:
        value = clean_text(row.get(col, ""))
        if value:
            return value
    return ""


def unique_mapping(df, key_col, value_col="CanonicalPlayer"):
    """
    Build a mapping only when one normalised key maps to exactly one canonical
    player. Ambiguous keys are retained separately rather than forced.
    """
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


def build_player_master(player_master):
    """Build identity maps and one canonical DOB record per player."""
    pm = player_master.copy()

    for col in ["AllName", "PlayerPage", "CurrentID", "Birthdate"]:
        if col not in pm.columns:
            pm[col] = pd.NA

    for col in ["AllName", "PlayerPage", "CurrentID"]:
        pm[col] = pm[col].astype("string").str.strip()

    pm["CanonicalPlayer"] = pm.apply(choose_canonical, axis=1)

    pm["BirthdateParsed"] = pd.to_datetime(pm["Birthdate"], errors="coerce")
    pm["BirthYear"] = pm["BirthdateParsed"].dt.year

    invalid = ~pm["BirthYear"].between(1980, 2012, inclusive="both")
    pm.loc[invalid, ["BirthdateParsed", "BirthYear"]] = pd.NA

    exact_alias = (
        pm.dropna(subset=["AllName"])
        .drop_duplicates("AllName")
        .set_index("AllName")["CanonicalPlayer"]
        .to_dict()
    )

    key_frames = []
    for col in ["AllName", "CurrentID", "PlayerPage"]:
        t = pm[[col, "CanonicalPlayer"]].copy()
        t = t.rename(columns={col: "RawKey"})
        t["BasicKey"] = t["RawKey"].map(norm_basic)
        t["DisplayKey"] = t["RawKey"].map(norm_display)
        key_frames.append(t)

    keys = pd.concat(key_frames, ignore_index=True)

    basic_map, ambiguous_basic = unique_mapping(
        keys.rename(columns={"BasicKey": "Key"}), "Key"
    )
    display_map, ambiguous_display = unique_mapping(
        keys.rename(columns={"DisplayKey": "Key"}), "Key"
    )

    maps = {
        "exact_alias": exact_alias,
        "basic_map": basic_map,
        "display_map": display_map,
        "ambiguous_basic": ambiguous_basic,
        "ambiguous_display": ambiguous_display,
    }

    dob_groups = (
        pm.dropna(subset=["CanonicalPlayer"])
        .groupby("CanonicalPlayer")["BirthdateParsed"]
        .agg(lambda s: sorted(set(x for x in s.dropna())))
    )

    dob_rows = []
    conflict_rows = []

    for player, dates in dob_groups.items():
        if len(dates) > 1:
            conflict_rows.append({
                "CanonicalPlayer": player,
                "Birthdates": ";".join(
                    str(pd.Timestamp(x).date()) for x in dates
                ),
                "N_Distinct_Birthdates": len(dates),
            })

        chosen = dates[0] if len(dates) == 1 else pd.NaT

        dob_rows.append({
            "CanonicalPlayer": player,
            "Birthdate": chosen,
            "BirthYear": chosen.year if pd.notna(chosen) else pd.NA,
        })

    canonical_dob = pd.DataFrame(dob_rows)

    dob_conflicts = pd.DataFrame(
        conflict_rows,
        columns=[
            "CanonicalPlayer",
            "Birthdates",
            "N_Distinct_Birthdates",
        ],
    )

    return maps, canonical_dob, dob_conflicts


def resolve_player(name, maps):
    """
    Resolve one appearance identifier using:
    1. exact alias
    2. case-insensitive normalisation
    3. display-name normalisation
    """
    raw = clean_text(name)

    if not raw:
        return pd.NA, "unresolved", ""

    if raw in maps["exact_alias"]:
        return maps["exact_alias"][raw], "exact", ""

    basic = norm_basic(raw)

    if basic in maps["basic_map"]:
        return maps["basic_map"][basic], "normalized_basic", ""

    if basic in maps["ambiguous_basic"]:
        return pd.NA, "ambiguous", ";".join(maps["ambiguous_basic"][basic])

    display = norm_display(raw)

    if display in maps["display_map"]:
        return maps["display_map"][display], "normalized_display", ""

    if display in maps["ambiguous_display"]:
        return pd.NA, "ambiguous", ";".join(maps["ambiguous_display"][display])

    return pd.NA, "unresolved", ""


# ---------------------------------------------------------------------
# 2. Construct annual elite-player sets
# ---------------------------------------------------------------------

def prepare_elite_sample(playoffs, maps, canonical_dob):
    """
    Canonicalise playoff appearances and construct one player-year record per
    league. Annual elite = union of Spring and Summer playoff participants.
    """
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

    # Preserve unresolved rows for QC rather than silently discarding them.
    d["CanonicalPlayer"] = d["CanonicalPlayer"].fillna(
        d["PlayerLink"].astype(str).str.strip()
    )

    d["LeagueShort"] = d["League"].map(LEAGUE_MAP)

    annual = (
        d[["LeagueShort", "Year", "CanonicalPlayer"]]
        .drop_duplicates()
        .merge(canonical_dob, on="CanonicalPlayer", how="left")
    )

    annual["KnownDOB"] = annual["BirthYear"].notna()
    annual["Age"] = (
        annual["Year"]
        - pd.to_numeric(annual["BirthYear"], errors="coerce")
    )

    return d, annual


def export_identity_qc(raw_appearances, prefix):
    """Export unresolved/ambiguous identity checks."""
    review = raw_appearances[
        raw_appearances["ResolutionMethod"].isin(["ambiguous", "unresolved"])
    ].copy()

    qc = pd.DataFrame([
        {
            "Check": "Unresolved or ambiguous playoff appearance rows",
            "Value": len(review),
            "Expected": 0,
        },
        {
            "Check": "Unique unresolved or ambiguous PlayerLinks",
            "Value": review["PlayerLink"].nunique(),
            "Expected": 0,
        },
    ])

    qc.to_csv(OUTPUT_DIR / f"{prefix}_identity_qc.csv", index=False)

    if not review.empty:
        review.to_csv(
            OUTPUT_DIR / f"{prefix}_manual_identity_review.csv",
            index=False,
        )

    return qc


# ---------------------------------------------------------------------
# 3. RQ1: Jaccard similarity and retention
# ---------------------------------------------------------------------

def calculate_q1(annual_elite):
    """
    Jaccard = |E_t intersection E_t+1| / |E_t union E_t+1|
    Retention = |E_t intersection E_t+1| / |E_t|
    """
    elite_sets = {}

    for (league, year), g in annual_elite.groupby(["LeagueShort", "Year"]):
        elite_sets[(league, int(year))] = set(
            g["CanonicalPlayer"].dropna().astype(str)
        )

    count_rows = []
    for (league, year), players in sorted(elite_sets.items()):
        count_rows.append({
            "League": league,
            "Year": year,
            "Elite_N": len(players),
        })

    elite_counts = pd.DataFrame(count_rows)

    metric_rows = []

    for league in ["LCK", "LPL"]:
        for year in range(ANALYSIS_START, ANALYSIS_END):

            current = elite_sets.get((league, year), set())
            next_year = elite_sets.get((league, year + 1), set())

            retained = current & next_year
            union = current | next_year

            metric_rows.append({
                "League": league,
                "FromYear": year,
                "ToYear": year + 1,
                "Transition": f"{year}-{year + 1}",
                "Elite_N_t": len(current),
                "Elite_N_t1": len(next_year),
                "Retained_N": len(retained),
                "Union_N": len(union),
                "Jaccard": (
                    len(retained) / len(union)
                    if union else float("nan")
                ),
                "Retention": (
                    len(retained) / len(current)
                    if current else float("nan")
                ),
            })

    metrics = pd.DataFrame(metric_rows)

    elite_counts.to_csv(
        OUTPUT_DIR / "q1_elite_counts.csv",
        index=False,
    )
    metrics.to_csv(
        OUTPUT_DIR / "q1_stability_metrics.csv",
        index=False,
    )

    return elite_counts, metrics


def plot_q1_stability(metrics):
    """Plot LCK/LPL Jaccard and retention in one figure."""
    fig, ax = plt.subplots(figsize=(9.5, 5.8))

    specs = [
        ("LCK", "Jaccard", "o", "-", "LCK - Jaccard"),
        ("LCK", "Retention", "s", "--", "LCK - Retention"),
        ("LPL", "Jaccard", "^", "-", "LPL - Jaccard"),
        ("LPL", "Retention", "D", "--", "LPL - Retention"),
    ]

    for league, metric, marker, linestyle, label in specs:
        g = metrics[metrics["League"] == league].sort_values("FromYear")
        ax.plot(
            g["Transition"],
            g[metric],
            marker=marker,
            linestyle=linestyle,
            linewidth=1.8,
            label=label,
        )

    ax.set_title(
        "Adjacent-Year Elite-Group Stability in the LCK and LPL, 2018-2024"
    )
    ax.set_xlabel("Adjacent-year transition")
    ax.set_ylabel("Stability measure")
    ax.set_ylim(0, 0.8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, frameon=False)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "q1_stability_trends.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


# ---------------------------------------------------------------------
# 4. RQ2: Elite-group age
# ---------------------------------------------------------------------

def summarize_elite_age(annual_elite):
    """Calculate annual elite mean/median age and DOB coverage."""
    rows = []

    for (league, year), g in annual_elite.groupby(["LeagueShort", "Year"]):
        known = g[g["KnownDOB"]].copy()

        elite_n = g["CanonicalPlayer"].nunique()
        known_n = known["CanonicalPlayer"].nunique()

        rows.append({
            "League": league,
            "Year": int(year),
            "Elite_N": elite_n,
            "KnownDOB_N": known_n,
            "DOB_Coverage": known_n / elite_n if elite_n else float("nan"),
            "Mean_Elite_Age": known["Age"].mean(),
            "Median_Elite_Age": known["Age"].median(),
            "Min_Elite_Age": known["Age"].min(),
            "Max_Elite_Age": known["Age"].max(),
        })

    summary = (
        pd.DataFrame(rows)
        .sort_values(["League", "Year"])
        .reset_index(drop=True)
    )

    summary.to_csv(
        OUTPUT_DIR / "q2_elite_age_summary.csv",
        index=False,
    )

    return summary


# ---------------------------------------------------------------------
# 5. RQ2: TrueDebut entry age
# ---------------------------------------------------------------------

def prepare_true_debuts(entries):
    """Restrict final entry-event data to TrueDebuts in 2018-2024."""
    e = entries.copy()

    e["Year"] = pd.to_numeric(e["Year"], errors="coerce")

    e = e[
        e["League"].isin(TOP_LEAGUES)
        & e["Year"].between(ANALYSIS_START, ANALYSIS_END)
        & (e["EntryType"] == "TrueDebut")
    ].copy()

    e["Year"] = e["Year"].astype(int)
    e["LeagueShort"] = e["League"].map(LEAGUE_MAP)
    e["EntryAge"] = pd.to_numeric(e["EntryAge"], errors="coerce")
    e["KnownDOB"] = e["EntryAge"].notna()

    return e


def summarize_true_debut_age(true_debuts):
    """Calculate annual TrueDebut mean/median entry age and DOB coverage."""
    rows = []

    for (league, year), g in true_debuts.groupby(["LeagueShort", "Year"]):
        known = g[g["KnownDOB"]].copy()

        rows.append({
            "League": league,
            "Year": int(year),
            "TrueDebut_N": len(g),
            "KnownDOB_N": len(known),
            "DOB_Coverage": len(known) / len(g) if len(g) else float("nan"),
            "Mean_TrueDebut_Age": known["EntryAge"].mean(),
            "Median_TrueDebut_Age": known["EntryAge"].median(),
            "Min_TrueDebut_Age": known["EntryAge"].min(),
            "Max_TrueDebut_Age": known["EntryAge"].max(),
        })

    summary = (
        pd.DataFrame(rows)
        .sort_values(["League", "Year"])
        .reset_index(drop=True)
    )

    summary.to_csv(
        OUTPUT_DIR / "q2_true_debut_age_summary.csv",
        index=False,
    )

    return summary


# ---------------------------------------------------------------------
# 6. Q2 mean-age figure
# ---------------------------------------------------------------------

def plot_mean_age(elite_age, debut_age):
    """
    Plot only mean age in the final figure.
    Median age remains available in the exported tables.
    """
    fig, ax = plt.subplots(figsize=(9.5, 5.8))

    specs = [
        ("LCK", elite_age, "Mean_Elite_Age", "o", "-", "LCK Elite Mean"),
        ("LCK", debut_age, "Mean_TrueDebut_Age", "s", "--", "LCK TrueDebut Mean"),
        ("LPL", elite_age, "Mean_Elite_Age", "^", "-", "LPL Elite Mean"),
        ("LPL", debut_age, "Mean_TrueDebut_Age", "D", "--", "LPL TrueDebut Mean"),
    ]

    for league, source, variable, marker, linestyle, label in specs:
        g = source[source["League"] == league].sort_values("Year")

        ax.plot(
            g["Year"],
            g[variable],
            marker=marker,
            linestyle=linestyle,
            linewidth=1.8,
            label=label,
        )

    ax.set_title("Mean Age of Elite Players and True Debuts, 2018-2024")
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean age")
    ax.set_xticks(list(range(ANALYSIS_START, ANALYSIS_END + 1)))
    ax.set_ylim(17.5, 23.5)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, frameon=False)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "q2_mean_age_trends.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


# ---------------------------------------------------------------------
# 7. Counts and TrueDebut share
# ---------------------------------------------------------------------

def build_age_and_count_table(elite_age, debut_age):
    """
    Combine annual RQ2 statistics and calculate:

        TrueDebut Share = TrueDebut N / Elite N
    """
    elite_part = elite_age[
        [
            "League",
            "Year",
            "Elite_N",
            "Mean_Elite_Age",
            "Median_Elite_Age",
        ]
    ].copy()

    debut_part = debut_age[
        [
            "League",
            "Year",
            "TrueDebut_N",
            "Mean_TrueDebut_Age",
            "Median_TrueDebut_Age",
        ]
    ].copy()

    combined = elite_part.merge(
        debut_part,
        on=["League", "Year"],
        how="left",
    )

    combined["TrueDebut_Share"] = (
        combined["TrueDebut_N"] / combined["Elite_N"]
    )

    combined["TrueDebut_Share_Percent"] = (
        combined["TrueDebut_Share"] * 100
    )

    combined = combined.sort_values(
        ["League", "Year"]
    ).reset_index(drop=True)

    combined.to_csv(
        OUTPUT_DIR / "q2_age_and_count_table.csv",
        index=False,
    )

    return combined


# ---------------------------------------------------------------------
# 8. Count figures
# ---------------------------------------------------------------------

def plot_elite_counts(combined):
    """Create Panel A as a standalone chart."""
    fig, ax = plt.subplots(figsize=(8.8, 5.4))

    for league, marker in [("LCK", "o"), ("LPL", "s")]:
        g = combined[combined["League"] == league].sort_values("Year")

        ax.plot(
            g["Year"],
            g["Elite_N"],
            marker=marker,
            linewidth=1.8,
            label=league,
        )

    ax.set_title("Panel A. Elite-group size")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of players")
    ax.set_xticks(list(range(ANALYSIS_START, ANALYSIS_END + 1)))
    ax.set_ylim(25, 85)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

    fig.tight_layout()

    path = OUTPUT_DIR / "q2_elite_count_trend.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return path


def plot_true_debut_counts(combined):
    """Create Panel B as a standalone chart."""
    fig, ax = plt.subplots(figsize=(8.8, 5.4))

    for league, marker in [("LCK", "o"), ("LPL", "s")]:
        g = combined[combined["League"] == league].sort_values("Year")

        ax.plot(
            g["Year"],
            g["TrueDebut_N"],
            marker=marker,
            linewidth=1.8,
            label=league,
        )

    ax.set_title("Panel B. True-debut count")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of players")
    ax.set_xticks(list(range(ANALYSIS_START, ANALYSIS_END + 1)))
    ax.set_ylim(0, 18)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

    fig.tight_layout()

    path = OUTPUT_DIR / "q2_true_debut_count_trend.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return path


def combine_count_panels(panel_a_path, panel_b_path):
    """
    Combine the two separately rendered count charts into one two-panel figure.

    Separate y-axis ranges are intentionally retained because Elite N and
    TrueDebut N differ substantially in scale.
    """
    img_a = Image.open(panel_a_path).convert("RGB")
    img_b = Image.open(panel_b_path).convert("RGB")

    target_h = max(img_a.height, img_b.height)

    def pad_to_height(img, height):
        if img.height == height:
            return img

        top = (height - img.height) // 2
        bottom = height - img.height - top

        return ImageOps.expand(
            img,
            border=(0, top, 0, bottom),
            fill="white",
        )

    img_a = pad_to_height(img_a, target_h)
    img_b = pad_to_height(img_b, target_h)

    gap = 30
    title_h = 110

    canvas = Image.new(
        "RGB",
        (img_a.width + gap + img_b.width, target_h + title_h),
        "white",
    )

    canvas.paste(img_a, (0, title_h))
    canvas.paste(img_b, (img_a.width + gap, title_h))

    draw = ImageDraw.Draw(canvas)

    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]

    font_path = next(
        (p for p in font_candidates if Path(p).exists()),
        None,
    )

    font = (
        ImageFont.truetype(font_path, 38)
        if font_path
        else ImageFont.load_default()
    )

    title = "Annual Elite-Group Size and True-Debut Counts, 2018-2024"

    bbox = draw.textbbox((0, 0), title, font=font)
    text_w = bbox[2] - bbox[0]

    draw.text(
        ((canvas.width - text_w) / 2, 28),
        title,
        fill="black",
        font=font,
    )

    out_path = (
        OUTPUT_DIR
        / "q2_elite_true_debut_counts_two_panel.png"
    )

    canvas.save(out_path, dpi=(300, 300))

    return out_path


# ---------------------------------------------------------------------
# 9. Main pipeline
# ---------------------------------------------------------------------

def main():
    print("Loading input files...")

    playoffs = pd.read_csv(PLAYOFFS_FILE)
    player_master = pd.read_csv(PLAYER_MASTER_FILE)
    entries = pd.read_csv(ENTRY_FILE)

    print("Building canonical identity maps and DOB table...")

    maps, canonical_dob, dob_conflicts = build_player_master(
        player_master
    )

    dob_conflicts.to_csv(
        OUTPUT_DIR / "q2_dob_conflicts.csv",
        index=False,
    )

    print("Constructing annual elite-player sets...")

    raw_elite, annual_elite = prepare_elite_sample(
        playoffs,
        maps,
        canonical_dob,
    )

    export_identity_qc(raw_elite, "q1")
    export_identity_qc(raw_elite, "q2")

    print("Calculating RQ1 Jaccard and retention...")

    elite_counts, q1_metrics = calculate_q1(
        annual_elite
    )

    plot_q1_stability(q1_metrics)

    print("Calculating RQ2 elite age...")

    elite_age = summarize_elite_age(
        annual_elite
    )

    print("Calculating RQ2 TrueDebut entry age...")

    true_debuts = prepare_true_debuts(
        entries
    )

    debut_age = summarize_true_debut_age(
        true_debuts
    )

    plot_mean_age(
        elite_age,
        debut_age,
    )

    print("Calculating counts and TrueDebut share...")

    combined = build_age_and_count_table(
        elite_age,
        debut_age,
    )

    elite_panel = plot_elite_counts(
        combined
    )

    debut_panel = plot_true_debut_counts(
        combined
    )

    combine_count_panels(
        elite_panel,
        debut_panel,
    )

    print("\nAnalysis complete.")
    print(f"Outputs saved to: {OUTPUT_DIR.resolve()}")

    print("\nRQ1 stability metrics:")
    print(q1_metrics.to_string(index=False))

    print("\nRQ2 age/count summary:")

    display_cols = [
        "League",
        "Year",
        "Elite_N",
        "TrueDebut_N",
        "TrueDebut_Share_Percent",
        "Mean_Elite_Age",
        "Median_Elite_Age",
        "Mean_TrueDebut_Age",
        "Median_TrueDebut_Age",
    ]

    print(
        combined[display_cols].to_string(index=False)
    )


if __name__ == "__main__":
    main()
# =====================================================================
# 10. RQ3: Under-22 entry among TrueDebuts
# =====================================================================

import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np


def prepare_q3_true_debuts(entries):
    """
    Prepare the player-level sample for RQ3.

    RQ3:
    What share of true debuts enters the elite group before age 22,
    and does that share differ between the two leagues or shift over time?

    The dependent variable is binary:
        U22 = 1 if EntryAge < 22
        U22 = 0 if EntryAge >= 22

    Time is centred at 2018:
        Time = Year - 2018

    LCK is the reference league.
    """
    d = entries.copy()

    d["Year"] = pd.to_numeric(
        d["Year"],
        errors="coerce"
    )

    d = d[
        d["League"].isin(TOP_LEAGUES)
        & d["Year"].between(ANALYSIS_START, ANALYSIS_END)
        & (d["EntryType"] == "TrueDebut")
    ].copy()

    d["Year"] = d["Year"].astype(int)
    d["LeagueShort"] = d["League"].map(LEAGUE_MAP)

    d["EntryAge"] = pd.to_numeric(
        d["EntryAge"],
        errors="coerce"
    )

    d["KnownDOB"] = d["EntryAge"].notna()

    # Binary Under-22 indicator.
    d["U22"] = np.where(
        d["KnownDOB"],
        (d["EntryAge"] < 22).astype(int),
        np.nan,
    )

    # Regression variables.
    d["Time"] = d["Year"] - ANALYSIS_START
    d["LPL"] = (d["LeagueShort"] == "LPL").astype(int)

    return d


# ---------------------------------------------------------------------
# 11. Annual descriptive U22 statistics
# ---------------------------------------------------------------------

def summarize_q3_u22(q3):
    """
    Calculate annual Under-22 shares among all TrueDebuts.

    Missing-DOB protocol:
    - Lower bound assumes every missing-DOB TrueDebut is not U22.
    - Upper bound assumes every missing-DOB TrueDebut is U22.

    In the final 2018-2024 sample all TrueDebuts have verified DOB,
    so the lower and upper bounds are identical.
    """
    rows = []

    for (league, year), g in q3.groupby(
        ["LeagueShort", "Year"]
    ):
        true_debut_n = len(g)

        known = g[g["KnownDOB"]].copy()
        known_n = len(known)

        missing_n = true_debut_n - known_n

        u22_known_n = int(
            known["U22"].sum()
        )

        known_only_rate = (
            u22_known_n / known_n
            if known_n
            else float("nan")
        )

        lower_bound = (
            u22_known_n / true_debut_n
            if true_debut_n
            else float("nan")
        )

        upper_bound = (
            (u22_known_n + missing_n) / true_debut_n
            if true_debut_n
            else float("nan")
        )

        rows.append({
            "League": league,
            "Year": int(year),
            "TrueDebut_N": true_debut_n,
            "KnownDOB_N": known_n,
            "MissingDOB_N": missing_n,
            "U22_N": u22_known_n,
            "U22_Rate_KnownOnly": known_only_rate,
            "U22_LowerBound": lower_bound,
            "U22_UpperBound": upper_bound,
            "U22_BoundWidth": upper_bound - lower_bound,
        })

    summary = (
        pd.DataFrame(rows)
        .sort_values(["League", "Year"])
        .reset_index(drop=True)
    )

    summary.to_csv(
        OUTPUT_DIR / "q3_u22_annual_summary.csv",
        index=False,
    )

    return summary


def summarize_q3_pooled(q3):
    """
    Calculate pooled Under-22 shares over the full 2018-2024 period.
    """
    rows = []

    for league, g in q3.groupby("LeagueShort"):
        known = g[g["KnownDOB"]].copy()

        rows.append({
            "League": league,
            "TrueDebut_N": len(g),
            "U22_N": int(known["U22"].sum()),
            "U22_Share": (
                known["U22"].sum() / len(g)
                if len(g)
                else float("nan")
            ),
        })

    pooled = pd.DataFrame(rows)

    pooled.to_csv(
        OUTPUT_DIR / "q3_u22_pooled_summary.csv",
        index=False,
    )

    return pooled


# ---------------------------------------------------------------------
# 12. Primary binomial model
# ---------------------------------------------------------------------

def fit_q3_binomial_model(q3):
    """
    Estimate the pre-specified exploratory binomial model:

        logit[P(U22_i = 1)]
        = beta0
        + beta1 * LPL_i
        + beta2 * Time_i
        + beta3 * (LPL_i x Time_i)

    Interpretation:
    beta1 = baseline LPL-LCK difference in 2018
    beta2 = annual trend in the LCK
    beta3 = difference in annual trend between LPL and LCK

    Only observations with verified DOB can enter the binary model.
    """
    model_data = q3[
        q3["KnownDOB"]
    ].copy()

    model_data["U22"] = model_data["U22"].astype(int)

    model = smf.glm(
        formula="U22 ~ LPL * Time",
        data=model_data,
        family=sm.families.Binomial(),
    ).fit()

    # Export coefficient table.
    coefficient_table = pd.DataFrame({
        "Term": model.params.index,
        "Coefficient": model.params.values,
        "SE": model.bse.values,
        "z": model.tvalues.values,
        "p_value": model.pvalues.values,
        "Odds_Ratio": np.exp(model.params.values),
        "CI_Lower": model.conf_int()[0].values,
        "CI_Upper": model.conf_int()[1].values,
    })

    coefficient_table.to_csv(
        OUTPUT_DIR / "q3_binomial_model.csv",
        index=False,
    )

    return model, model_data, coefficient_table


# ---------------------------------------------------------------------
# 13. Model-predicted probabilities
# ---------------------------------------------------------------------

def predict_q3_probabilities(model):
    """
    Generate fitted U22 probabilities for both leagues in every year.
    """
    prediction_rows = []

    for league in ["LCK", "LPL"]:
        for year in range(
            ANALYSIS_START,
            ANALYSIS_END + 1
        ):
            new_data = pd.DataFrame({
                "LPL": [
                    1 if league == "LPL" else 0
                ],
                "Time": [
                    year - ANALYSIS_START
                ],
            })

            pred = model.get_prediction(
                new_data
            ).summary_frame()

            prediction_rows.append({
                "League": league,
                "Year": year,
                "Predicted_U22_Probability": float(
                    pred["mean"].iloc[0]
                ),
                "CI_Lower": float(
                    pred["mean_ci_lower"].iloc[0]
                ),
                "CI_Upper": float(
                    pred["mean_ci_upper"].iloc[0]
                ),
            })

    predictions = pd.DataFrame(
        prediction_rows
    )

    predictions.to_csv(
        OUTPUT_DIR / "q3_predicted_probabilities.csv",
        index=False,
    )

    return predictions


# ---------------------------------------------------------------------
# 14. Model diagnostics
# ---------------------------------------------------------------------

def export_q3_diagnostics(model):
    """
    Report simple binomial-model diagnostics.

    Values close to 1 for deviance/df and Pearson chi-square/df indicate
    no material evidence of overdispersion.
    """
    diagnostics = pd.DataFrame([
        {
            "Statistic": "Residual deviance",
            "Value": model.deviance,
        },
        {
            "Statistic": "Residual df",
            "Value": model.df_resid,
        },
        {
            "Statistic": "Deviance / df",
            "Value": (
                model.deviance
                / model.df_resid
            ),
        },
        {
            "Statistic": "Pearson chi-square",
            "Value": model.pearson_chi2,
        },
        {
            "Statistic": "Pearson chi-square / df",
            "Value": (
                model.pearson_chi2
                / model.df_resid
            ),
        },
    ])

    diagnostics.to_csv(
        OUTPUT_DIR / "q3_model_diagnostics.csv",
        index=False,
    )

    return diagnostics


# ---------------------------------------------------------------------
# 15. Annual U22-share figure
# ---------------------------------------------------------------------

def plot_q3_u22_share(annual_summary):
    """
    Plot the observed annual Under-22 share among TrueDebuts.

    Because all final-sample TrueDebuts have known DOB, the observed share
    equals both the lower and upper missing-DOB bounds.
    """
    fig, ax = plt.subplots(
        figsize=(8.8, 5.4)
    )

    for league, marker in [
        ("LCK", "o"),
        ("LPL", "s"),
    ]:
        g = (
            annual_summary[
                annual_summary["League"] == league
            ]
            .sort_values("Year")
        )

        ax.plot(
            g["Year"],
            g["U22_LowerBound"] * 100,
            marker=marker,
            linewidth=1.8,
            label=league,
        )

    ax.set_title(
        "Under-22 Share among True Debuts, 2018-2024"
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Under-22 share (%)")
    ax.set_xticks(
        list(
            range(
                ANALYSIS_START,
                ANALYSIS_END + 1
            )
        )
    )
    ax.set_ylim(0, 105)
    ax.grid(
        axis="y",
        alpha=0.25,
    )
    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "q3_u22_share_trend.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ---------------------------------------------------------------------
# 16. Optional specification check: no interaction
# ---------------------------------------------------------------------

def fit_q3_no_interaction_check(q3):
    """
    OPTIONAL robustness check only.

    This additive model removes the League x Time interaction:

        U22 ~ LPL + Time

    It is not part of the main Findings specification.
    """
    model_data = q3[
        q3["KnownDOB"]
    ].copy()

    model_data["U22"] = model_data["U22"].astype(int)

    check_model = smf.glm(
        formula="U22 ~ LPL + Time",
        data=model_data,
        family=sm.families.Binomial(),
    ).fit()

    check_table = pd.DataFrame({
        "Term": check_model.params.index,
        "Coefficient": check_model.params.values,
        "SE": check_model.bse.values,
        "p_value": check_model.pvalues.values,
    })

    check_table.to_csv(
        OUTPUT_DIR
        / "q3_optional_no_interaction_check.csv",
        index=False,
    )

    return check_model, check_table


# ---------------------------------------------------------------------
# 17. Run RQ3
# ---------------------------------------------------------------------

def run_q3_analysis(entries):
    """
    Run the complete RQ3 analysis and export all Appendix outputs.
    """
    print(
        "\nPreparing RQ3 TrueDebut sample..."
    )

    q3 = prepare_q3_true_debuts(
        entries
    )

    print(
        "Calculating annual Under-22 shares..."
    )

    annual_summary = summarize_q3_u22(
        q3
    )

    pooled_summary = summarize_q3_pooled(
        q3
    )

    print(
        "Estimating primary binomial model..."
    )

    model, model_data, coefficient_table = fit_q3_binomial_model(
        q3
    )

    predictions = predict_q3_probabilities(
        model
    )

    diagnostics = export_q3_diagnostics(
        model
    )

    plot_q3_u22_share(
        annual_summary
    )

    print(
        "Running optional no-interaction specification check..."
    )

    check_model, check_table = fit_q3_no_interaction_check(
        q3
    )

    print(
        "\nRQ3 analysis complete."
    )

    print(
        "\nAnnual Under-22 summary:"
    )
    print(
        annual_summary.to_string(
            index=False
        )
    )

    print(
        "\nPooled Under-22 summary:"
    )
    print(
        pooled_summary.to_string(
            index=False
        )
    )

    print(
        "\nPrimary binomial model:"
    )
    print(
        coefficient_table.to_string(
            index=False
        )
    )

    print(
        "\nPredicted probabilities:"
    )
    print(
        predictions.to_string(
            index=False
        )
    )

    print(
        "\nModel diagnostics:"
    )
    print(
        diagnostics.to_string(
            index=False
        )
    )

    return {
        "q3_data": q3,
        "annual_summary": annual_summary,
        "pooled_summary": pooled_summary,
        "model": model,
        "coefficient_table": coefficient_table,
        "predictions": predictions,
        "diagnostics": diagnostics,
        "no_interaction_model": check_model,
        "no_interaction_table": check_table,
    }


# ---------------------------------------------------------------------
# Add this line inside the existing main() function after the Q1/RQ2 code:
# ---------------------------------------------------------------------
#
#     q3_results = run_q3_analysis(entries)
#
# ---------------------------------------------------------------------
