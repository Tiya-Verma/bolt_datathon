# visualize_case.py
# core Matplotlib visualizations for the datasets.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.style.use("seaborn-v0_8-whitegrid")
FIGDIR = Path("figures")
FIGDIR.mkdir(exist_ok=True)

AGE_ORDER = ["Under 18","18-24","25-34","35-44","45-54","55+"]

def ensure_month(df, date_col=None, month_col="Month"):
    # If Month exists (1-12), keep; else derive from date_col or index
    if month_col in df.columns:
        return df
    if date_col and date_col in df.columns:
        dt = pd.to_datetime(df[date_col], errors="coerce")
        df["Month"] = dt.dt.month
    return df

def pct_share(df, group_cols, value_col):
    g = df.groupby(group_cols, dropna=False, observed=True)[value_col].sum().reset_index()
    total = g.groupby(group_cols[0], observed=True)[value_col].transform("sum")
    g["Share"] = np.where(total.eq(0), np.nan, g[value_col] / total)
    return g

def safe_read_csv(path):
    return pd.read_csv(path) if Path(path).exists() else None

def safe_read_excel(path):
    return pd.read_excel(path) if Path(path).exists() else None

def infer_region_type(s):
    if pd.isna(s): return np.nan
    v = str(s).strip().lower()
    if v in {"domestic", "canada", "ca"}: return "Domestic"
    if v in {"international","intl","int'l","overseas","global"}: return "International"
    # country heuristic
    return "Domestic" if str(s).strip() in {"Canada"} else "International"

# ---------- Load data ----------
stadium = safe_read_csv("bolt_datathon/cleaned dataset/stadium_operations_clean.csv")
if stadium is None:
    # fallback to provided Excel if cleaned CSV not present
    stadium = safe_read_excel("BOLT-UBC-First-Byte-Stadium-Operations.xlsx")

merch = safe_read_csv("bolt_datathon/cleaned dataset/merchandise_sales_clean.csv")
fan = safe_read_csv("bolt_datathon/cleaned dataset/fanbase_engagement_clean.csv")

# ---------- Stadium visuals (3) ----------
if stadium is not None:
    # Expect columns: Month, Source, Revenue
    stadium = stadium.copy()
    stadium.columns = [c.strip() for c in stadium.columns]
    stadium = ensure_month(stadium, month_col="Month")
    # Aggregate to monthly by source
    st_m = stadium.groupby(["Month","Source"], observed=True, dropna=False)["Revenue"].sum().reset_index()
    pivot_st = st_m.pivot(index="Month", columns="Source", values="Revenue").fillna(0).sort_index()

    # 1) Multi-line monthly revenue by Source
    ax = pivot_st.plot(kind="line", marker="o", figsize=(10,6))
    ax.set_title("Stadium: Monthly Revenue by Source")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue")
    ax.legend(title="Source", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(FIGDIR / "stadium_1_monthly_by_source_lines.png", dpi=200)
    plt.close()

    # 2) 100% stacked bar: revenue mix share by Source
    shares = pivot_st.div(pivot_st.sum(axis=1).replace(0, np.nan), axis=0)
    shares.plot(kind="bar", stacked=True, figsize=(10,6))
    plt.title("Stadium: Revenue Mix Share by Source (100%)")
    plt.xlabel("Month"); plt.ylabel("Share")
    plt.legend(title="Source", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(FIGDIR / "stadium_2_mix_share_100pct.png", dpi=200)
    plt.close()

    # 3) Premium vs Total by month (bar + line)
    # Guess premium label if present
    possible_premium_labels = [c for c in pivot_st.columns if "premium" in str(c).lower()]
    if possible_premium_labels:
        prem_col = possible_premium_labels[0]
        total = pivot_st.sum(axis=1)
        fig, ax1 = plt.subplots(figsize=(10,6))
        ax1.bar(total.index, total.values, color="#9ecae1", label="Total Revenue")
        ax1.set_xlabel("Month"); ax1.set_ylabel("Total Revenue")
        ax2 = ax1.twinx()
        prem_share = np.where(total.values==0, np.nan, pivot_st[prem_col].values/total.values)
        ax2.plot(total.index, prem_share, color="#de2d26", marker="o", label="Premium Share")
        ax2.set_ylim(0,1); ax2.set_ylabel("Premium Share")
        ax1.set_title("Stadium: Premium Revenue vs Total (Share Line)")
        ax1.legend(loc="upper left"); ax2.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(FIGDIR / "stadium_3_premium_vs_total.png", dpi=200)
        plt.close()

# ---------- Merchandise visuals (3) ----------
if merch is not None:
    m = merch.copy()
    m.columns = [c.strip() for c in m.columns]
    # Expect: Channel, Selling_Date, Item_Category, Region_Type, Promotion, Unit_Price
    if "Selling_Date" in m.columns:
        m["Selling_Date"] = pd.to_datetime(m["Selling_Date"], errors="coerce")
        m["Month"] = m["Selling_Date"].dt.month
    # Revenue proxy: Unit_Price (if no quantity col)
    if "Unit_Price" in m.columns:
        m["Revenue"] = pd.to_numeric(m["Unit_Price"], errors="coerce")
    else:
        # fallback if revenue provided directly
        if "Revenue" not in m.columns:
            m["Revenue"] = np.nan

    # 4) Dual-line: Online vs In-Store monthly revenue with promo overlay (% of orders with Promotion=True)
    g_rev = m.groupby(["Month","Channel"], observed=True, dropna=False)["Revenue"].sum().reset_index()
    pivot_rev = g_rev.pivot(index="Month", columns="Channel", values="Revenue").sort_index()
    fig, ax1 = plt.subplots(figsize=(10,6))
    for col in pivot_rev.columns:
        ax1.plot(pivot_rev.index, pivot_rev[col], marker="o", label=str(col))
    ax1.set_title("Merch: Monthly Revenue by Channel")
    ax1.set_xlabel("Month"); ax1.set_ylabel("Revenue")
    ax1.legend(title="Channel", loc="upper left")

    if "Promotion" in m.columns:
        promo_rate = m.groupby("Month", observed=True)["Promotion"].mean().reindex(pivot_rev.index)
        ax2 = ax1.twinx()
        ax2.plot(promo_rate.index, promo_rate.values, color="#de2d26", marker="s", linestyle="--", label="Promo Rate")
        ax2.set_ylabel("Promo Rate")
        ax2.set_ylim(0,1)
    plt.tight_layout()
    plt.savefig(FIGDIR / "merch_4_monthly_channel_with_promo.png", dpi=200)
    plt.close()

    # 5) Grouped bar: Item_Category revenue by Channel (top 8 categories)
    topcats = m.groupby("Item_Category", dropna=False)["Revenue"].sum().nlargest(8).index
    m_top = m[m["Item_Category"].isin(topcats)]
    g_cat = m_top.groupby(["Item_Category","Channel"], observed=True, dropna=False)["Revenue"].sum().reset_index()
    categories = list(topcats)
    channels = list(g_cat["Channel"].dropna().unique())
    x = np.arange(len(categories))
    width = 0.8 / max(1, len(channels))
    fig, ax = plt.subplots(figsize=(12,6))
    for i, ch in enumerate(channels):
        vals = [g_cat[(g_cat["Item_Category"]==c) & (g_cat["Channel"]==ch)]["Revenue"].sum() for c in categories]
        ax.bar(x + i*width, vals, width, label=str(ch))
    ax.set_xticks(x + (len(channels)-1)*width/2)
    ax.set_xticklabels(categories, rotation=30, ha="right")
    ax.set_title("Merch: Revenue by Item Category and Channel (Top 8)")
    ax.set_xlabel("Item Category"); ax.set_ylabel("Revenue")
    ax.legend(title="Channel")
    plt.tight_layout()
    plt.savefig(FIGDIR / "merch_5_category_by_channel.png", dpi=200)
    plt.close()

    # 6) Box plot: Delivery_Latency_Days by Region_Type (only Online)
    if "Delivery_Latency_Days" in m.columns:
        dfb = m.copy()
        if "Channel" in dfb.columns:
            dfb = dfb[dfb["Channel"].astype(str)=="Online"]
        if "Region_Type" not in dfb.columns and "Customer_Region" in dfb.columns:
            dfb["Region_Type"] = dfb["Customer_Region"].map(infer_region_type)
        groups = [g.dropna().values for _, g in dfb.groupby("Region_Type")["Delivery_Latency_Days"]]
        labels = [str(k) for k, _ in dfb.groupby("Region_Type")]
        if len(groups) >= 1:
            fig, ax = plt.subplots(figsize=(8,6))
            ax.boxplot(groups, labels=labels, showmeans=True)
            ax.set_title("Merch: Delivery Latency by Region Type (Online)")
            ax.set_xlabel("Region Type"); ax.set_ylabel("Latency (days)")
            plt.tight_layout()
            plt.savefig(FIGDIR / "merch_6_delivery_latency_box.png", dpi=200)
            plt.close()

# ---------- Fanbase visuals (3) ----------
if fan is not None:
    f = fan.copy()
    f.columns = [c.strip() for c in f.columns]
    # Normalize age order if present
    if "Age_Group_Normalized" in f.columns:
        f["Age_Group_Normalized"] = pd.Categorical(f["Age_Group_Normalized"], categories=AGE_ORDER, ordered=True)
    elif "Age_Group" in f.columns:
        f["Age_Group_Normalized"] = f["Age_Group"]

    # 7) Grouped bar: Games_Attended frequency buckets by Age_Group
    if "Games_Attended" in f.columns:
        bins = [-0.1,0,2,5,1000]
        labels = ["0","1-2","3-5","6+"]
        f["Attend_Bucket"] = pd.cut(pd.to_numeric(f["Games_Attended"], errors="coerce"), bins=bins, labels=labels)
        g = f.groupby(["Age_Group_Normalized","Attend_Bucket"], observed=True).size().reset_index(name="Count")
        ages = [a for a in AGE_ORDER if a in g["Age_Group_Normalized"].astype(str).unique()]
        buckets = labels
        x = np.arange(len(ages))
        width = 0.8/len(buckets)
        fig, ax = plt.subplots(figsize=(12,6))
        for i, b in enumerate(buckets):
            vals = [g[(g["Age_Group_Normalized"]==a) & (g["Attend_Bucket"]==b)]["Count"].sum() for a in ages]
            ax.bar(x + i*width, vals, width, label=b)
        ax.set_xticks(x + (len(buckets)-1)*width/2)
        ax.set_xticklabels(ages, rotation=0)
        ax.set_title("Fanbase: Attendance Frequency by Age Group")
        ax.set_xlabel("Age Group"); ax.set_ylabel("Count")
        ax.legend(title="Games Attended")
        plt.tight_layout()
        plt.savefig(FIGDIR / "fan_7_attendance_frequency_by_age.png", dpi=200)
        plt.close()

    # 8) Bar: Seasonal_Pass rate by Region_Type
    if "Seasonal_Pass" in f.columns:
        if "Region_Type" not in f.columns and "Customer_Region" in f.columns:
            f["Region_Type"] = f["Customer_Region"].map(infer_region_type)
        sp = f.groupby("Region_Type", observed=True)["Seasonal_Pass"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(8,6))
        ax.bar(sp["Region_Type"].astype(str), sp["Seasonal_Pass"].values, color="#9ecae1")
        for i, v in enumerate(sp["Seasonal_Pass"].values):
            ax.text(i, v + 0.01, f"{v:.0%}", ha="center")
        ax.set_ylim(0, 1.05)
        ax.set_title("Fanbase: Seasonal Pass Rate by Region Type")
        ax.set_xlabel("Region Type"); ax.set_ylabel("Rate")
        plt.tight_layout()
        plt.savefig(FIGDIR / "fan_8_seasonal_pass_rate_by_region.png", dpi=200)
        plt.close()

    # 9) Mean games attended by Age Group (bar with error bars)
    if "Games_Attended" in f.columns:
        agg = f.groupby("Age_Group_Normalized", observed=True)["Games_Attended"].agg(["mean","count","std"]).reset_index()
        agg = agg[agg["Age_Group_Normalized"].astype(str).isin(AGE_ORDER)]
        y = agg["mean"].values
        err = (agg["std"] / np.sqrt(agg["count"].clip(lower=1))).fillna(0).values
        fig, ax = plt.subplots(figsize=(10,6))
        ax.bar(agg["Age_Group_Normalized"].astype(str), y, yerr=err, capsize=4, color="#a1d99b")
        ax.set_title("Fanbase: Average Games Attended by Age Group")
        ax.set_xlabel("Age Group"); ax.set_ylabel("Avg. Games")
        plt.tight_layout()
        plt.savefig(FIGDIR / "fan_9_avg_games_by_age.png", dpi=200)
        plt.close()

# ---------- Cross-pillar KPI view (1) ----------
# 10) 2x2 dashboard style KPIs (saved as one figure)
fig, axes = plt.subplots(2, 2, figsize=(12,8))
idx = 0

# KPI A: Stadium total revenue by month
if stadium is not None:
    s = stadium.copy()
    s = ensure_month(s, month_col="Month")
    k = s.groupby("Month", dropna=False)["Revenue"].sum().reset_index()
    axes.flat[idx].bar(k["Month"], k["Revenue"], color="#9ecae1")
    axes.flat[idx].set_title("Stadium Total Revenue by Month"); idx += 1

# KPI B: Merch total revenue by month
if merch is not None:
    msum = merch.copy()
    if "Selling_Date" in msum.columns:
        msum["Selling_Date"] = pd.to_datetime(msum["Selling_Date"], errors="coerce")
        msum["Month"] = msum["Selling_Date"].dt.month
    if "Revenue" not in msum.columns and "Unit_Price" in msum.columns:
        msum["Revenue"] = pd.to_numeric(msum["Unit_Price"], errors="coerce")
    km = msum.groupby("Month", dropna=False)["Revenue"].sum().reset_index()
    axes.flat[idx].plot(km["Month"], km["Revenue"], marker="o", color="#31a354")
    axes.flat[idx].set_title("Merch Total Revenue by Month"); idx += 1

# KPI C: Merch Promo Rate by month
if merch is not None and "Promotion" in merch.columns:
    prm = merch.copy()
    if "Selling_Date" in prm.columns:
        prm["Selling_Date"] = pd.to_datetime(prm["Selling_Date"], errors="coerce")
        prm["Month"] = prm["Selling_Date"].dt.month
    kp = prm.groupby("Month", dropna=False)["Promotion"].mean().reset_index()
    axes.flat[idx].plot(kp["Month"], kp["Promotion"], marker="s", color="#de2d26")
    axes.flat[idx].set_ylim(0,1); axes.flat[idx].set_title("Merch Promo Rate by Month"); idx += 1

# KPI D: Fan Seasonal Pass rate (overall)
if fan is not None and "Seasonal_Pass" in fan.columns:
    rate = fan["Seasonal_Pass"].mean()
    axes.flat[idx].bar(["Seasonal Pass Rate"], [rate], color="#bcbddc")
    axes.flat[idx].set_ylim(0,1.05)
    for i, v in enumerate([rate]):
        axes.flat[idx].text(i, v + 0.02, f"{v:.0%}", ha="center")
    axes.flat[idx].set_title("Fan Seasonal Pass Rate")

plt.tight_layout()
plt.savefig(FIGDIR / "cross_10_kpi_dashboard.png", dpi=200)
plt.close()

print("Saved visualizations to ./figures")