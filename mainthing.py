# cleaning_pipeline.py

import pandas as pd
import numpy as np
import re



# Canonical age groups; add/adjust to match your raw labels
AGE_GROUP_ORDER = [
    "Under 18",
    "18-24",
    "25-34",
    "35-44",
    "45-54",
    "55+"
]

# Mapping patterns to canonical age groups
AGE_PATTERNS = [
    (re.compile(r'^(u(?:nder)?\s*1?8|<\s*18|0-1?7|youth)$', re.I), "Under 18"),
    (re.compile(r'^(18\s*[-–to/]\s*24|18_24)$', re.I), "18-24"),
    (re.compile(r'^(25\s*[-–to/]\s*34|25_34)$', re.I), "25-34"),
    (re.compile(r'^(35\s*[-–to/]\s*44|35_44)$', re.I), "35-44"),
    (re.compile(r'^(45\s*[-–to/]\s*54|45_54)$', re.I), "45-54"),
    (re.compile(r'^(55\s*\+|55\s*[-–to/]\s*99|55_99|senior|55plus)$', re.I), "55+"),
]

DOMESTIC_COUNTRIES = {"Canada"}  #if needed

CHANNEL_MAP = {
    "online": "Online",
    "web": "Online",
    "ecomm": "Online",
    "in-store": "In-Store",
    "instore": "In-Store",
    "stadium": "In-Store",
    "kiosk": "In-Store"
}

BOOL_TRUE = {"true", "t", "yes", "y", "1", 1, True}
BOOL_FALSE = {"false", "f", "no", "n", "0", 0, False}

# ---------------------------
# Helper functions
# ---------------------------

def _strip_lower(x):
    if pd.isna(x):
        return x
    return str(x).strip().lower()

def normalize_age_group(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    # quick canonical if already exact
    if s in AGE_GROUP_ORDER:
        return s
    s_compact = _strip_lower(s)
    s_compact = s_compact.replace("years", "").replace("yrs", "").replace("yo", "").strip()
    s_compact = re.sub(r'\s+', ' ', s_compact)
    for pat, canon in AGE_PATTERNS:
        if pat.match(s_compact):
            return canon
    # attempts like "18 to 24", "18 – 24"
    s_compact = s_compact.replace(" to ", "-").replace("–", "-").replace("/", "-").replace("_", "-")
    if re.match(r'^\d{1,2}-\d{1,2}$', s_compact):
        lo, hi = s_compact.split("-")
        try:
            lo = int(lo); hi = int(hi)
            candidates = {
                (0,17): "Under 18",
                (18,24): "18-24",
                (25,34): "25-34",
                (35,44): "35-44",
                (45,54): "45-54",
                (55,120): "55+"
            }
            for (a,b), canon in candidates.items():
                if lo>=a and hi<=b:
                    return canon
        except:
            pass
    # fallback
    return np.nan

def normalize_region_type_from_text(val):
    # Returns Domestic/International for inputs like "Domestic", "International", countries, or regions
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    sl = s.lower()
    if sl in {"domestic", "canada", "ca"}:
        return "Domestic"
    if sl in {"international", "int'l", "intl", "overseas", "global"}:
        return "International"
    # country name heuristic
    return "Domestic" if s in DOMESTIC_COUNTRIES else "International"

def normalize_channel(val):
    if pd.isna(val):
        return np.nan
    sl = _strip_lower(val)
    return CHANNEL_MAP.get(sl, val if val in {"Online","In-Store"} else ("Online" if "online" in sl or "web" in sl else ("In-Store" if "store" in sl or "stadium" in sl or "kiosk" in sl else val)))

def coerce_bool(val):
    if pd.isna(val):
        return np.nan
    if val in BOOL_TRUE or (isinstance(val, str) and _strip_lower(val) in BOOL_TRUE):
        return True
    if val in BOOL_FALSE or (isinstance(val, str) and _strip_lower(val) in BOOL_FALSE):
        return False
    return np.nan

def to_datetime_col(s):
    return pd.to_datetime(s, errors="coerce", infer_datetime_format=True)

def to_int_nonneg(s):
    out = pd.to_numeric(s, errors="coerce").astype("Int64")
    out = out.where((out.isna()) | (out>=0))
    return out

def clean_string(s):
    return s.astype("string").str.strip()

# ---------------------------
# Stadium Operations cleaning
# ---------------------------

def clean_stadium(df):
    df = df.copy()
    # Standardize columns
    df.columns = [c.strip() for c in df.columns]
    required = {"Month","Source","Revenue"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Stadium missing columns: {missing}")

    # Month as int 1-12
    df["Month"] = pd.to_numeric(df["Month"], errors="coerce").astype("Int64")
    df.loc[~df["Month"].between(1,12), "Month"] = pd.NA

    # Source tidy
    df["Source"] = df["Source"].astype("string").str.strip()

    # Revenue numeric (can include negative values for costs)
    df["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce")
    df["Is_Cost"] = df["Revenue"] < 0

    # Drop exact duplicates
    df = df.drop_duplicates()

    return df

# ---------------------------
# Merchandise Sales cleaning
# ---------------------------

def clean_merchandise(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Expected columns
    expected = {
        "Product_ID","Barcode","Item_Category","Item_Name","Size",
        "Unit_Price","Customer_Age_Group","Customer_Region","Promotion",
        "Channel","Selling_Date","Member_ID","Arrival_Date"
    }
    missing = expected - set(df.columns)
    if missing:
        # Try to accommodate common typos/breaks in PDF copy
        rename_map = {}
        for c in df.columns:
            if c.replace(" ", "").lower() == "customer_age_group":
                rename_map[c] = "Customer_Age_Group"
        if rename_map:
            df = df.rename(columns=rename_map)
            missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Merchandise missing columns: {missing}")

    # Trim strings
    for col in ["Item_Category","Item_Name","Size","Customer_Region","Channel"]:
        df[col] = clean_string(df[col])

    # Numeric price
    df["Unit_Price"] = pd.to_numeric(df["Unit_Price"], errors="coerce")
    df.loc[df["Unit_Price"] < 0, "Unit_Price"] = np.nan

    # Age group normalization
    df["Customer_Age_Group_Normalized"] = df["Customer_Age_Group"].apply(normalize_age_group)

    # Region handling
    df["Region_Type"] = df["Customer_Region"].apply(normalize_region_type_from_text)
    # Keep original country/region text for context
    df["Region_Name"] = df["Customer_Region"]

    # Promotion to boolean
    df["Promotion"] = df["Promotion"].apply(coerce_bool)

    # Channel normalization
    df["Channel"] = df["Channel"].apply(normalize_channel)

    # Datetimes
    df["Selling_Date"] = to_datetime_col(df["Selling_Date"])
    df["Arrival_Date"] = to_datetime_col(df["Arrival_Date"])

    # Delivery latency: only for Online (in-store should be NaT)
    online_mask = df["Channel"].eq("Online")
    df.loc[~online_mask, "Arrival_Date"] = pd.NaT
    df["Delivery_Latency_Days"] = (df["Arrival_Date"] - df["Selling_Date"]).dt.days
    df.loc[~online_mask, "Delivery_Latency_Days"] = np.nan

    # Member_ID as string key
    df["Member_ID"] = df["Member_ID"].astype("string").str.strip()

    # Drop perfect duplicates
    df = df.drop_duplicates()

    # Transaction-level de-duplication heuristic: same barcode, buyer, timestamp => keep first
    df = df.sort_values(["Selling_Date"]).drop_duplicates(subset=["Barcode","Member_ID","Selling_Date"], keep="first")

    return df

# ---------------------------
# Fanbase Engagement cleaning
# ---------------------------

def clean_fanbase(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    expected = {"Membership_ID","Age_Group","Games_Attended","Seasonal_Pass","Customer_Region"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Fanbase missing columns: {missing}")

    # Key fields
    df["Membership_ID"] = df["Membership_ID"].astype("string").str.strip()

    # Age group normalization
    df["Age_Group_Normalized"] = df["Age_Group"].apply(normalize_age_group)

    # Games attended non-negative int
    df["Games_Attended"] = to_int_nonneg(df["Games_Attended"])

    # Seasonal_Pass boolean
    df["Seasonal_Pass"] = df["Seasonal_Pass"].apply(coerce_bool)

    # Region normalization
    df["Customer_Region"] = clean_string(df["Customer_Region"])
    df["Region_Type"] = df["Customer_Region"].apply(normalize_region_type_from_text)
    df["Region_Name"] = df["Customer_Region"]

    # Drop exact duplicates
    df = df.drop_duplicates()

    return df

# ---------------------------
# Cross-table consistency checks
# ---------------------------

def cross_checks(merch_df, fan_df):
    report = {}

    # If your Member_ID corresponds to Membership_ID, we can try a direct join; if not, skip
    if "Member_ID" in merch_df.columns and "Membership_ID" in fan_df.columns:
        # Attempt a left join on identical IDs
        merged = merch_df.merge(
            fan_df[["Membership_ID","Age_Group_Normalized","Region_Type"]].rename(columns={
                "Membership_ID":"Member_ID",
                "Age_Group_Normalized":"FB_Age_Group_Normalized",
                "Region_Type":"FB_Region_Type"
            }),
            on="Member_ID",
            how="left",
            validate="m:1"
        )

        # Age conflicts: same member shows materially different normalized age groups
        age_conflict = merged[
            merged["FB_Age_Group_Normalized"].notna() &
            merged["Customer_Age_Group_Normalized"].notna() &
            (merged["FB_Age_Group_Normalized"] != merged["Customer_Age_Group_Normalized"])
        ]
        report["age_conflict_rows"] = age_conflict.shape[0]

        # Region conflicts: merchandise region type vs fan region type disagree
        region_conflict = merged[
            merged["FB_Region_Type"].notna() &
            merged["Region_Type"].notna() &
            (merged["FB_Region_Type"] != merged["Region_Type"])
        ]
        report["region_conflict_rows"] = region_conflict.shape[0]

        # Return merged sample of conflicts for review
        report["age_conflicts_sample"] = age_conflict.head(25)
        report["region_conflicts_sample"] = region_conflict.head(25)
    else:
        report["note"] = "Member_ID and Membership_ID could not be aligned; provide a mapping if available."

    return report

# ---------------------------
# Main execution
# ---------------------------

if __name__ == "__main__":
    # Update file paths if your filenames differ
    stadium_path = "src/BOLT UBC First Byte - Stadium Operations.xlsx"
    merch_path = "src/BOLT UBC First Byte - Merchandise Sales.xlsx"
    fan_path = "src/BOLT UBC First Byte - Fanbase Engagement.xlsx"

    stadium_raw = pd.read_excel(stadium_path)
    merch_raw = pd.read_excel(merch_path)
    fan_raw = pd.read_excel(fan_path)

    stadium_cln = clean_stadium(stadium_raw)
    merch_cln = clean_merchandise(merch_raw)
    fan_cln = clean_fanbase(fan_raw)

    # Optional cross-table checks for age/region inconsistencies
    consistency = cross_checks(merch_cln, fan_cln)

    # Save cleaned outputs
    stadium_cln.to_csv("stadium_operations_clean.csv", index=False)
    merch_cln.to_csv("merchandise_sales_clean.csv", index=False)
    fan_cln.to_csv("fanbase_engagement_clean.csv", index=False)

    # Save quick anomaly reports
    pd.Series(consistency).to_json("cross_table_consistency_summary.json")
    if "age_conflicts_sample" in consistency:
        consistency["age_conflicts_sample"].to_csv("age_conflicts_sample.csv", index=False)
    if "region_conflicts_sample" in consistency:
        consistency["region_conflicts_sample"].to_csv("region_conflicts_sample.csv", index=False)