# Vancouver City FC — Revenue Growth Analytics

A concise, reproducible analysis of stadium operations, merchandise sales, and fanbase engagement to recommend data-backed revenue strategies.

## Overview

- Objective: Clean, analyze, and visualize three datasets to identify short- and long-term actions across stadium, merchandise, and fanbase pillars.
- Deliverables: Cleaned data, 10 core visualizations, and a slide deck with insights and recommendations.

## Data

- Stadium Operations: Month, Source, Revenue.
- Merchandise Sales: Product_ID, Barcode, Item_Category, Item_Name, Size, Unit_Price, Customer_Age_Group, Customer_Region, Promotion, Channel, Selling_Date, Member_ID, Arrival_Date.
- Fanbase Engagement: Membership_ID, Age_Group, Games_Attended, Seasonal_Pass, Customer_Region.
- Known inconsistency: Age group vs region; flagged in cross-table checks.

## Repository Structure

- src/: original files
- data_cleaned/: cleaned CSVs
- scripts/: cleaning_pipeline.py, visualize_case.py
- figures/: exported PNG charts
- slides/: presentation assets
