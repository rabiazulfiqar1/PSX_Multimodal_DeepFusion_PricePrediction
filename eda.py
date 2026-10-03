import pandas as pd
import numpy as np

# LOAD DATA

price_data = pd.read_csv("UBL.csv")


fundamental_data = pd.read_csv(
    "UBL_fundamental_features.csv",
    na_values=["-", "--", ""]
)
fundamental_data = fundamental_data.rename(columns={
    "Unnamed: 0": "QUARTER",
    "Unnamed: 1": "DATE"
})
fundamental_data["DATE"] = pd.to_datetime(
    fundamental_data["DATE"].str.extract(r"([A-Z][a-z]{2} \d{2}, \d{4})")[0]
)


news_data = pd.read_excel("UBL_news.xlsx")

# PRICE DATA

# Convert date
price_data["DATE"] = pd.to_datetime(price_data["DATE"])

# Select 5 years
price_data_5yr = price_data[
    price_data["DATE"] >= pd.Timestamp("2021-01-01")
].copy()


# ==========================================
# 1. MISSING VALUES
# ==========================================

print("\nMissing Values:")
print(price_data_5yr.isnull().sum())


# ==========================================
# 2. EXACT DUPLICATES
# ==========================================

duplicate_rows = price_data_5yr[
    price_data_5yr.duplicated(keep=False)
]

print("\nExact Duplicate Rows:")
print(duplicate_rows)

# Remove exact duplicates
price_data_5yr = price_data_5yr.drop_duplicates()


# ==========================================
# 3. VALIDATE PRICE VALUES
# ==========================================

valid_price = (
    (price_data_5yr["OPEN"] > 0) &
    (price_data_5yr["HIGH"] > 0) &
    (price_data_5yr["LOW"] > 0) &
    (price_data_5yr["CLOSE"] > 0) &
    (price_data_5yr["VOLUME"] >= 0) &
    (price_data_5yr["HIGH"] >= price_data_5yr["OPEN"]) &
    (price_data_5yr["HIGH"] >= price_data_5yr["CLOSE"]) &
    (price_data_5yr["HIGH"] >= price_data_5yr["LOW"]) &
    (price_data_5yr["LOW"] <= price_data_5yr["OPEN"]) &
    (price_data_5yr["LOW"] <= price_data_5yr["CLOSE"])
)

invalid_price_rows = price_data_5yr[~valid_price]

print("\nInvalid Price Records:")
print(invalid_price_rows)


# Remove invalid records
price_data_5yr = price_data_5yr[valid_price].copy()


# ==========================================
# 4. CHECK DUPLICATE DATES AGAIN
# ==========================================

duplicate_dates = price_data_5yr[
    price_data_5yr["DATE"].duplicated(keep=False)
]

print("\nRemaining Duplicate Dates:")
print(duplicate_dates)


# ==========================================
# 5. FINAL CHECK
# ==========================================

print("\nFinal Shape:", price_data_5yr.shape)
print("Exact Duplicates:", price_data_5yr.duplicated().sum())
print("Duplicate Dates:", price_data_5yr["DATE"].duplicated().sum())

# FUNDAMENTALS

print("\nFundamental Data:")
print(fundamental_data.head())

print("\nShape:", fundamental_data.shape)
print("\nMissing Values:")
print(fundamental_data.isnull().sum())

print("\nRows containing missing values:")
print(
    fundamental_data[
        fundamental_data.isnull().any(axis=1)
    ].to_string(index=False)
)

fundamental_data = fundamental_data.sort_values("DATE").reset_index(drop=True)

fundamental_data["Revenue Growth (YoY)"] = (
    fundamental_data["Revenue"].pct_change(4) * 100
)

fundamental_data["Net Income Growth"] = (
    fundamental_data["Net Income"].pct_change(4) * 100
)

fundamental_data["EPS Growth"] = (
    fundamental_data["EPS (Basic)"].pct_change(4) * 100
)

print("\nDuplicate Rows:")
print(fundamental_data.duplicated().sum())

# ==========================================
# ALIGN FUNDAMENTALS WITH DAILY PRICE DATA
# ==========================================

# ==========================================
# ALIGN DATE RANGES FIRST
# ==========================================

# Sort by date
price_data_5yr = price_data_5yr.sort_values("DATE").reset_index(drop=True)
fundamental_data = fundamental_data.sort_values("DATE").reset_index(drop=True)

# Find common date range
start_date = max(
    price_data_5yr["DATE"].min(),
    fundamental_data["DATE"].min()
)

end_date = min(
    price_data_5yr["DATE"].max(),
    fundamental_data["DATE"].max()
)

print("Common Date Range:")
print(start_date, "to", end_date)

# Restrict both datasets to common range
price_aligned = price_data_5yr[
    (price_data_5yr["DATE"] >= start_date) &
    (price_data_5yr["DATE"] <= end_date)
].copy()

fundamental_aligned = fundamental_data[
    (fundamental_data["DATE"] >= start_date) &
    (fundamental_data["DATE"] <= end_date)
].copy()

print("\nPrice data after date alignment:")
print(price_aligned.shape)

print("\nFundamental data after date alignment:")
print(fundamental_aligned.shape)

# ==========================================
# MERGE QUARTERLY FUNDAMENTALS WITH DAILY PRICE
# ==========================================

aligned_data = pd.merge_asof(
    price_aligned,
    fundamental_aligned,
    on="DATE",
    direction="backward"
)

print("\nAligned Data:")
print(aligned_data.head())

print("\nShape:", aligned_data.shape)

print("\nMissing Values:")
print(aligned_data.isnull().sum())


fundamental_columns = [
    "Revenue",
    "Revenue Growth (YoY)",
    "Net Income",
    "Net Income Growth",
    "EPS (Basic)",
    "EPS (Diluted)",
    "EPS Growth",
    "Total Assets",
    "Cash & Equivalents"
]

print("\nMissing Fundamental Values After Alignment:")
print(aligned_data[fundamental_columns].isnull().sum())