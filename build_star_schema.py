from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


INPUT_CSV = Path("pc_data.csv")
OUT_DIR = Path("star_schema_output")


def _clean_str(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def build_dim(
    df: pd.DataFrame,
    columns: list[str],
    key_name: str,
    dim_name: str,
) -> pd.DataFrame:
    """Build a dimension table with a surrogate integer key."""
    dim = df[columns].copy()

    # Normalize string columns only; keep numeric columns as numeric.
    for col in columns:
        if pd.api.types.is_object_dtype(dim[col]) or pd.api.types.is_string_dtype(dim[col]):
            dim[col] = dim[col].fillna("").astype(str).map(lambda v: v.strip())

    dim = dim.drop_duplicates().reset_index(drop=True)
    dim.insert(0, key_name, range(1, len(dim) + 1))
    dim.attrs["dim_name"] = dim_name
    return dim


def build_dim_date(all_dates: pd.Series, key_name: str = "date_key") -> pd.DataFrame:
    """Build a classic date dimension. Uses YYYYMMDD as key."""
    dates = pd.to_datetime(all_dates, errors="coerce").dropna().dt.normalize()
    unique_dates = pd.Series(dates.unique()).sort_values().reset_index(drop=True)

    dim = pd.DataFrame({"full_date": unique_dates})
    dim[key_name] = dim["full_date"].dt.strftime("%Y%m%d").astype(int)

    dim["year"] = dim["full_date"].dt.year
    dim["quarter"] = dim["full_date"].dt.quarter
    dim["month"] = dim["full_date"].dt.month
    dim["month_name"] = dim["full_date"].dt.strftime("%B")
    dim["day"] = dim["full_date"].dt.day
    dim["day_of_week"] = dim["full_date"].dt.dayofweek + 1  # Mon=1
    dim["day_name"] = dim["full_date"].dt.strftime("%A")
    dim["is_weekend"] = dim["full_date"].dt.dayofweek.isin([5, 6]).astype(int)

    dim = dim[[key_name, "full_date", "year", "quarter", "month", "month_name", "day", "day_of_week", "day_name", "is_weekend"]]
    dim = dim.sort_values(key_name).reset_index(drop=True)
    return dim


def main() -> None:
    if not INPUT_CSV.exists():
        raise SystemExit(f"Input not found: {INPUT_CSV}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    # -------- Dimensions --------
    dim_geography = build_dim(
        df,
        ["Continent", "Country or State", "Province or City"],
        key_name="geography_key",
        dim_name="dim_geography",
    )

    # Keep shop at a stable grain (name). Shop Age varies across rows, so we keep it as a
    # snapshot attribute in the fact table rather than splitting the shop dimension.
    dim_shop = build_dim(
        df,
        ["Shop Name"],
        key_name="shop_key",
        dim_name="dim_shop",
    )

    dim_customer = build_dim(
        df,
        ["Customer Name", "Customer Surname", "Customer Contact Number", "Customer Email Address"],
        key_name="customer_key",
        dim_name="dim_customer",
    )

    dim_salesperson = build_dim(
        df,
        ["Sales Person Name", "Sales Person Department"],
        key_name="salesperson_key",
        dim_name="dim_salesperson",
    )

    dim_product = build_dim(
        df,
        ["PC Make", "PC Model", "Storage Type", "Storage Capacity", "RAM"],
        key_name="product_key",
        dim_name="dim_product",
    )

    # Junk dimension for low-cardinality order attributes
    dim_sales_context = build_dim(
        df,
        ["Payment Method", "Channel", "Priority"],
        key_name="sales_context_key",
        dim_name="dim_sales_context",
    )

    # Date dimensions (purchase and ship share same dim)
    dim_date = build_dim_date(pd.concat([df["Purchase Date"], df["Ship Date"]], ignore_index=True))

    # -------- Fact table --------
    fact = df.copy()
    fact.insert(0, "transaction_id", range(1, len(fact) + 1))

    # Map geography & shop
    fact = fact.merge(
        dim_geography,
        on=["Continent", "Country or State", "Province or City"],
        how="left",
        validate="many_to_one",
    )

    fact = fact.merge(
        dim_shop,
        on=["Shop Name"],
        how="left",
        validate="many_to_one",
    )

    # Map customer
    fact = fact.merge(
        dim_customer,
        on=["Customer Name", "Customer Surname", "Customer Contact Number", "Customer Email Address"],
        how="left",
        validate="many_to_one",
    )

    # Map salesperson
    fact = fact.merge(
        dim_salesperson,
        on=["Sales Person Name", "Sales Person Department"],
        how="left",
        validate="many_to_one",
    )

    # Map product
    fact = fact.merge(
        dim_product,
        on=["PC Make", "PC Model", "Storage Type", "Storage Capacity", "RAM"],
        how="left",
        validate="many_to_one",
    )

    # Map sales context
    fact = fact.merge(
        dim_sales_context,
        on=["Payment Method", "Channel", "Priority"],
        how="left",
        validate="many_to_one",
    )

    # Map dates to YYYYMMDD int keys; unknown ship dates become 0
    purchase_date_key = pd.to_datetime(fact["Purchase Date"], errors="coerce").dt.strftime("%Y%m%d")
    ship_date_key = pd.to_datetime(fact["Ship Date"], errors="coerce").dt.strftime("%Y%m%d")
    fact["purchase_date_key"] = purchase_date_key.fillna("0").astype(int)
    fact["ship_date_key"] = ship_date_key.fillna("0").astype(int)

    # Keep only keys + measures (and a couple of snapshot attributes)
    fact_sales = fact[
        [
            "transaction_id",
            "purchase_date_key",
            "ship_date_key",
            "geography_key",
            "shop_key",
            "customer_key",
            "salesperson_key",
            "product_key",
            "sales_context_key",
            # Snapshot attribute (can change over time)
            "Shop Age",
            "Credit Score",
            # Measures
            "Cost Price",
            "Sale Price",
            "Discount Amount",
            "Finance Amount",
            "Cost of Repairs",
            "Total Sales per Employee",
            "PC Market Price",
        ]
    ].rename(
        columns={
            "Shop Age": "shop_age",
            "Credit Score": "credit_score",
            "Cost Price": "cost_price",
            "Sale Price": "sale_price",
            "Discount Amount": "discount_amount",
            "Finance Amount": "finance_amount",
            "Cost of Repairs": "cost_of_repairs",
            "Total Sales per Employee": "total_sales_per_employee",
            "PC Market Price": "pc_market_price",
        }
    )

    # Sanity checks: keys should be populated
    key_cols = [
        "geography_key",
        "shop_key",
        "customer_key",
        "salesperson_key",
        "product_key",
        "sales_context_key",
    ]
    missing_keys = {k: int(fact_sales[k].isna().sum()) for k in key_cols}
    if any(v > 0 for v in missing_keys.values()):
        raise SystemExit(f"Missing surrogate keys in fact table: {missing_keys}")

    # -------- Write outputs --------
    dim_geography.to_csv(OUT_DIR / "dim_geography.csv", index=False)
    dim_shop.to_csv(OUT_DIR / "dim_shop.csv", index=False)
    dim_customer.to_csv(OUT_DIR / "dim_customer.csv", index=False)
    dim_salesperson.to_csv(OUT_DIR / "dim_salesperson.csv", index=False)
    dim_product.to_csv(OUT_DIR / "dim_product.csv", index=False)
    dim_sales_context.to_csv(OUT_DIR / "dim_sales_context.csv", index=False)
    dim_date.to_csv(OUT_DIR / "dim_date.csv", index=False)

    fact_sales.to_csv(OUT_DIR / "fact_sales.csv", index=False)

    # Quick row counts summary
    summary = {
        "dim_geography": len(dim_geography),
        "dim_shop": len(dim_shop),
        "dim_customer": len(dim_customer),
        "dim_salesperson": len(dim_salesperson),
        "dim_product": len(dim_product),
        "dim_sales_context": len(dim_sales_context),
        "dim_date": len(dim_date),
        "fact_sales": len(fact_sales),
    }
    (OUT_DIR / "row_counts.txt").write_text(
        "Row counts\n" + "\n".join([f"{k}: {v}" for k, v in summary.items()]) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
