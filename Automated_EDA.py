import pandas as pd
import numpy as np
import re

class EDA:

    def data_loading(df: str):
        filetype = df.split(".")[-1]
        if filetype == "csv":
            data = pd.read_csv(df)
        elif filetype == "xlsx":
            data = pd.read_excel(df)
        elif filetype == "tsv":
            data = pd.read_csv(df, sep="\t")
        else:
            raise ValueError("Unsupported file type")

        return data

    def data_overview(df: pd.DataFrame):
        print("Data Overview:")
        print(df.info())
        print("\nFirst 5 Rows:")
        print(df.head())
        print("\nStatistical Summary:")
        print(df.describe())
    
    def remove_duplicates(df: pd.DataFrame):
        while True:
            choice = input("Do you want to remove duplicate rows? (yes/no): ").strip().lower()
            if choice == 'no' or choice!= 'n':
                return df
            elif choice == 'yes' or choice == 'y':
                initial_count = len(df)
                df_cleaned = df.drop_duplicates()
                final_count = len(df_cleaned)
                print(f"Removed {initial_count - final_count} duplicate rows.")
                return df_cleaned
            else:
                print("Invalid input. Please enter 'yes' or 'no'.")

    def missing_values(df: pd.DataFrame):
        null_counts = df.isnull().sum()
        print("Missing Values in Each Column:")
        print(null_counts[null_counts > 0])

    def data_type_conversion(df: pd.DataFrame):

        log_entries = []

        date_regex = r'^\s*(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}|[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4})\s*$'
        numeric_regex = r'^[\+\-]?(?:\d+|\d{1,3}(?:,\d{3})+)?(?:\.\d+)?(?:[eE][\+\-]?\d+)?$'

        for col in df.columns:
            old_type = df[col].dtype
            series = df[col].astype(str).str.strip()
            non_null = series[series.notna() & (series != 'nan')]

            if non_null.empty:
                log_entries.append([col, old_type, old_type, 0, "empty/unchanged"])
                continue

            date_mask = non_null.str.match(date_regex, na=False)
            date_ratio = date_mask.sum() / len(non_null)

            if date_ratio > 0.5 or any(k in col.lower() for k in ("date", "time", "ship", "order", "dob")):
                parsed = pd.to_datetime(df[col], errors='coerce', dayfirst=True, infer_datetime_format=True)
                success = parsed.notna().sum() / len(df[col])
                if success > 0.5:
                    df[col] = parsed
                    log_entries.append([col, old_type, "datetime64[ns]", round(success * 100, 2), "date pattern matched"])
                    continue

            candidate = non_null[~non_null.str.match(date_regex, na=False)]
            cand_clean = candidate.str.replace(r'\s+', '', regex=True).str.replace(',', '', regex=False)
            numeric_mask = cand_clean.str.match(numeric_regex, na=False)
            numeric_ratio = numeric_mask.sum() / len(non_null)

            if numeric_ratio > 0.8:
                cleaned = non_null.str.replace(r'[^\d\.\-\+eE,]', '', regex=True).str.replace(',', '', regex=False)
                df[col] = pd.to_numeric(cleaned, errors='coerce')

                if np.all(df[col].dropna() == np.floor(df[col].dropna())):
                    df[col] = df[col].astype("Int64")
                    new_type = "int"
                else:
                    new_type = "float"

                log_entries.append([col, old_type, new_type, round(numeric_ratio * 100, 2), "numeric pattern matched"])
                continue

            log_entries.append([col, old_type, old_type, 0, "unchanged"])

        log_df = pd.DataFrame(log_entries, columns=["Column", "Old Type", "New Type", "Confidence (%)", "Reason"])
        print(f"Conversion Log: {log_df}")
        return df

    def fill_missing_values(df: pd.DataFrame): 

        for col in df.columns:

            if df[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(df[col]):
                    skewness = df[col].skew()
                    if abs(skewness) < 0.5:
                        fill_value = df[col].mean()
                        print(f"Filled missing values in numeric column '{col}' with mean: {fill_value}.")
                    else:
                        fill_value = df[col].median()
                        print(f"Filled missing values in numeric column '{col}' with median: {fill_value}.")
                    df[col].fillna(fill_value, inplace=True)
            
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    fill_value = df[col].mode()[0]
                    df[col].fillna(fill_value, inplace=True)
                    print(f"Filled missing values in datetime column '{col}' with mode: {fill_value}.")

                elif pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                    fill_value = df[col].mode()[0]
                    df[col].fillna(fill_value, inplace=True)
                    print(f"Filled missing values in categorical column '{col}' with mode: {fill_value}.")

        return df

    def main(file_type: str = None):
        data = EDA.data_loading(file_type)
        EDA.data_overview(data)
        data = EDA.remove_duplicates(data)
        EDA.missing_values(data)
        data = EDA.data_type_conversion(data)
        data = EDA.fill_missing_values(data)
        print("Data cleaning completed.")
