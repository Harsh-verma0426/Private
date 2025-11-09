import numpy as np
import pandas as pd
import re


class EDA:
    @staticmethod
    def get_useful_columns(df: pd.DataFrame):
        """Remove useless columns like IDs or constants."""
        return [
            col for col in df.columns
            if not col.lower().startswith("unnamed")
            and df[col].nunique() > 1
            and df[col].nunique() < len(df) * 0.9
        ]

    @staticmethod
    def get_columns_by_type(df: pd.DataFrame, type_choice: str, useful_cols=None):
        """Return columns filtered by data type."""
        if useful_cols is None:
            useful_cols = df.columns

        if type_choice == "Numeric":
            return [col for col in useful_cols if pd.api.types.is_numeric_dtype(df[col])]
        elif type_choice == "Categorical":
            return [col for col in useful_cols if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col])]
        elif type_choice == "Datetime":
            return [col for col in useful_cols if pd.api.types.is_datetime64_any_dtype(df[col])]
        return []

    @staticmethod
    def sample_dataframe(df: pd.DataFrame, n=10000):
        """Return a random sample of the dataframe to avoid lag."""
        return df.sample(min(len(df), n), random_state=42)

    @staticmethod
    def correlation_matrix(df: pd.DataFrame):
        """Return numeric correlation matrix."""
        return df.select_dtypes(include=np.number).corr()
    
    @staticmethod
    def get_data_overview(df: pd.DataFrame):
        """Return info summary: column name, non-null count, and dtype."""
        return pd.DataFrame({
            "Column": df.columns,
            "Non-Null Count": [df[col].count() for col in df.columns],
            "Dtype": [df[col].dtype for col in df.columns]
        })

    @staticmethod
    def get_statistical_summary(df: pd.DataFrame):
        """Return descriptive statistics for all columns."""
        return df.describe(include='all')
        
    @staticmethod
    def get_useful_columns(df: pd.DataFrame, missing_threshold: float = 0.9, id_threshold: float = 0.9):
        """
        Filters out useless columns for visualization:
        - Columns starting with 'Unnamed'
        - Columns with one unique value (constants)
        - Columns with too many unique values (likely IDs)
        - Columns with too many missing values
        """
        useful_cols = []
        for col in df.columns:
            nunique = df[col].nunique(dropna=True)
            missing_ratio = df[col].isna().mean()

            if col.lower().startswith("unnamed"):
                continue
            if nunique <= 1:
                continue
            if nunique > len(df) * id_threshold:
                continue
            if missing_ratio > missing_threshold:
                continue

            useful_cols.append(col)

        return useful_cols

    @staticmethod
    def prepare_bivariate_data(df: pd.DataFrame, x_col: str, y_col: str, sample_limit=10000):
        """
        Prepare data and suggest plot type for two-column relationship.
        Returns a dict with:
        - 'plot_type': str
        - 'data': pd.DataFrame
        - 'summary': str (description or insight)
        """
        if x_col not in df.columns or y_col not in df.columns:
            return {"plot_type": "unsupported", "data": None, "summary": "Invalid columns selected."}

        data = df[[x_col, y_col]].dropna()
        if len(data) > sample_limit:
            data = data.sample(sample_limit, random_state=42)

        # Numeric vs Numeric → Scatter + Correlation
        if pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
            corr = data[x_col].corr(data[y_col])
            summary = f"Both `{x_col}` and `{y_col}` are numeric. Correlation coefficient: **{corr:.2f}**"
            return {"plot_type": "scatter", "data": data, "summary": summary}

        # Categorical vs Numeric → Bar (mean aggregation)
        elif pd.api.types.is_object_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
            mean_df = data.groupby(x_col)[y_col].mean().sort_values(ascending=False).head(20).reset_index()
            mean_df.columns = [x_col, f"Mean {y_col}"]
            summary = f"`{x_col}` is categorical and `{y_col}` is numeric. Showing top 20 mean values."
            return {"plot_type": "bar", "data": mean_df, "summary": summary}

        # Datetime vs Numeric → Line trend
        elif pd.api.types.is_datetime64_any_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
            sorted_df = data.sort_values(by=x_col)
            summary = f"Trend of `{y_col}` over time (`{x_col}`)."
            return {"plot_type": "line", "data": sorted_df, "summary": summary}

        # Categorical vs Categorical → Grouped bar
        elif pd.api.types.is_object_dtype(df[x_col]) and pd.api.types.is_object_dtype(df[y_col]):
            cross = data.groupby([x_col, y_col]).size().unstack(fill_value=0)
            summary = f"Grouped frequency count between `{x_col}` and `{y_col}`."
            return {"plot_type": "grouped_bar", "data": cross, "summary": summary}

        else:
            return {"plot_type": "unsupported", "data": None, "summary": "Unsupported combination of column types."}