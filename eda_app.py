import streamlit as st
import pandas as pd
import numpy as np
import io, contextlib
from Automated_EDA import EDA

st.set_page_config(page_title="Automated EDA Tool", layout="wide")
st.title("🤖 Automated EDA & Data Cleaning App")

uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx", "tsv"])

if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1]
    if file_type == "csv":
        df = pd.read_csv(uploaded_file)
    elif file_type == "xlsx":
        df = pd.read_excel(uploaded_file)
    elif file_type == "tsv":
        df = pd.read_csv(uploaded_file, sep="\t")
    else:
        st.error("Unsupported file type.")
        st.stop()

    st.success(f"✅ Successfully loaded {uploaded_file.name}")
    st.dataframe(df.head())

    # --- Data Overview ---
    with st.expander("📊 Data Overview"):
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
        st.dataframe(df.describe(include='all'))

    # --- Remove Duplicates ---
    if st.button("Remove Duplicates"):
        log_stream = io.StringIO()
        with contextlib.redirect_stdout(log_stream):
            df = EDA.remove_duplicates(df)
        st.text(log_stream.getvalue())
        st.success("Duplicates removed successfully!")

    # --- Data Type Conversion ---
    if st.button("Convert Data Types Automatically"):
        log_stream = io.StringIO()
        with contextlib.redirect_stdout(log_stream):
            df = EDA.data_type_conversion(df)
        st.text(log_stream.getvalue())
        st.success("✅ Data type conversion complete!")

    # --- Missing Value Summary ---
    st.write("### ⚠️ Missing Value Summary")
    null_counts = df.isnull().sum()
    st.dataframe(null_counts[null_counts > 0])

    # --- Fill Missing Values ---
    if st.button("Fill Missing Values Automatically"):
        log_stream = io.StringIO()
        with contextlib.redirect_stdout(log_stream):
            df = EDA.fill_missing_values(df)
        st.text_area("🧾 Cleaning Log", log_stream.getvalue(), height=300)
        st.success("✅ Missing values filled successfully!")

    # --- Cleaned Output ---
    st.write("### 🧾 Cleaned Data Sample")
    st.dataframe(df.head(20))

    # --- Download ---
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Cleaned CSV",
        data=csv,
        file_name="cleaned_dataset.csv",
        mime="text/csv",
    )
else:
    st.info("👆 Please upload a CSV, Excel, or TSV file to begin.")
