import streamlit as st
import pandas as pd
import numpy as np
import io
from Automated_EDA import EDA  # Import your class 

st.set_page_config(page_title="Automated EDA Tool", layout="wide")
 
st.title("🤖 Automated EDA & Data Cleaning App")

# --- 1️⃣ File Upload ---
uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx", "tsv"])

if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1]

    # Load dataset using your EDA class
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
    st.write("### Preview of Data")
    st.dataframe(df.head())

    # --- 2️⃣ Data Overview ---
    with st.expander("📊 Data Overview"):
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = "\n".join(buffer)
        st.text(s)
        st.write("### Summary Statistics")
        st.dataframe(df.describe(include='all'))

    # --- 3️⃣ Duplicate Handling ---
    st.write("### 🧹 Duplicate Rows")
    if st.button("Remove Duplicates"):
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)
        st.success(f"Removed {before - after} duplicate rows.")

    # --- 5️⃣ Data Type Conversion ---
    if st.button("Convert Data Types Automatically"):
        st.info("⏳ Converting column data types...")
        df = EDA.data_type_conversion(df)
        st.success("✅ Data type conversion complete!")

    # --- 4️⃣ Missing Values Overview ---
    st.write("### ⚠️ Missing Value Summary")
    null_counts = df.isnull().sum()
    st.dataframe(null_counts[null_counts > 0])

    # --- 6️⃣ Fill Missing Values ---
    if st.button("Fill Missing Values Automatically"):
        st.info("🧠 Filling missing values using intelligent logic...")
        df = EDA.fill_missing_values(df)
        st.success("✅ Missing values filled successfully!")

    # --- 2️⃣ Data Overview ---
    with st.expander("📊 Data Overview"):
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = "\n".join(buffer)
        st.text(s)
        st.write("### Summary Statistics")
        st.dataframe(df.describe(include='all'))

    # --- 7️⃣ Final Output ---
    st.write("### 🧾 Cleaned Data Sample")
    st.dataframe(df.head(20))

    # --- 8️⃣ Download cleaned file ---
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Cleaned CSV",
        data=csv,
        file_name="cleaned_dataset.csv",
        mime="text/csv",
    )
else:
    st.info("👆 Please upload a CSV, Excel, or TSV file to begin.")
