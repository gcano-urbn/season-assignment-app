"""Streamlit app for visualizing garment seasonality analysis."""

import io
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from google.auth.transport.requests import Request
from google.cloud import storage
from google.oauth2.credentials import Credentials

# App configuration
st.set_page_config(page_title="Seasonality Viewer", page_icon="📊", layout="wide")

# GCS configuration
GCS_BUCKET = st.secrets["gcs"]["bucket"]
GCS_PREFIX = st.secrets["gcs"]["prefix"]


@st.cache_resource
def get_gcs_client():
    """Get GCS client using Streamlit secrets (authorized_user credentials)."""
    creds_info = st.secrets["gcp_user_credentials"]
    credentials = Credentials(
        token=None,
        refresh_token=creds_info["refresh_token"],
        token_uri="https://oauth2.googleapis.com/token",
        client_id=creds_info["client_id"],
        client_secret=creds_info["client_secret"],
    )
    credentials.refresh(Request())
    return storage.Client(credentials=credentials, project=creds_info.get("quota_project_id", "rental-ds"))


def download_blob_to_memory(client: storage.Client, blob_name: str) -> bytes:
    """Download a blob from GCS to memory."""
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(f"{GCS_PREFIX}{blob_name}")
    return blob.download_as_bytes()


@st.cache_resource
def load_data():
    """Load all required data files from GCS."""
    client = get_gcs_client()

    # Main evaluation dataframe - index by Choice ID for fast lookup
    csv_bytes = download_blob_to_memory(client, "combined_eval_df_w_ensemble.csv")
    df = pd.read_csv(io.BytesIO(csv_bytes))
    df.set_index("Choice ID", inplace=True, drop=False)

    # Prophet forecasts
    pkl_bytes = download_blob_to_memory(client, "dict_choice_prophet.pkl")
    dict_choice_prophet = pickle.loads(pkl_bytes)

    # Detected peaks and troughs
    pkl_bytes = download_blob_to_memory(client, "dict_detected_peaks.pkl")
    dict_detected_peaks = pickle.loads(pkl_bytes)

    pkl_bytes = download_blob_to_memory(client, "dict_detected_troughs.pkl")
    dict_detected_troughs = pickle.loads(pkl_bytes)

    # Weekly orders data - pre-group by choice_id for O(1) lookup
    try:
        csv_bytes = download_blob_to_memory(client, "df_weekly_choice_orders.csv")
        df_weekly_orders = pd.read_csv(
            io.BytesIO(csv_bytes),
            names=["choice_id", "week_id", "image_url", "num_orders", "mean_orders_per_week", "total_weeks_history"],
            index_col=0,
        )
        df_weekly_orders["week_id"] = pd.to_datetime(df_weekly_orders["week_id"])
        # Pre-group into dict for fast lookup
        weekly_orders_dict = {cid: grp for cid, grp in df_weekly_orders.groupby("choice_id")}
    except Exception:
        weekly_orders_dict = {}

    return df, dict_choice_prophet, dict_detected_peaks, dict_detected_troughs, weekly_orders_dict


def plot_seasonality(
    choice_id: str,
    dict_choice_prophet: dict,
    dict_detected_peaks: dict,
    dict_detected_troughs: dict,
    weekly_orders_dict: dict,
):
    """Generate the seasonality analysis plot for a given choice."""
    fig, ax = plt.subplots(figsize=(10, 4))

    # Plot actual orders from weekly orders data (O(1) dict lookup)
    if choice_id in weekly_orders_dict:
        series_data = weekly_orders_dict[choice_id]
        ax.plot(series_data["week_id"], series_data["num_orders"], color="black", label="Actual Orders")

    # Get Prophet forecast data if available
    if choice_id in dict_choice_prophet:
        forecast_df = dict_choice_prophet[choice_id]
        ds = pd.to_datetime(forecast_df["ds"])
        ax.plot(ds, forecast_df["yhat"], color="gray", alpha=0.5, linestyle=":", label="Prophet Forecast")

    # Add vertical lines for detected peaks (Red) and troughs (Blue)
    peaks = dict_detected_peaks.get(choice_id, [])
    troughs = dict_detected_troughs.get(choice_id, [])

    # Convert to list if needed (handles numpy arrays, pandas Series, etc.)
    if hasattr(peaks, "tolist"):
        peaks = peaks.tolist()
    if hasattr(troughs, "tolist"):
        troughs = troughs.tolist()

    for i, x in enumerate(peaks):
        label = "Detected Peaks" if i == 0 else "_nolegend_"
        ax.axvline(x=x, color="red", linestyle="--", alpha=0.4, label=label)

    for i, x in enumerate(troughs):
        label = "Detected Troughs" if i == 0 else "_nolegend_"
        ax.axvline(x=x, color="blue", linestyle="--", alpha=0.4, label=label)

    ax.set_title(f"Seasonality Analysis: {choice_id}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Orders")
    plt.xticks(rotation=45)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.1)
    plt.tight_layout()

    return fig


def main():
    """Main app function."""
    st.title("🧥 Garment Seasonality Viewer")
    st.markdown("Enter a Choice ID to view the seasonality analysis and model predictions.")

    # Load data (cached - only runs once)
    try:
        df, dict_choice_prophet, dict_detected_peaks, dict_detected_troughs, weekly_orders_dict = load_data()
    except FileNotFoundError as e:
        st.error(f"Data files not found. Make sure the data/ folder contains the required files. Error: {e}")
        return

    if not weekly_orders_dict:
        st.warning("Weekly orders data not found. Save df_weekly_choice_orders.csv to see actual orders.")

    # Input for choice ID
    choice_id = st.text_input("Enter Choice ID:", placeholder="e.g., 90195835_001")

    if choice_id:
        # Fast O(1) lookup using index
        if choice_id not in df.index:
            st.warning(f"Choice ID '{choice_id}' not found in the dataset.")
            return

        row = df.loc[choice_id]

        # Display model results
        st.subheader("Model Results")
        col1, col2, col3 = st.columns(3)

        with col1:
            actual_season = row.get("Season", None)
            st.metric("Actual Season", actual_season if pd.notna(actual_season) else "N/A")

        with col2:
            model_season = row.get("Final Model Season", "N/A")
            is_transitional = row.get("transitional", False)
            if is_transitional is True or is_transitional == 1 or str(is_transitional).lower() == "true":
                model_season = f"{model_season} (Transitional)"
            st.metric("Model Prediction", model_season)

        with col3:
            confidence = row.get("Final Model Confidence", 0.0)
            if pd.isna(confidence):
                st.metric("Confidence", "N/A")
            else:
                st.metric("Confidence", f"{confidence:.1%}")

        # Additional details
        with st.expander("View Model Details"):
            final_score = row.get("Final Model Score", None)
            visual_score = row.get("Visual Model Score", None)
            history_score = row.get("Order History Model Score", None)
            mean_orders = row.get("Mean Orders Per Week", None)

            def fmt_score(val):
                return f"{val:.2f}" if pd.notna(val) else "N/A"

            st.write(f"**Final Model Score:** {fmt_score(final_score)}")
            st.write(f"**Visual Model Score:** {fmt_score(visual_score)}")
            st.write(f"**Order History Model Score:** {fmt_score(history_score)}")
            class_name = row.get("Class Name", None)
            st.write(f"**Class Name:** {class_name if pd.notna(class_name) else 'N/A'}")
            st.write(f"**Mean Orders Per Week:** {fmt_score(mean_orders)}")

            st.divider()
            with st.expander("How to read scores"):
                st.markdown(
                    """
                    - Scores range from **-1** to **+1**
                    - **Positive** → Warm Weather | **Negative** → Cold Weather | **Near zero** → No Season
                    - If the final score is between -0.4 and +0.4, it's classified as **No Season**
                    """
                )
            with st.expander("How confidence works"):
                st.markdown(
                    """
                    - If Warm or Cold season is assigned, higher absolute scores indicate higher confidence.
                    - If No Season is assigned, confidence is based on how close the score is to zero (lower absolute value = higher confidence).
                    - Confidence is raised when both models agree on the classification.
                    - Confidence is lowered when models disagree or only one model has a score.
                    """
                )

        # Display the plot
        st.subheader("Order History & Seasonality")
        fig = plot_seasonality(
            choice_id,
            dict_choice_prophet,
            dict_detected_peaks,
            dict_detected_troughs,
            weekly_orders_dict,
        )
        st.pyplot(fig)
        plt.close(fig)

        # Display the product image
        st.subheader("Product Image")
        image_url = row.get("Image URL")
        if pd.notna(image_url):
            st.image(image_url, width=300)
        else:
            st.info("No image available for this product.")


if __name__ == "__main__":
    main()
