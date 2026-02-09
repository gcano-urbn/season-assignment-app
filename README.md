# Seasonality Viewer App

A Streamlit app for stakeholders to view seasonality analysis for garments.

## Setup

1. Connect this repository to Streamlit Cloud. 

2. Configure GCP credentials in Streamlit Cloud secrets (see Deployment section below).

3. Deploy the app from Streamlit Cloud. 

## Configuration and Deployment Details

1. Once the repo is linked, add a secret in Streamlit Cloud dashboard under **Settings** → **Secrets**:

```toml
[gcp_user_credentials]
type = "authorized_user"
client_id = "client-id.apps.googleusercontent.com"
client_secret = "client-secret"
refresh_token = "refresh-token"
quota_project_id = "rental-ds"
```

To get these credentials, run `cat ~/.config/gcloud/application_default_credentials.json` after authenticating with `gcloud auth application-default login`. This secret allows for securely accessing season data saved in GCS.

2. Deploy the app from Streamlit Cloud

## App Usage

1. Enter a Choice ID in the text box
2. View the model prediction, confidence, and actual season
3. Expand "View Model Details" for score breakdowns
4. See the order history chart with Prophet forecast
5. View the product image

## How Scoring Works

- **Scores range from -1 to +1**
  - Positive → Warm Weather
  - Negative → Cold Weather  
  - Near zero → No Season
- If the final score is between -0.4 and +0.4, it's classified as **No Season**
- Confidence is higher when both visual and order history models agree
