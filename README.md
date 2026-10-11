# Signal: Card Transaction Fraud Monitoring

Signal is a capstone research application for exploring fraud classification on anonymized credit card transaction features. It loads a checksum-pinned 45-tree Random Forest, reports held-out test metrics, and offers a clearly labeled synthetic stream simulation. CSV scoring is fail-closed: it stays unavailable until an administrator configures OIDC and an explicit email allowlist in the deployment secrets.

## Run locally

Python 3.12 is recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

## Retrain the model

Extract `creditcard.csv` from the supplied archive, then run:

```powershell
python -m pip install -r requirements.txt
python train_model.py --data "C:\path\to\creditcard.csv"
```

Training uses a stratified 70/15/15 train, validation, and test split. The alert threshold is chosen to maximize validation F1. Test metrics are calculated on the held-out test split. The included 1,200-row stream file is synthetic sample data generated for interface demonstration. It is not live or historical customer activity. The original dataset is not included in this repository.

## Pages

- **Home:** source dataset summary, historical activity, held-out model measures, and task links.
- **Live monitor:** process synthetic profiles in user-triggered batches and inspect the session review queue.
- **Analyze transactions:** after an authorized OIDC sign-in, score a one-row or batch CSV with exactly Time, V1 through V28, and Amount fields; download results. The file limit is 5 MB and 20,000 rows.
- **Model evaluation:** confusion matrix, test metrics, experiment configuration, and limitations.
- **Activity history:** summaries from actions in the active browser session only.
- **About, Contact, Privacy policy, Terms and conditions:** project scope, source links, data handling, and use conditions.

## Data handling and limitations

Scoring is disabled until OIDC is configured and the logged-in email is included in the deployment's allowed email list. Uploaded CSVs must contain exactly the required feature schema, must be no larger than 5 MB or 20,000 rows, and must not contain direct identifiers or labels. The application has no transaction database and does not write uploads to the repository. Files and results may be held in active session memory for display. Streamlit Community Cloud may process technical information to host the app. Access control does not make the app suitable for confidential, regulated, or operational payment data.

The stream monitor is a user-triggered simulation. This deployment does not connect to Kafka, Spark, a bank, a payment processor, or a live event source. V1 through V28 are anonymized PCA features. Results are educational model estimates, not financial decisions or guarantees of future detection.

## Publish

This repository is deployed on [Streamlit Community Cloud](https://capstone-project-1-maav8mrafi6mbadybs8xci.streamlit.app/). Select app.py as the entrypoint if you create a separate deployment. Keep the original transaction dataset and any confidential records out of the repository.

## Secure scoring setup

File scoring deliberately fails closed until an administrator configures Streamlit's OIDC authentication. In the Streamlit Community Cloud app settings, add secrets using the OIDC values from your identity provider and allow only the account email(s) that should be able to upload files:

```toml
[auth]
redirect_uri = "https://capstone-project-1-maav8mrafi6mbadybs8xci.streamlit.app/oauth2callback"
cookie_secret = "REPLACE_WITH_A_LONG_RANDOM_SECRET"
client_id = "REPLACE_WITH_IDP_CLIENT_ID"
client_secret = "REPLACE_WITH_IDP_CLIENT_SECRET"
server_metadata_url = "REPLACE_WITH_IDP_OIDC_DISCOVERY_URL"

[access]
allowed_emails = ["your-authorized-account@example.com"]
```

Register the exact `redirect_uri` with the identity provider. Never put real credentials in GitHub or this file. After saving secrets and restarting the app, verify that an allowlisted account can sign in, a non-allowlisted account is denied, and CSV scoring works only for the allowlisted account. Without these secrets, the public app remains in read-only demo mode and file scoring is unavailable.

