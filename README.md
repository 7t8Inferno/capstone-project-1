# Signal: Card Transaction Fraud Monitoring

Signal is a capstone research application for exploring fraud classification on anonymized credit card transaction features. It loads a checksum-pinned 45-tree Random Forest, reports held-out test metrics, and offers a clearly labeled synthetic stream simulation. CSV scoring is publicly accessible for small, feature-only test files; no login is required. The website is an educational demo and must not receive confidential or identifying payment records.

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
- **Analyze transactions:** any visitor can score a one-row or batch CSV with exactly Time, V1 through V28, and Amount fields; download results. Files are limited to 1 MB and 1,000 rows. A ready-made synthetic test CSV is downloadable in the app.
- **Model evaluation:** confusion matrix, test metrics, experiment configuration, and limitations.
- **Activity history:** summaries from actions in the active browser session only.
- **About, Contact, Privacy policy, Terms and conditions:** project scope, source links, data handling, and use conditions.

## Data handling and limitations

Public scoring does not require sign-in. Anyone with the site link can upload feature-only CSV files; a file must contain exactly the 30 expected numeric columns and must be no larger than 1 MB or 1,000 rows. Direct identifiers, fraud labels, card numbers, real customer information, and other confidential data must never be uploaded. The application does not write uploaded CSVs to the repository or a persistent transaction database; temporary data may remain in session memory and the hosting provider may handle transient uploads. This is a limited public academic demonstration, not a suitable service for confidential, regulated, or operational payment data. Do not publish your original creditcard.csv in GitHub.

The stream monitor is a user-triggered simulation. This deployment does not connect to Kafka, Spark, a bank, a payment processor, or a live event source. V1 through V28 are anonymized PCA features. Results are educational model estimates, not financial decisions or guarantees of future detection.

## Publish

This repository is deployed on [Streamlit Community Cloud](https://capstone-project-1-maav8mrafi6mbadybs8xci.streamlit.app/). Select app.py as the entrypoint if you create a separate deployment. Keep the original transaction dataset and any confidential records out of the repository.

## Public test uploads

Open [Analyze transactions](https://capstone-project-1-maav8mrafi6mbadybs8xci.streamlit.app/?view=Analyze%20transactions). No Google login, OIDC setup, or email allowlist is required. Download the bundled five-row **synthetic** example CSV from the page and upload it through **Batch file**. Click **Score batch** to see model scores and Review/Clear decisions.

To test a source-format ULB CSV, remove the `Class` label column and keep only `Time`, `V1`–`V28`, and `Amount`. Split larger files into subsets of 1,000 rows or less and 1 MB or less. No arbitrary merchant/date/account CSV can be scored without the original PCA feature transformation.

**Privacy/security:** This is a public unauthenticated upload form. The application checks CSV schema, row count, file size and finite numeric values, but cannot guarantee that submitted data is non-confidential or prevent abuse. Submit only synthetic or non-sensitive data that you have permission to share. Do not treat the scores as calibrated fraud probabilities or use them to make financial decisions.

If you later require private or larger-scale scoring, restore authentication and add server-side rate limiting, resource quotas, and monitoring before handling any non-demo data.
