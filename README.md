# Signal: Card Transaction Fraud Monitoring

Signal is a capstone research application for exploring fraud classification on anonymized credit card transaction features. It loads a trained 45-tree Random Forest, reports held-out test metrics, scores feature-only CSV files, and offers a clearly labeled synthetic stream simulation.

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
- **Analyze transactions:** score a one-row or batch CSV with Time, V1 through V28, and Amount fields; download results.
- **Model evaluation:** confusion matrix, test metrics, experiment configuration, and limitations.
- **Activity history:** summaries from actions in the active browser session only.
- **About, Contact, Privacy policy, Terms and conditions:** project scope, source links, data handling, and use conditions.

## Data handling and limitations

Uploaded CSVs must contain only the required feature schema. Remove names, account IDs, card numbers, and other direct identifiers before upload. The application has no account system or database and does not write uploads to the repository. Files and results may be held in active session memory for display. Streamlit Community Cloud may process technical information to host the app.

The stream monitor is a user-triggered simulation. This deployment does not connect to Kafka, Spark, a bank, a payment processor, or a live event source. V1 through V28 are anonymized PCA features. Results are educational model estimates, not financial decisions or guarantees of future detection.

## Publish

This repository is deployed on [Streamlit Community Cloud](https://capstone-project-1-maav8mrafi6mbadybs8xci.streamlit.app/). Select app.py as the entrypoint if you create a separate deployment. Keep the original transaction dataset and any confidential records out of the repository.
