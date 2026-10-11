# Signal — Real-Time Financial Transaction Fraud Detection

A Streamlit capstone prototype for scoring anonymized card transactions with a Random Forest, replaying a simulated transaction stream, and reviewing fraud alerts.

## Run the trained dashboard

Python 3.12 is recommended. From this folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The trained model, evaluation metrics, and a synthetic stream sample are included. The original CSV and individual source rows are not duplicated in this folder. The Random Forest uses bootstrap samples, randomized feature/threshold candidates, and class-balanced Gini splits in a compact NumPy implementation, so the included model can be loaded without a compiled ML extension.

## Publish on Streamlit Community Cloud

Put this folder at the root of a GitHub repository, push the project files, then create an app in [Streamlit Community Cloud](https://share.streamlit.io/) using `app.py` as the entrypoint. Select Python 3.12 in the initial deployment settings. The `requirements.txt`, `.streamlit/config.toml`, trained model, metrics, and demo sample are included. Keep the original `creditcard.csv` and archive out of the repository.

## Retrain with the supplied dataset

Extract `creditcard.csv` from the supplied `archive.zip`, then run:

```powershell
python -m pip install -r requirements.txt
python train_model.py --data "C:\path\to\creditcard.csv"
```

The training script replaces `fraud_model.pkl`, `model_metrics.json`, and `demo_stream.csv`. It uses a stratified 70/15/15 train/validation/test split. The alert threshold is chosen to maximize F1 on validation data; the test set is held out for the final metrics. The 1,200-row replay file is synthetic, generated from class-level feature distributions, and is intended only to demonstrate the interface. Use the full-dataset prevalence and held-out metrics for evaluation.

## Pages

- **Overview:** dataset size and class balance, historical activity, model quality, and the end-to-end prototype flow.
- **Live stream:** automatically replays the held-out demo sample in small batches and builds a rolling alert queue.
- **Score a transaction:** scores a sample profile or an uploaded CSV containing `Time`, `V1`–`V28`, and `Amount`.
- **Model & data:** training configuration, confusion matrix, held-out metrics, data notes, and prototype limitations.

## Scope and limitations

This local website simulates the transaction stream. It does not start or connect to Kafka or Spark; those are the distributed streaming components named in the capstone outline for a later integration stage. Dataset features `V1`–`V28` are anonymized PCA components, so the manual scoring page uses an existing feature profile and lets you adjust time and amount. Historical model metrics do not guarantee future fraud detection performance. This educational prototype is not a bank or payment-processing system.
