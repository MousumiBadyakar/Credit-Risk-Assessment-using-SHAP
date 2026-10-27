# Credit Risk Assessment using SHAP

An explainable machine-learning project that estimates the probability of loan default from applicant and loan information. The project combines a tuned XGBoost-based classification pipeline, probability calibration, a saved decision threshold, a FastAPI prediction API, and a responsive web interface called **Credit Ledger**.

> **Disclaimer:** This is an educational/demo project. Its predictions are model estimates, not financial advice or a lending decision. It has not been established as suitable for real-world credit underwriting.


## Live Demo

- **[Application](https://credit-risk-assessment-using-shap-mxmh.onrender.com/)**

## Project Highlights

- **Credit default prediction:** Returns a predicted default probability for a loan application.
- **Threshold-based classification:** Converts the predicted probability into a `High Risk` or `Low Risk` label using the threshold loaded from `best_threshold.pkl`.
- **Calibrated model:** The notebook trains a sigmoid-calibrated version of the tuned XGBoost pipeline and saves it as `credit_risk_model.pkl`.
- **Explainable ML exploration:** The notebook uses SHAP for global feature-importance analysis and a local waterfall explanation for an individual test example.
- **Prediction API:** FastAPI validates application inputs and returns a JSON response.
- **Interactive web interface:** Credit Ledger lets users enter applicant, loan, and credit-history information and view the probability, threshold, and risk result.
- **Reproducible deployment setup:** Includes pinned dependencies, a Python version file, and a Render configuration.

## Tech Stack

| Area | Tools |
|---|---|
| Language | Python, HTML, CSS, JavaScript |
| Machine learning | scikit-learn, XGBoost |
| Explainability | SHAP (analysis in the Jupyter notebook) |
| API | FastAPI, Pydantic, Uvicorn |
| Data processing | pandas |
| Model persistence | joblib, pickle (`.pkl` artifacts) |
| Deployment | Render |

## How It Works

1. **Applicant input:** The web interface collects applicant demographics, income, employment length, loan details, and credit-history information.
2. **Request validation:** FastAPI validates the submitted JSON against the `LoanApplication` schema.
3. **Probability estimation:** The saved model loaded from `credit_risk_model.pkl` calculates the probability for class `1` (loan default).
4. **Threshold comparison:** The API compares that probability with the threshold loaded from `best_threshold.pkl`.
5. **Risk result:** If the probability is greater than or equal to the threshold, the application is labelled `High Risk`; otherwise it is labelled `Low Risk`.
6. **Result display:** The frontend displays the estimated probability, decision threshold, and result.

The notebook also explores SHAP explanations to investigate how input features influence model output. **SHAP plots are part of the notebook analysis; the current FastAPI endpoint does not return per-applicant SHAP explanations.**

## Model Development

The notebook `Credit_Risk_Assessment_using_SHAP.ipynb` documents the modelling workflow, including:

- Separating features from the `loan_status` target.
- A stratified train/test split (`test_size=0.2`, `random_state=42`).
- Preprocessing pipelines for numerical and categorical features, including imputation and one-hot encoding.
- Comparing Logistic Regression with XGBoost using cross-validation.
- Hyperparameter search for XGBoost using `RandomizedSearchCV` and average precision as the search score.
- Probability calibration with `CalibratedClassifierCV(method="sigmoid", cv=5)`.
- Threshold exploration using a precision-recall curve and F1 score.
- Global and local model interpretation using SHAP.
- Saving the calibrated model and decision threshold as `.pkl` files.

### Reported test-set results

The notebook reports the following results for the tuned XGBoost model in its model-comparison section, before the later calibration and threshold-selection steps:

| Metric | Result |
|---|---:|
| Accuracy | 0.93 |
| Precision (default class) | 0.86 |
| Recall (default class) | 0.80 |
| F1-score (default class) | 0.83 |
| Test samples | 6,305 |

These are notebook-reported results for that evaluation step. They should not be interpreted as a separate evaluation of the final deployed, calibrated model at the saved threshold. Re-evaluate the final artifacts at the exact threshold used by the app before presenting those metrics as deployed-model performance.

## Dataset

The repository includes `credit_risk_dataset.csv`, and the notebook uses `loan_status` as the target column. The features used by the application include:

- `person_age`
- `person_income`
- `person_home_ownership`
- `person_emp_length`
- `loan_intent`
- `loan_grade`
- `loan_amnt`
- `loan_int_rate`
- `loan_percent_income`
- `cb_person_default_on_file`
- `cb_person_cred_hist_length`

Use the dataset in accordance with its original licence and terms. Do not commit private or personally identifiable applicant information.

## Repository Structure

```text
Credit-Risk-Assessment-using-SHAP/
├── main.py
├── requirements.txt
├── .python-version
├── render.yaml
├── runtime.txt
├── credit_risk_model.pkl
├── best_threshold.pkl
├── credit_risk_dataset.csv
├── Credit_Risk_Assessment_using_SHAP.ipynb
└── static/
    ├── index.html
    ├── style.css
    └── script.js
```

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/MousumiBadyakar/Credit-Risk-Assessment-using-SHAP.git
cd Credit-Risk-Assessment-using-SHAP
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the API and web app

```bash
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000` in your browser. Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

The model files `credit_risk_model.pkl` and `best_threshold.pkl` must be present in the project root. The saved model should be loaded with compatible versions of its training dependencies; the versions in `requirements.txt` are the deployment environment's pinned dependencies.

## API Reference

### `POST /predict`

Accepts a JSON object containing the 11 application fields listed below.

#### Example request

```json
{
  "person_age": 30,
  "person_income": 600000,
  "person_home_ownership": "RENT",
  "person_emp_length": 5,
  "loan_intent": "PERSONAL",
  "loan_grade": "B",
  "loan_amnt": 100000,
  "loan_int_rate": 11.5,
  "loan_percent_income": 0.17,
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 6
}
```

#### Example response shape

```json
{
  "default_probability": 0.23,
  "default_prediction": 0,
  "threshold": 0.51,
  "Result": "Low Risk"
}
```

The numbers above are illustrative only; actual output depends on the saved model, threshold, and submitted values. `default_prediction` is `1` when the estimated probability meets or exceeds the threshold and `0` otherwise.

## Deploy on Render

The repository includes `render.yaml` with the following configuration:

- **Runtime:** Python
- **Build command:** `pip install -r requirements.txt`
- **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Python version:** specified by `.python-version`

To deploy:

1. Push the repository to GitHub.
2. In Render, create a new **Web Service** and connect the repository, or use the included Blueprint configuration if preferred.
3. Confirm the build and start commands match `render.yaml`.
4. Deploy and review the logs for successful dependency installation and application startup.
5. Open the Render URL and submit a test application. You can also visit `/docs` to test the API.

Keep the model and threshold artifacts in the repository or provide another reliable way for the service to access them at startup. Never commit API keys, credentials, or private applicant data.

## Limitations and Responsible Use

- The result is a model estimate, not an approval or rejection recommendation by a regulated lender.
- Model performance can vary across populations and over time; evaluate bias, calibration, drift, and decision thresholds before any real-world use.
- A threshold changes the balance between false positives and false negatives. Select it based on the project's evaluation criteria and validate it on data that was not used to choose it.
- SHAP explanations describe model behaviour; they do not establish causation or guarantee that a prediction is correct.
- The current `/predict` API returns probability, threshold, binary prediction, and risk label, but does not expose SHAP explanations in the live interface.

## Future Improvements

- Add per-application SHAP explanations to the API and UI.
- Add automated tests for API validation, model loading, and prediction responses.
- Add monitoring for input drift and prediction distributions.
- Evaluate calibration and classification metrics using the exact deployed artifacts and saved threshold.
- Add versioning for model artifacts and documented dataset provenance.

## Author

**Mousumi Badyakar**