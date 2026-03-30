"""
LLM Data Leakage Detection - Experiment Runner
Author: Aditya Raghunandan
Date: March 2026

This script evaluates multiple LLMs on their ability to detect data leakage 
in machine learning pipelines. It queries OpenAI, Anthropic, and Google APIs
using both generic and targeted prompts, saving the results for manual grading.
"""

import os
import json
import csv
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Configuration & API Setup ---

API_KEYS = {
    "openai": os.environ.get("OPENAI_API_KEY", ""),
    "anthropic": os.environ.get("ANTHROPIC_API_KEY", ""),
    "gemini": os.environ.get("GEMINI_API_KEY", ""),
}

MODELS = {
    "openai_flagship": {"name": "GPT-4o", "model_id": "gpt-4o", "provider": "openai"},
    "openai_mini": {"name": "GPT-4o-mini", "model_id": "gpt-4o-mini", "provider": "openai"},
    "anthropic_flagship": {"name": "Claude 4.6 Sonnet", "model_id": "claude-sonnet-4-6", "provider": "anthropic"},
    "anthropic_mini": {"name": "Claude 4.5 Haiku", "model_id": "claude-haiku-4-5-20251001", "provider": "anthropic"},
    "gemini": {"name": "Gemini 2.5 Flash", "model_id": "gemini-2.5-flash", "provider": "gemini"},
}

PROMPTS = {
    "generic": (
        "Review the following Python code for any issues, bugs, or bad practices. "
        "Identify all problems you find and explain each one.\n\n```python\n{code}\n```"
    ),
    "specific": (
        "Review the following Python machine learning pipeline code. Specifically check whether "
        "this code contains any form of data leakage — that is, whether information from the test "
        "set could be inadvertently used during training, or whether any features encode information "
        "about the target that would not be available at prediction time.\n\n"
        "If you find data leakage, explain exactly where it occurs and why it is a problem. "
        "If the code is free of data leakage, state that clearly.\n\n```python\n{code}\n```"
    )
}

# --- Dataset ---
# Snippets are mapped by ID. 10 Leaky (A), 10 Clean (B).
SNIPPETS = {
    "1A": {
        "leakage_type": "preprocessing",
        "has_leakage": True,
        "difficulty": "easy",
        "code": "import pandas as pd\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import LinearRegression\nfrom sklearn.metrics import mean_squared_error\n\ndf = pd.read_csv('housing.csv')\nX = df.drop('price', axis=1)\ny = df['price']\n\nscaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)\n\nX_train, X_test, y_train, y_test = train_test_split(\n    X_scaled, y, test_size=0.2, random_state=42\n)\n\nmodel = LinearRegression()\nmodel.fit(X_train, y_train)\ny_pred = model.predict(X_test)\nprint(f'MSE: {mean_squared_error(y_test, y_pred):.4f}')"
    },
    "1B": {
        "leakage_type": "preprocessing_clean",
        "has_leakage": False,
        "difficulty": "n/a",
        "code": "import pandas as pd\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import LinearRegression\nfrom sklearn.metrics import mean_squared_error\n\ndf = pd.read_csv('housing.csv')\nX = df.drop('price', axis=1)\ny = df['price']\n\nX_train, X_test, y_train, y_test = train_test_split(\n    X, y, test_size=0.2, random_state=42\n)\n\nscaler = StandardScaler()\nX_train = scaler.fit_transform(X_train)\nX_test = scaler.transform(X_test)\n\nmodel = LinearRegression()\nmodel.fit(X_train, y_train)\ny_pred = model.predict(X_test)\nprint(f'MSE: {mean_squared_error(y_test, y_pred):.4f}')"
    },
    "2A": {
        "leakage_type": "preprocessing",
        "has_leakage": True,
        "difficulty": "easy",
        "code": "import pandas as pd\nfrom sklearn.preprocessing import MinMaxScaler\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.svm import SVR\nfrom sklearn.metrics import r2_score\n\ndf = pd.read_csv('energy.csv')\nX = df[['temperature', 'humidity', 'wind_speed', 'pressure']]\ny = df['energy_consumption']\n\nscaler = MinMaxScaler()\nX_normalized = scaler.fit_transform(X)\n\nX_train, X_test, y_train, y_test = train_test_split(\n    X_normalized, y, test_size=0.25, random_state=7\n)\n\nmodel = SVR(kernel='rbf')\nmodel.fit(X_train, y_train)\ny_pred = model.predict(X_test)\nprint(f'R2 Score: {r2_score(y_test, y_pred):.4f}')"
    },
    "2B": {
        "leakage_type": "preprocessing_clean",
        "has_leakage": False,
        "difficulty": "n/a",
        "code": "import pandas as pd\nfrom sklearn.preprocessing import MinMaxScaler\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.svm import SVR\nfrom sklearn.metrics import r2_score\n\ndf = pd.read_csv('energy.csv')\nX = df[['temperature', 'humidity', 'wind_speed', 'pressure']]\ny = df['energy_consumption']\n\nX_train, X_test, y_train, y_test = train_test_split(\n    X, y, test_size=0.25, random_state=7\n)\n\nscaler = MinMaxScaler()\nX_train = scaler.fit_transform(X_train)\nX_test = scaler.transform(X_test)\n\nmodel = SVR(kernel='rbf')\nmodel.fit(X_train, y_train)\ny_pred = model.predict(X_test)\nprint(f'R2 Score: {r2_score(y_test, y_pred):.4f}')"
    },
    "3A": {
        "leakage_type": "feature_selection",
        "has_leakage": True,
        "difficulty": "easy-medium",
        "code": "import pandas as pd\nfrom sklearn.feature_selection import SelectKBest, f_classif\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.metrics import accuracy_score\n\ndf = pd.read_csv('medical.csv')\nX = df.drop('diagnosis', axis=1)\ny = df['diagnosis']\n\nselector = SelectKBest(score_func=f_classif, k=10)\nX_selected = selector.fit_transform(X, y)\n\nX_train, X_test, y_train, y_test = train_test_split(\n    X_selected, y, test_size=0.2, random_state=42\n)\n\nmodel = RandomForestClassifier(n_estimators=100, random_state=42)\nmodel.fit(X_train, y_train)\ny_pred = model.predict(X_test)\nprint(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')"
    },
    "3B": {
        "leakage_type": "feature_selection_clean",
        "has_leakage": False,
        "difficulty": "n/a",
        "code": "import pandas as pd\nfrom sklearn.feature_selection import SelectKBest, f_classif\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.metrics import accuracy_score\n\ndf = pd.read_csv('medical.csv')\nX = df.drop('diagnosis', axis=1)\ny = df['diagnosis']\n\nX_train, X_test, y_train, y_test = train_test_split(\n    X, y, test_size=0.2, random_state=42\n)\n\nselector = SelectKBest(score_func=f_classif, k=10)\nX_train = selector.fit_transform(X_train, y_train)\nX_test = selector.transform(X_test)\n\nmodel = RandomForestClassifier(n_estimators=100, random_state=42)\nmodel.fit(X_train, y_train)\ny_pred = model.predict(X_test)\nprint(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')"
    },
    "4A": {
        "leakage_type": "imputation",
        "has_leakage": True,
        "difficulty": "easy-medium",
        "code": "import pandas as pd\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import accuracy_score\n\ndf = pd.read_csv('patients.csv')\nX = df.drop('readmitted', axis=1)\ny = df['readmitted']\n\nX['age'].fillna(X['age'].mean(), inplace=True)\nX['blood_pressure'].fillna(X['blood_pressure'].median(), inplace=True)\nX['cholesterol'].fillna(X['cholesterol'].mean(), inplace=True)\n\nX_train, X_test, y_train, y_test = train_test_split(\n    X, y, test_size=0.2, random_state=42\n)\n\nmodel = LogisticRegression(max_iter=1000)\nmodel.fit(X_train, y_train)\ny_pred = model.predict(X_test)\nprint(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')"
    },
    "4B": {
        "leakage_type": "imputation_clean",
        "has_leakage": False,
        "difficulty": "n/a",
        "code": "import pandas as pd\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import accuracy_score\n\ndf = pd.read_csv('patients.csv')\nX = df.drop('readmitted', axis=1)\ny = df['readmitted']\n\nX_train, X_test, y_train, y_test = train_test_split(\n    X, y, test_size=0.2, random_state=42\n)\n\ntrain_age_mean = X_train['age'].mean()\ntrain_bp_median = X_train['blood_pressure'].median()\ntrain_chol_mean = X_train['cholesterol'].mean()\n\nX_train['age'].fillna(train_age_mean, inplace=True)\nX_train['blood_pressure'].fillna(train_bp_median, inplace=True)\nX_train['cholesterol'].fillna(train_chol_mean, inplace=True)\n\nX_test['age'].fillna(train_age_mean, inplace=True)\nX_test['blood_pressure'].fillna(train_bp_median, inplace=True)\nX_test['cholesterol'].fillna(train_chol_mean, inplace=True)\n\nmodel = LogisticRegression(max_iter=1000)\nmodel.fit(X_train, y_train)\ny_pred = model.predict(X_test)\nprint(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')"
    },
    "5A": {
        "leakage_type": "imputation",
        "has_leakage": True,
        "difficulty": "easy-medium",
        "code": "import pandas as pd\nfrom sklearn.impute import SimpleImputer\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.metrics import f1_score\n\ndf = pd.read_csv('credit.csv')\nX = df.drop('default', axis=1)\ny = df['default']\n\nimputer = SimpleImputer(strategy='mean')\nX_imputed = pd.DataFrame(\n    imputer.fit_transform(X), columns=X.columns\n)\n\nX_train, X_test, y_train, y_test = train_test_split(\n    X_imputed, y, test_size=0.3, random_state=10\n)\n\nmodel = DecisionTreeClassifier(random_state=10)\nmodel.fit(X_train, y_train)\ny_pred = model.predict(X_test)\nprint(f'F1 Score: {f1_score(y_test, y_pred):.4f}')"
    },
    "5B": {
        "leakage_type": "imputation_clean",
        "has_leakage": False,
        "difficulty": "n/a",
        "code": "import pandas as pd\nfrom sklearn.impute import SimpleImputer\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.metrics import f1_score\n\ndf = pd.read_csv('credit.csv')\nX = df.drop('default', axis=1)\ny = df['default']\n\nX_train, X_test, y_train, y_test = train_test_split(\n    X, y, test_size=0.3, random_state=10\n)\n\nimputer = SimpleImputer(strategy='mean')\nX_train = pd.DataFrame(\n    imputer.fit_transform(X_train), columns=X.columns\n)\nX_test = pd.DataFrame(\n    imputer.transform(X_test), columns=X.columns\n)\n\nmodel = DecisionTreeClassifier(random_state=10)\nmodel.fit(X_train, y_train)\ny_pred = model.predict(X_test)\nprint(f'F1 Score: {f1_score(y_test, y_pred):.4f}')"
    },
    "6A": {
        "leakage_type": "cv_preprocessing",
        "has_leakage": True,
        "difficulty": "medium",
        "code": "import pandas as pd\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.model_selection import cross_val_score\nfrom sklearn.linear_model import Ridge\n\ndf = pd.read_csv('cars.csv')\nX = df.drop('price', axis=1)\ny = df['price']\n\nscaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)\n\nmodel = Ridge(alpha=1.0)\nscores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')\nprint(f'Mean R2: {scores.mean():.4f} (+/- {scores.std():.4f})')"
    },
    "6B": {
        "leakage_type": "cv_preprocessing_clean",
        "has_leakage": False,
        "difficulty": "n/a",
        "code": "import pandas as pd\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.model_selection import cross_val_score\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.linear_model import Ridge\n\ndf = pd.read_csv('cars.csv')\nX = df.drop('price', axis=1)\ny = df['price']\n\nmodel = make_pipeline(StandardScaler(), Ridge(alpha=1.0))\nscores = cross_val_score(model, X, y, cv=5, scoring='r2')\nprint(f'Mean R2: {scores.mean():.4f} (+/- {scores.std():.4f})')"
    },
    "7A": {
        "leakage_type": "oversampling",
        "has_leakage": True,
        "difficulty": "medium-hard",
        "code": "import pandas as pd\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.ensemble import GradientBoostingClassifier\nfrom sklearn.metrics import classification_report\nfrom imblearn.over_sampling import SMOTE\n\ndf = pd.read_csv('fraud.csv')\nX = df.drop('is_fraud', axis=1)\ny = df['is_fraud']\n\nsmote = SMOTE(random_state=42)\nX_resampled, y_resampled = smote.fit_resample(X, y)\n\nX_train, X_test, y_train, y_test = train_test_split(\n    X_resampled, y_resampled, test_size=0.2, random_state=42\n)\n\nmodel = GradientBoostingClassifier(n_estimators=100, random_state=42)\nmodel.fit(X_train, y_train)\ny_pred = model.predict(X_test)\nprint(classification_report(y_test, y_pred))"
    },
    "7B": {
        "leakage_type": "oversampling_clean",
        "has_leakage": False,
        "difficulty": "n/a",
        "code": "import pandas as pd\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.ensemble import GradientBoostingClassifier\nfrom sklearn.metrics import classification_report\nfrom imblearn.over_sampling import SMOTE\n\ndf = pd.read_csv('fraud.csv')\nX = df.drop('is_fraud', axis=1)\ny = df['is_fraud']\n\nX_train, X_test, y_train, y_test = train_test_split(\n    X, y, test_size=0.2, random_state=42\n)\n\nsmote = SMOTE(random_state=42)\nX_train, y_train = smote.fit_resample(X_train, y_train)\n\nmodel = GradientBoostingClassifier(n_estimators=100, random_state=42)\nmodel.fit(X_train, y_train)\ny_pred = model.predict(X_test)\nprint(classification_report(y_test, y_pred))"
    },
    "8A": {
        "leakage_type": "temporal",
        "has_leakage": True,
        "difficulty": "medium-hard",
        "code": "import pandas as pd\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.ensemble import RandomForestRegressor\nfrom sklearn.metrics import mean_absolute_error\n\ndf = pd.read_csv('stock_prices.csv')\ndf['date'] = pd.to_datetime(df['date'])\ndf = df.sort_values('date')\n\nfeatures = ['open', 'high', 'low', 'volume', 'moving_avg_7']\nX = df[features]\ny = df['close']\n\nX_train, X_test, y_train, y_test = train_test_split(\n    X, y, test_size=0.2, random_state=42\n)\n\nmodel = RandomForestRegressor(n_estimators=100, random_state=42)\nmodel.fit(X_train, y_train)\ny_pred = model.predict(X_test)\nprint(f'MAE: {mean_absolute_error(y_test, y_pred):.4f}')"
    },
    "8B": {
        "leakage_type": "temporal_clean",
        "has_leakage": False,
        "difficulty": "n/a",
        "code": "import pandas as pd\nfrom sklearn.ensemble import RandomForestRegressor\nfrom sklearn.metrics import mean_absolute_error\n\ndf = pd.read_csv('stock_prices.csv')\ndf['date'] = pd.to_datetime(df['date'])\ndf = df.sort_values('date')\n\nfeatures = ['open', 'high', 'low', 'volume', 'moving_avg_7']\nX = df[features]\ny = df['close']\n\nsplit_index = int(len(df) * 0.8)\nX_train, X_test = X.iloc[:split_index], X.iloc[split_index:]\ny_train, y_test = y.iloc[:split_index], y.iloc[split_index:]\n\nmodel = RandomForestRegressor(n_estimators=100, random_state=42)\nmodel.fit(X_train, y_train)\ny_pred = model.predict(X_test)\nprint(f'MAE: {mean_absolute_error(y_test, y_pred):.4f}')"
    },
    "9A": {
        "leakage_type": "target_leakage",
        "has_leakage": True,
        "difficulty": "hard",
        "code": "import pandas as pd\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import accuracy_score\n\ndf = pd.read_csv('loans.csv')\n\nfeatures = [\n    'income', 'credit_score', 'loan_amount',\n    'employment_years', 'collection_recovery_fee'\n]\nX = df[features]\ny = df['defaulted']\n\nX_train, X_test, y_train, y_test = train_test_split(\n    X, y, test_size=0.2, random_state=42\n)\n\nmodel = LogisticRegression(max_iter=1000)\nmodel.fit(X_train, y_train)\ny_pred = model.predict(X_test)\nprint(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')"
    },
    "9B": {
        "leakage_type": "target_leakage_clean",
        "has_leakage": False,
        "difficulty": "n/a",
        "code": "import pandas as pd\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import accuracy_score\n\ndf = pd.read_csv('loans.csv')\n\nfeatures = [\n    'income', 'credit_score', 'loan_amount',\n    'employment_years'\n]\nX = df[features]\ny = df['defaulted']\n\nX_train, X_test, y_train, y_test = train_test_split(\n    X, y, test_size=0.2, random_state=42\n)\n\nmodel = LogisticRegression(max_iter=1000)\nmodel.fit(X_train, y_train)\ny_pred = model.predict(X_test)\nprint(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')"
    },
    "10A": {
        "leakage_type": "target_leakage",
        "has_leakage": True,
        "difficulty": "hard",
        "code": "import pandas as pd\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.ensemble import GradientBoostingClassifier\nfrom sklearn.metrics import accuracy_score\n\ndf = pd.read_csv('hospital.csv')\n\nfeatures = [\n    'age', 'bmi', 'blood_pressure', 'num_symptoms',\n    'insurance_claim_amount', 'discharge_type'\n]\nX = df[features]\ny = df['readmitted_within_30_days']\n\nX_train, X_test, y_train, y_test = train_test_split(\n    X, y, test_size=0.2, random_state=42\n)\n\nmodel = GradientBoostingClassifier(n_estimators=100, random_state=42)\nmodel.fit(X_train, y_train)\ny_pred = model.predict(X_test)\nprint(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')"
    },
    "10B": {
        "leakage_type": "target_leakage_clean",
        "has_leakage": False,
        "difficulty": "n/a",
        "code": "import pandas as pd\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.ensemble import GradientBoostingClassifier\nfrom sklearn.metrics import accuracy_score\n\ndf = pd.read_csv('hospital.csv')\n\nfeatures = [\n    'age', 'bmi', 'blood_pressure', 'num_symptoms'\n]\nX = df[features]\ny = df['readmitted_within_30_days']\n\nX_train, X_test, y_train, y_test = train_test_split(\n    X, y, test_size=0.2, random_state=42\n)\n\nmodel = GradientBoostingClassifier(n_estimators=100, random_state=42)\nmodel.fit(X_train, y_train)\ny_pred = model.predict(X_test)\nprint(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')"
    },
}

# --- Initialization of API Clients ---
# We initialize them globally to avoid recreating clients on every loop iteration.

openai_client = None
if API_KEYS["openai"]:
    from openai import OpenAI
    openai_client = OpenAI(api_key=API_KEYS["openai"])

anthropic_client = None
if API_KEYS["anthropic"]:
    import anthropic
    anthropic_client = anthropic.Anthropic(api_key=API_KEYS["anthropic"])

gemini_client = None
if API_KEYS["gemini"]:
    from google import genai
    gemini_client = genai.Client(api_key=API_KEYS["gemini"])


def call_openai(prompt: str, model_id: str) -> str:
    """Send prompt to OpenAI models."""
    response = openai_client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
        max_tokens=2000,
    )
    return response.choices[0].message.content


def call_anthropic(prompt: str, model_id: str) -> str:
    """Send prompt to Anthropic Claude models."""
    response = anthropic_client.messages.create(
        model=model_id,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def call_gemini(prompt: str, model_id: str) -> str:
    """Send prompt to Google Gemini models."""
    response = gemini_client.models.generate_content(
        model=model_id,
        contents=prompt
    )
    return response.text


API_CALLERS = {
    "openai": call_openai,
    "anthropic": call_anthropic,
    "gemini": call_gemini,
}


def run_experiment(run_number: int) -> List[Dict[str, Any]]:
    """
    Executes all permutations of snippets, prompts, and models for a single run.
    """
    run_dir = Path(f"results/run_{run_number}")
    (run_dir / "raw_responses").mkdir(parents=True, exist_ok=True)

    all_results = []
    total_queries = len(SNIPPETS) * len(PROMPTS) * len(MODELS)
    current_query = 0

    logger.info(f"Starting Run {run_number} ({total_queries} queries total)")

    for snippet_id, snippet_data in SNIPPETS.items():
        for prompt_type, prompt_template in PROMPTS.items():
            full_prompt = prompt_template.format(code=snippet_data["code"])

            for provider, model_config in MODELS.items():
                current_query += 1
                logger.info(f"[Run {run_number} | {current_query}/{total_queries}] Querying {model_config['name']} for {snippet_id} ({prompt_type})...")

                try:
                    provider_key = model_config["provider"]
                    caller = API_CALLERS[provider_key]
                    response_text = caller(full_prompt, model_config["model_id"])
                    status = "success"
                except Exception as e:
                    response_text = f"ERROR: {str(e)}"
                    status = "error"
                    logger.error(f"Failed query on {model_config['name']}: {e}")

                result = {
                    "run_number": run_number,
                    "snippet_id": snippet_id,
                    "leakage_type": snippet_data["leakage_type"],
                    "has_leakage_ground_truth": snippet_data["has_leakage"],
                    "difficulty": snippet_data["difficulty"],
                    "prompt_type": prompt_type,
                    "provider": provider,
                    "model_name": model_config["name"],
                    "model_id": model_config["model_id"],
                    "status": status,
                    "response": response_text,
                    "timestamp": datetime.now().isoformat(),
                }
                all_results.append(result)

                # Save raw JSON backup
                filename = f"{snippet_id}_{prompt_type}_{provider}.json"
                with open(run_dir / "raw_responses" / filename, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2)

                # Rate limiting logic
                if model_config["provider"] == "gemini":
                    time.sleep(4.1) 
                else:
                    time.sleep(0.5) 

    # Save CSV for this specific run
    csv_fields = list(all_results[0].keys())
    with open(run_dir / "responses.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(all_results)

    success_count = sum(1 for r in all_results if r['status'] == 'success')
    logger.info(f"Run {run_number} complete. {success_count}/{total_queries} succeeded.")

    return all_results


def combine_runs(all_runs_results: List[Dict[str, Any]]):
    """
    Merges data from all experimental runs and generates a grading template.
    """
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Output merged log
    if all_runs_results:
        csv_fields = list(all_runs_results[0].keys())
        with open(results_dir / "all_runs_combined.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            writer.writerows(all_runs_results)

    # Generate manual grading sheet
    grading_fields = [
        "run_number", "snippet_id", "leakage_type", "has_leakage_ground_truth",
        "difficulty", "prompt_type", "model_name",
        "llm_detected_leakage", "detection_result",
        "explanation_quality", "notes"
    ]

    with open(results_dir / "grading_sheet.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=grading_fields)
        writer.writeheader()
        for r in all_runs_results:
            writer.writerow({
                "run_number": r["run_number"],
                "snippet_id": r["snippet_id"],
                "leakage_type": r["leakage_type"],
                "has_leakage_ground_truth": r["has_leakage_ground_truth"],
                "difficulty": r["difficulty"],
                "prompt_type": r["prompt_type"],
                "model_name": r["model_name"],
                "llm_detected_leakage": "",   
                "detection_result": "",        
                "explanation_quality": "",     
                "notes": "",                   
            })

    logger.info("Generated all_runs_combined.csv and grading_sheet.csv")


def main():
    parser = argparse.ArgumentParser(description="Run LLM Data Leakage Detection Experiment")
    parser.add_argument("--runs", type=int, default=3, help="Number of experimental runs (default: 3)")
    args = parser.parse_args()

    # Pre-flight API check
    missing_keys = [k for k, v in API_KEYS.items() if not v]
    if missing_keys:
        logger.warning(f"Missing API keys for: {', '.join(missing_keys)}")
        logger.warning("Make sure to set them as environment variables (e.g., export OPENAI_API_KEY='...')")
        
        available = [k for k, v in API_KEYS.items() if v]
        if not available:
            logger.error("No API keys found. Exiting.")
            exit(1)
            
        logger.info(f"Continuing with available APIs only: {', '.join(available)}")
        for m in missing_keys:
            if m in MODELS:
                del MODELS[m]

    all_runs_results = []
    for run_num in range(1, args.runs + 1):
        results = run_experiment(run_num)
        all_runs_results.extend(results)

    combine_runs(all_runs_results)
    logger.info("Experiment finished successfully.")


if __name__ == "__main__":
    main()