"""
AUTOMATED LLM LEAKAGE DETECTION EXPERIMENT
============================================
Sends 20 code snippets × 2 prompt types × 3 LLMs = 120 queries.
Saves all responses to a structured CSV for manual grading.

SETUP:
1. pip install openai anthropic google-generativeai
2. Set your API keys as environment variables:
   export OPENAI_API_KEY="sk-..."
   export ANTHROPIC_API_KEY="sk-ant-..."
   export GEMINI_API_KEY="AI..."
   
   Or paste them directly in the API_KEYS dict below (less secure).

3. Run: python run_experiment.py

OUTPUT:
- results/all_responses.csv — one row per query with full response text
- results/grading_sheet.csv — same rows but with empty columns for you to grade
- results/raw_responses/ — individual JSON files per query (backup)
"""

import os
import json
import csv
import time
from datetime import datetime
from pathlib import Path

# ============================================================
# API KEYS — set via environment variables or paste here
# ============================================================
API_KEYS = {
    "openai": os.environ.get("OPENAI_API_KEY", ""),
    "anthropic": os.environ.get("ANTHROPIC_API_KEY", ""),
    "gemini": os.environ.get("GEMINI_API_KEY", ""),
}

# MODEL CONFIGURATION
MODELS = {
    "openai_flagship": {
        "name": "GPT-4o",
        "model_id": "gpt-4o",
        "provider": "openai",  
    },
    "openai_mini": {
        "name": "GPT-4o-mini",
        "model_id": "gpt-4o-mini",
        "provider": "openai",  
    },
    "anthropic_flagship": {
        "name": "Claude 4.6 Sonnet",
        "model_id": "claude-sonnet-4-6",          # <-- Fixed using your screenshot
        "provider": "anthropic", 
    },
    "anthropic_mini": {
        "name": "Claude 4.5 Haiku",
        "model_id": "claude-haiku-4-5-20251001",  # <-- Fixed using your screenshot
        "provider": "anthropic", 
    },
    "gemini": {
        "name": "Gemini 2.5 Flash",
        "model_id": "gemini-2.5-flash",
        "provider": "gemini",   
    },
}
# ============================================================
# PROMPTS
# ============================================================
PROMPT_GENERIC = """Review the following Python code for any issues, bugs, or bad practices. Identify all problems you find and explain each one.

```python
{code}
```"""

PROMPT_SPECIFIC = """Review the following Python machine learning pipeline code. Specifically check whether this code contains any form of data leakage — that is, whether information from the test set could be inadvertently used during training, or whether any features encode information about the target that would not be available at prediction time.

If you find data leakage, explain exactly where it occurs and why it is a problem. If the code is free of data leakage, state that clearly.

```python
{code}
```"""

# ============================================================
# CODE SNIPPETS — all 20 snippets
# ============================================================
SNIPPETS = {
    "1A": {
        "leakage_type": "preprocessing",
        "has_leakage": True,
        "difficulty": "easy",
        "code": '''import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv("housing.csv")
X = df.drop("price", axis=1)
y = df["price"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")'''
    },
    "1B": {
        "leakage_type": "preprocessing_clean",
        "has_leakage": False,
        "difficulty": "n/a",
        "code": '''import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv("housing.csv")
X = df.drop("price", axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")'''
    },
    "2A": {
        "leakage_type": "preprocessing",
        "has_leakage": True,
        "difficulty": "easy",
        "code": '''import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import r2_score

df = pd.read_csv("energy.csv")
X = df[["temperature", "humidity", "wind_speed", "pressure"]]
y = df["energy_consumption"]

scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y, test_size=0.25, random_state=7
)

model = SVR(kernel="rbf")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")'''
    },
    "2B": {
        "leakage_type": "preprocessing_clean",
        "has_leakage": False,
        "difficulty": "n/a",
        "code": '''import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import r2_score

df = pd.read_csv("energy.csv")
X = df[["temperature", "humidity", "wind_speed", "pressure"]]
y = df["energy_consumption"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=7
)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVR(kernel="rbf")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")'''
    },
    "3A": {
        "leakage_type": "feature_selection",
        "has_leakage": True,
        "difficulty": "easy-medium",
        "code": '''import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("medical.csv")
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")'''
    },
    "3B": {
        "leakage_type": "feature_selection_clean",
        "has_leakage": False,
        "difficulty": "n/a",
        "code": '''import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("medical.csv")
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

selector = SelectKBest(score_func=f_classif, k=10)
X_train = selector.fit_transform(X_train, y_train)
X_test = selector.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")'''
    },
    "4A": {
        "leakage_type": "imputation",
        "has_leakage": True,
        "difficulty": "easy-medium",
        "code": '''import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("patients.csv")
X = df.drop("readmitted", axis=1)
y = df["readmitted"]

X["age"].fillna(X["age"].mean(), inplace=True)
X["blood_pressure"].fillna(X["blood_pressure"].median(), inplace=True)
X["cholesterol"].fillna(X["cholesterol"].mean(), inplace=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")'''
    },
    "4B": {
        "leakage_type": "imputation_clean",
        "has_leakage": False,
        "difficulty": "n/a",
        "code": '''import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("patients.csv")
X = df.drop("readmitted", axis=1)
y = df["readmitted"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train_age_mean = X_train["age"].mean()
train_bp_median = X_train["blood_pressure"].median()
train_chol_mean = X_train["cholesterol"].mean()

X_train["age"].fillna(train_age_mean, inplace=True)
X_train["blood_pressure"].fillna(train_bp_median, inplace=True)
X_train["cholesterol"].fillna(train_chol_mean, inplace=True)

X_test["age"].fillna(train_age_mean, inplace=True)
X_test["blood_pressure"].fillna(train_bp_median, inplace=True)
X_test["cholesterol"].fillna(train_chol_mean, inplace=True)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")'''
    },
    "5A": {
        "leakage_type": "imputation",
        "has_leakage": True,
        "difficulty": "easy-medium",
        "code": '''import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

df = pd.read_csv("credit.csv")
X = df.drop("default", axis=1)
y = df["default"]

imputer = SimpleImputer(strategy="mean")
X_imputed = pd.DataFrame(
    imputer.fit_transform(X), columns=X.columns
)

X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.3, random_state=10
)

model = DecisionTreeClassifier(random_state=10)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")'''
    },
    "5B": {
        "leakage_type": "imputation_clean",
        "has_leakage": False,
        "difficulty": "n/a",
        "code": '''import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

df = pd.read_csv("credit.csv")
X = df.drop("default", axis=1)
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=10
)

imputer = SimpleImputer(strategy="mean")
X_train = pd.DataFrame(
    imputer.fit_transform(X_train), columns=X.columns
)
X_test = pd.DataFrame(
    imputer.transform(X_test), columns=X.columns
)

model = DecisionTreeClassifier(random_state=10)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")'''
    },
    "6A": {
        "leakage_type": "cv_preprocessing",
        "has_leakage": True,
        "difficulty": "medium",
        "code": '''import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge

df = pd.read_csv("cars.csv")
X = df.drop("price", axis=1)
y = df["price"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = Ridge(alpha=1.0)
scores = cross_val_score(model, X_scaled, y, cv=5, scoring="r2")
print(f"Mean R2: {scores.mean():.4f} (+/- {scores.std():.4f})")'''
    },
    "6B": {
        "leakage_type": "cv_preprocessing_clean",
        "has_leakage": False,
        "difficulty": "n/a",
        "code": '''import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge

df = pd.read_csv("cars.csv")
X = df.drop("price", axis=1)
y = df["price"]

model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
scores = cross_val_score(model, X, y, cv=5, scoring="r2")
print(f"Mean R2: {scores.mean():.4f} (+/- {scores.std():.4f})")'''
    },
    "7A": {
        "leakage_type": "oversampling",
        "has_leakage": True,
        "difficulty": "medium-hard",
        "code": '''import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

df = pd.read_csv("fraud.csv")
X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))'''
    },
    "7B": {
        "leakage_type": "oversampling_clean",
        "has_leakage": False,
        "difficulty": "n/a",
        "code": '''import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

df = pd.read_csv("fraud.csv")
X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))'''
    },
    "8A": {
        "leakage_type": "temporal",
        "has_leakage": True,
        "difficulty": "medium-hard",
        "code": '''import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("stock_prices.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

features = ["open", "high", "low", "volume", "moving_avg_7"]
X = df[features]
y = df["close"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")'''
    },
    "8B": {
        "leakage_type": "temporal_clean",
        "has_leakage": False,
        "difficulty": "n/a",
        "code": '''import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("stock_prices.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

features = ["open", "high", "low", "volume", "moving_avg_7"]
X = df[features]
y = df["close"]

split_index = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")'''
    },
    "9A": {
        "leakage_type": "target_leakage",
        "has_leakage": True,
        "difficulty": "hard",
        "code": '''import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("loans.csv")

features = [
    "income", "credit_score", "loan_amount",
    "employment_years", "collection_recovery_fee"
]
X = df[features]
y = df["defaulted"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")'''
    },
    "9B": {
        "leakage_type": "target_leakage_clean",
        "has_leakage": False,
        "difficulty": "n/a",
        "code": '''import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("loans.csv")

features = [
    "income", "credit_score", "loan_amount",
    "employment_years"
]
X = df[features]
y = df["defaulted"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")'''
    },
    "10A": {
        "leakage_type": "target_leakage",
        "has_leakage": True,
        "difficulty": "hard",
        "code": '''import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("hospital.csv")

features = [
    "age", "bmi", "blood_pressure", "num_symptoms",
    "insurance_claim_amount", "discharge_type"
]
X = df[features]
y = df["readmitted_within_30_days"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")'''
    },
    "10B": {
        "leakage_type": "target_leakage_clean",
        "has_leakage": False,
        "difficulty": "n/a",
        "code": '''import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("hospital.csv")

features = [
    "age", "bmi", "blood_pressure", "num_symptoms"
]
X = df[features]
y = df["readmitted_within_30_days"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")'''
    },
}

# ============================================================
# API CALL FUNCTIONS
# ============================================================

def call_openai(prompt, model_id):
    """Call OpenAI API (GPT-4o-mini)"""
    from openai import OpenAI
    client = OpenAI(api_key=API_KEYS["openai"])
    
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,  # default
        max_tokens=2000,
    )
    return response.choices[0].message.content


def call_anthropic(prompt, model_id):
    """Call Anthropic API (Claude)"""
    import anthropic
    client = anthropic.Anthropic(api_key=API_KEYS["anthropic"])
    
    response = client.messages.create(
        model=model_id,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def call_gemini(prompt, model_id):
    """New 2026 Google GenAI API Call"""
    from google import genai
    client = genai.Client(api_key=API_KEYS["gemini"])
    response = client.models.generate_content(
        model=model_id,
        contents=prompt
    )
    return response.text


API_CALLERS = {
    "openai": call_openai,
    "anthropic": call_anthropic,
    "gemini": call_gemini,
}

# ============================================================
# CONFIGURATION
# ============================================================
NUM_RUNS = 3  # Number of times to repeat the full experiment for validity


# ============================================================
# MAIN EXPERIMENT RUNNER
# ============================================================

def run_experiment(run_number):
    """Run all 120 queries for a single run and save results."""

    run_dir = f"results/run_{run_number}"
    Path(f"{run_dir}/raw_responses").mkdir(parents=True, exist_ok=True)

    prompts = {
        "generic": PROMPT_GENERIC,
        "specific": PROMPT_SPECIFIC,
    }

    all_results = []
    total = len(SNIPPETS) * len(prompts) * len(MODELS)
    current = 0

    print(f"\n{'='*60}")
    print(f"STARTING RUN {run_number} of {NUM_RUNS}  ({total} queries)")
    print(f"{'='*60}")

    for snippet_id, snippet_data in SNIPPETS.items():
        for prompt_type, prompt_template in prompts.items():

            # Build the full prompt with the code inserted
            full_prompt = prompt_template.format(code=snippet_data["code"])

            for provider, model_config in MODELS.items():
                current += 1
                print(f"  Run {run_number} [{current}/{total}] Snippet {snippet_id} | "
                      f"{prompt_type} | {model_config['name']}...", end=" ")

                try:
                    provider_key = model_config["provider"]
                    caller = API_CALLERS[provider_key]
                    response_text = caller(full_prompt, model_config["model_id"])
                    status = "success"
                    print("OK")
                except Exception as e:
                    response_text = f"ERROR: {str(e)}"
                    status = "error"
                    print(f"FAILED: {e}")

                # Build result record
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

                # Save individual response as JSON backup
                filename = f"{snippet_id}_{prompt_type}_{provider}.json"
                with open(f"{run_dir}/raw_responses/{filename}", "w") as f:
                    json.dump(result, f, indent=2)

               # Rate limiting — only sleep long for Gemini's free tier
                if model_config["provider"] == "gemini":
                    time.sleep(4.1)  # 60 seconds / 15 requests = 4 seconds 
                else:
                    time.sleep(0.5)  # A tiny half-second pause is plenty for OpenAI/Anthropic
    # --------------------------------------------------------
    # Save this run's CSVs
    # --------------------------------------------------------
    csv_fields = [
        "run_number", "snippet_id", "leakage_type", "has_leakage_ground_truth",
        "difficulty", "prompt_type", "provider", "model_name",
        "model_id", "status", "response", "timestamp"
    ]

    with open(f"{run_dir}/responses.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n  Run {run_number} complete — saved to {run_dir}/responses.csv")
    print(f"  Successful: {sum(1 for r in all_results if r['status'] == 'success')} / {total}")

    return all_results


def combine_runs(all_runs_results):
    """Combine all runs into one CSV and produce a grading sheet."""

    Path("results").mkdir(parents=True, exist_ok=True)

    csv_fields = [
        "run_number", "snippet_id", "leakage_type", "has_leakage_ground_truth",
        "difficulty", "prompt_type", "provider", "model_name",
        "model_id", "status", "response", "timestamp"
    ]

    with open("results/all_runs_combined.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(all_runs_results)

    grading_fields = [
        "run_number", "snippet_id", "leakage_type", "has_leakage_ground_truth",
        "difficulty", "prompt_type", "model_name",
        "llm_detected_leakage", "detection_result",
        "explanation_quality", "notes"
    ]

    with open("results/grading_sheet.csv", "w", newline="", encoding="utf-8") as f:
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
                "llm_detected_leakage": "",   # YOU FILL THIS
                "detection_result": "",        # TP/TN/FP/FN
                "explanation_quality": "",     # Correct/Partial/Incorrect/Hallucinated
                "notes": "",                   # Your observations
            })

    total = len(all_runs_results)
    print(f"\n{'='*60}")
    print(f"ALL {NUM_RUNS} RUNS COMPLETE")
    print(f"{'='*60}")
    print(f"Total queries across all runs: {total}")
    print(f"Successful: {sum(1 for r in all_runs_results if r['status'] == 'success')}")
    print(f"Failed:     {sum(1 for r in all_runs_results if r['status'] == 'error')}")
    print(f"\nOutput files:")
    print(f"  results/run_1/responses.csv    — run 1 responses")
    print(f"  results/run_2/responses.csv    — run 2 responses")
    print(f"  results/run_3/responses.csv    — run 3 responses")
    print(f"  results/all_runs_combined.csv  — all {total} rows combined")
    print(f"  results/grading_sheet.csv      — empty grading columns (for you)")
    print(f"\nNEXT STEP: Open grading_sheet.csv and fill in the 3 empty columns")
    print(f"by reading responses in all_runs_combined.csv")


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    # Check for API keys
    missing = [k for k, v in API_KEYS.items() if not v]
    if missing:
        print(f"WARNING: Missing API keys for: {', '.join(missing)}")
        print("Set them as environment variables or paste into the script.")
        print("Example: export OPENAI_API_KEY='sk-...'")
        print()

        # Ask if user wants to continue with available APIs only
        available = [k for k, v in API_KEYS.items() if v]
        if not available:
            print("No API keys found. Exiting.")
            exit(1)

        print(f"Available APIs: {', '.join(available)}")
        resp = input("Continue with available APIs only? (y/n): ")
        if resp.lower() != "y":
            exit(0)

        # Remove unavailable models
        for m in missing:
            del MODELS[m]

    # Run the experiment NUM_RUNS times and collect all results
    all_runs_results = []
    for run_num in range(1, NUM_RUNS + 1):
        run_results = run_experiment(run_num)
        all_runs_results.extend(run_results)

    # Combine all runs into final output files
    combine_runs(all_runs_results)