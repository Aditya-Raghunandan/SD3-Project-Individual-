# How Reliably Can Large Language Models Detect Data Leakage in Machine Learning Pipelines?

**Course:** Software Development III (ELEN4010)  
**Institution:** University of the Witwatersrand, Johannesburg  
**Author:** Aditya Raghunandan  

## 📌 Project Overview
This repository contains the automation scripts, prompt datasets, and evaluation logs for my empirical investigation into whether Large Language Models (LLMs) can reliably detect methodological data leakage in Machine Learning (ML) pipelines.

The study evaluates five state-of-the-art LLMs (`GPT-4o`, `GPT-4o-mini`, `Claude 4.6 Sonnet`, `Claude 4.5 Haiku`, and `Gemini 2.5 Flash`) across 20 curated Python snippets containing various forms of structural and semantic data leakage. The experiment compares a **Generic** code-review prompt against a **Leakage-Specific** prompt to measure the "Prompt-Sensitivity Gap."

## 📂 Repository Structure

The repository is structured to separate the automation code from the generated experimental data:

```text
├── analysisScript.py               # The main Python script used to query the LLM APIs
├── README.md                       # Project documentation
└── results/                        # Directory containing all experimental outputs
    ├── all_runs_combined.csv       # The raw text responses for all 360 LLM queries
    ├── grading_sheet.csv           # The blank template generated for manual evaluation
    ├── Grading.xlsx                # The final, manually graded results and analysis
    ├── run_1/                      # Raw JSON backups and CSVs for Experiment Run 1
    ├── run_2/                      # Raw JSON backups and CSVs for Experiment Run 2
    └── run_3/                      # Raw JSON backups and CSVs for Experiment Run 3
```

## 📊 Data Collection & Prompts
The 20 Python snippets (10 buggy, 10 clean) and the exact prompts used for the experiment are hardcoded directly inside `analysisScript.py`. 

Because LLM outputs are non-deterministic, the script is designed to run the entire suite 3 times, generating 360 total data points to ensure statistical validity. The full unedited responses from the LLMs can be found in `results/all_runs_combined.csv`.

## 🚀 How to Replicate the Experiment

If you wish to re-run the API querying script yourself, you will need active API keys for OpenAI, Anthropic, and Google Gemini.

**1. Install dependencies:**
```bash
pip install openai anthropic google-generativeai
```

**2. Set Environment Variables:**
Set your API keys in your terminal environment (or paste them directly into the `API_KEYS` dictionary in the script):
```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="AI..."
```

**3. Execute the script:**
```bash
python analysisScript.py --runs 3
```
The script will automatically create a `results/` folder, manage rate limits, and output the consolidated CSVs for grading.

## 📝 License & Acknowledgements
In accordance with the Wits School of Electrical and Information Engineering LLM Report Writing Policy, an LLM was utilized to assist in the generation of the Python API automation script and formatting of the LaTeX report. The experimental design, dataset curation, manual grading, and intellectual conclusions are entirely the original work of the author.
```
