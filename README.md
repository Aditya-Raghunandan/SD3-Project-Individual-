# How Reliably Can Large Language Models Detect Data Leakage in Machine Learning Pipelines?

**Course:** Software Development III (ELEN4010)  
**Institution:** University of the Witwatersrand, Johannesburg  
**Author:** Aditya Raghunandan  

## Project Overview
This repository contains the code, prompts, and results for an empirical study evaluating whether Large Language Models (LLMs) can detect data leakage in ML pipelines.

Five LLMs (`GPT-4o`, `GPT-4o-mini`, `Claude 4.6 Sonnet`, `Claude 4.5 Haiku`, `Gemini 2.5 Flash`) are tested on 20 Python snippets (10 buggy, 10 clean). The study compares a **generic code-review prompt** with a **leakage-specific prompt** to measure prompt sensitivity.

## Repository Structure
```text
├── analysisScript.py        # Script used to query LLM APIs
├── README.md
└── results/
    ├── all_runs_combined.csv   # Raw responses (600 total)
    ├── SD3_Results_Graded.xlsx # Grading, prompts, and analysis
    ├── run_1/                  # Raw outputs (Run 1)
    ├── run_2/                  # Raw outputs (Run 2)
    └── run_3/                  # Raw outputs (Run 3)
```

## Data & Analysis
- All prompts, grading, and results are consolidated in **`SD3_Results_Graded.xlsx`**
- The Excel file contains:
  - Raw grading data  
  - Analysis sheet  
  - Extended analysis sheet  

The script runs 3 times (600 total responses) to account for LLM variability. Full raw outputs are stored in `all_runs_combined.csv`.

## Replication

**1. Install dependencies**
```bash
pip install openai anthropic google-generativeai
```

**2. Set API keys**
```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="AI..."
```

**3. Run the experiment**
```bash
python analysisScript.py --runs 3
```

The script will generate the `results/` folder and all outputs automatically.
