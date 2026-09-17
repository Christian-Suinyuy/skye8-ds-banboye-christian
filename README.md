![CI Status]( https://github.com/Christian-Suinyuy/skye8-ds-banboye-christian/actions/workflows/ci.yml/badge.svg)

# skye8-ds-banboye-christian
Personalized development roadmap for the Skye8 Data Science / ML - AI Internship. Focused on building AI Engineer skills, including version control, software engineering, SQL, machine learning, and LLM-powered systems.

## Environment setup

This project requires Python 3.11 or newer. From the repository root, create a virtual environment and install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run commands from the repository root so the Brief 3 dataset path resolves correctly. Use the local SQLite-backed MLflow store in the active PowerShell session:

```powershell
$env:MLFLOW_TRACKING_URI = "sqlite:///mlflow_local.db"
```

The optional hosted LLM evaluation requires a `.env` file in the repository root:

```text
GROQ_API_KEY=your-key-here
```

Do not commit `.env` or API keys.

## Quickstart

Run the tests from the project root with:

```powershell
python -m pytest
```

Train all enabled Brief 3 models with the default split:

```powershell
python -m src.brief3.train_all_models
```

Detailed instructions for running individual models and viewing MLflow runs are in [docs/training.md](docs/training.md). The split strategies are documented separately in [docs/split.md](docs/split.md).

To load the sample telemetry into PostgreSQL, create a database, run `src/agent_telimetry/sql/schema.sql`, and add a `.env` file containing your connection string:

```text
DATABASE_STRING=postgresql://username:password@localhost:5432/agent_telemetry
```

Then run the loader from the project root:

```powershell
python -m agent_telimetry.loader
```

## Layout

```text
src/agent_telimetry/
|-- costing.py              Cost calculation helpers
|-- loader.py               Load CSV data into PostgreSQL
|-- data/
|   |-- raw/                Source CSV files
|   |-- cleaned/            Cleaned telemetry data
|-- sql/schema.sql          PostgreSQL table definitions
|-- tests/                  Pytest tests
|-- notebooks/              Exploration notebook
|-- docs/                   Data-quality and reconciliation notes
```

The pricing values used by the calculator live in `costing.py`. The CSV files are kept in the repository so the data preparation and loading steps can be reproduced locally.
