![CI Status]( https://github.com/Christian-Suinyuy/skye8-ds-banboye-christian/actions/workflows/ci.yml/badge.svg)

# skye8-ds-banboye-christian
Personalized development roadmap for the Skye8 Data Science / ML - AI Internship. Focused on building AI Engineer skills, including version control, software engineering, SQL, machine learning, and LLM-powered systems.

## Install

This project requires Python 3.11 or newer. From the project directory, create a virtual environment and install the package:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Quickstart

The cost calculator can be used without a database or csv.

Run the tests from the project root with:

```powershell
python -m pytest
```

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
