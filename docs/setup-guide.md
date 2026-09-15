# Setup Guide

> **This file is read by the automated evaluation pipeline. Be precise and complete.**

## Prerequisites

Before you begin, ensure you have the following installed:

* [ ] Python 3.10+
* [ ] Node.js 18+
* [ ] npm 9+
* [ ] Git
* [ ] NVIDIA GPU with CUDA support recommended for deep learning inference/training
* [ ] PostgreSQL 14+ (or a PostgreSQL-compatible deployment)
* [ ] Optional: Docker and Docker Compose for containerized deployment

## Environment Variables

Copy `.env.example` to `.env` and fill in the values:

```bash
cp .env.example .env
```

| Variable       | Description                                              | Required |
| -------------- | -------------------------------------------------------- | -------- |
| `DATABASE_URL` | PostgreSQL connection string                             | Yes      |
| `MODEL_DIR`    | Directory containing trained AI models and model weights | Yes      |
| `DATA_DIR`     | Directory containing molecular and protein datasets      | Yes      |
| `DEVICE`       | Compute device such as `cuda` or `cpu`                   | Yes      |
| `API_HOST`     | FastAPI host address                                     | No       |
| `API_PORT`     | FastAPI port                                             | No       |
| `CORS_ORIGINS` | Allowed frontend origins                                 | No       |

Example:

```env
DATABASE_URL=postgresql://username:password@localhost:5432/mmp12_ai
MODEL_DIR=./models
DATA_DIR=./data
DEVICE=cuda
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=http://localhost:3000
```

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/AbhayNath001/bob-ai-hackathon-Kakarot.git
cd bob-ai-hackathon-Kakarot

# 2. Create and activate a Python virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

# 3. Upgrade pip
python -m pip install --upgrade pip

# 4. Install backend and AI/ML dependencies
pip install -r requirements.txt

# 5. Install frontend dependencies
cd frontend
npm install
cd ..

# 6. Configure environment variables
cp .env.example .env
```

### Core AI/ML Dependencies

The backend environment should provide the packages required by the proposed pipeline, including:

```text
numpy
pandas
scikit-learn
tensorflow
torch
rdkit
deepchem
networkx
shap
fastapi
uvicorn
psycopg2-binary
python-dotenv
```

The exact versions should follow `requirements.txt` used by the repository.

## Database Setup

Create the PostgreSQL database before starting the backend:

```bash
createdb mmp12_ai
```

Or create it from the PostgreSQL shell:

```sql
CREATE DATABASE mmp12_ai;
```

Then configure:

```env
DATABASE_URL=postgresql://username:password@localhost:5432/mmp12_ai
```

If the project provides database initialization or migration scripts, execute them before starting the API.

## Model and Data Setup

Place trained model weights and preprocessing artifacts in the configured model directory:

```text
models/
├── msr_arn/
├── folding/
├── docking/
└── affinity/
```

Place required datasets and molecular inputs under:

```text
data/
├── mmp12/
├── selectivity/
├── proteins/
└── ligands/
```

The molecular-processing stage uses RDKit/DeepChem representations, while the MSR-ARN model operates on molecular feature representations for MMP-12 activity prediction. The structure module accepts protein sequence and ligand information for ligand-conditioned 3D prediction, followed by SE(3)-equivariant docking and affinity prediction.

## Running the Application

Start the FastAPI backend:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Start the React frontend in a separate terminal:

```bash
cd frontend
npm run dev
```

The application will normally be available at:

```text
Frontend: http://localhost:3000
Backend API: http://localhost:8000
API Documentation: http://localhost:8000/docs
```

Use the actual port defined in the repository configuration when different.

## Running the Prediction Pipeline

For a molecular prediction workflow, provide the ligand SMILES and MMP-12 protein sequence through the frontend or API.

The expected processing sequence is:

```text
SMILES / Molecular Library
        ↓
RDKit / DeepChem Preprocessing
        ↓
MSR-ARN Activity Prediction
        ↓
MMP-12 Selectivity Prediction
        ↓
Candidate Prioritization
        ↓
Ligand-Conditioned Diffusion-Transformer Folding
        ↓
Predicted 3D MMP-12 Structure
        ↓
SE(3)-Equivariant Docking
        ↓
Ligand-Aware Binding Affinity
        ↓
Final Candidate Ranking
```

The folding module integrates ligand information through cross-attention and uses iterative SE(3)-equivariant coordinate refinement, while the docking branch performs iterative pose refinement and affinity prediction.

## Running Tests

Run backend tests with:

```bash
pytest tests/ -v
```

Run frontend tests, when configured:

```bash
cd frontend
npm test
```

For model validation, evaluate the individual modules independently before running the complete pipeline.

## Quick Demo

A minimal API health check can be performed using:

```bash
curl http://localhost:8000/health
```

Then open the frontend:

```text
http://localhost:3000
```

A typical demonstration should show:

```text
Molecular Input
      ↓
MMP-12 Activity
      ↓
Selectivity
      ↓
Predicted 3D Structure
      ↓
Docked Ligand Pose
      ↓
Binding Affinity
      ↓
Final Ranked Candidate
```

## Troubleshooting

| Issue                             | Solution                                                                                                           |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `ModuleNotFoundError`             | Activate `.venv` and run `pip install -r requirements.txt` again.                                                  |
| `RDKit` installation failure      | Use a supported Python environment or install RDKit through Conda if required by the target platform.              |
| `CUDA is not available`           | Set `DEVICE=cpu` for testing or install a CUDA-compatible PyTorch/TensorFlow environment.                          |
| FastAPI does not start            | Verify `app.main:app`, installed dependencies, `.env`, and the configured port.                                    |
| PostgreSQL connection refused     | Ensure PostgreSQL is running and verify `DATABASE_URL`.                                                            |
| Model weights not found           | Check `MODEL_DIR` and confirm all required trained weights are present.                                            |
| Frontend cannot reach backend     | Verify that FastAPI is running and that `CORS_ORIGINS` matches the frontend URL.                                   |
| Prediction input rejected         | Verify that the SMILES string is chemically valid and that the protein sequence contains valid amino-acid symbols. |
| Out-of-memory error on GPU        | Reduce batch size, use a smaller inference workload, or run the prediction stage on a larger GPU.                  |
| Slow structure/affinity inference | Use GPU acceleration and process candidate molecules in batches where supported.                                   |
