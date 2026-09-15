# Source Code

The project integrates two complementary AI pipelines into a unified computational drug-discovery workflow: **MSR-ARN-based chemical-space and MMP-12 activity prediction**, and **diffusion-transformer/SE(3)-based 3D protein structure and ligand-aware binding analysis**.

## Structure Guidelines

The source code is organized according to the functional stages of the integrated AI pipeline:

```text
src/
├── data/
│   ├── preprocessing.py
│   └── feature_processing.py
│
├── models/
│   ├── msr_arn/
│   │   └── model.py
│   │
│   ├── protein_structure/
│   │   ├── attention.py
│   │   ├── pairformer.py
│   │   ├── se3.py
│   │   ├── utils.py
│   │   └── init.py
│   │
│   └── docking_affinity/
│       └── complete_sample.py
│
├── inference/
│   ├── activity_prediction.py
│   ├── selectivity_prediction.py
│   ├── structure_prediction.py
│   └── affinity_prediction.py
│
├── explainability/
│   └── shap_analysis.py
│
├── api/
│   └── main.py
│
└── notebooks/
    └── experiments/
```

### Data / AI Components

**`src/data/`** contains molecular-data loading, cleaning, descriptor generation, feature selection, scaling, and preparation for model inference.

**`src/models/msr_arn/`** contains the Multi-Scale Residual Attentive Regression Network. The implementation uses the molecular descriptor pipeline together with the trained feature configuration and scaler used by the model. The original implementation includes `preprocessing.py`, `feature_columns.pkl`, and `feature_scaler.pkl`.

**`src/models/protein_structure/`** contains the modular components required for ligand-aware protein structure prediction, including attention, pairformer, SE(3) geometric processing, and supporting utilities.

**`src/models/docking_affinity/`** contains the integrated implementation for protein-ligand modelling, including iterative geometric processing, pose refinement, and affinity prediction. The complete implementation is provided through the integrated model script.

**`src/inference/`** connects the individual models into the final prediction workflow:

```text
Molecular Input
      ↓
Molecular Preprocessing
      ↓
MSR-ARN Activity Prediction
      ↓
MMP-12 Selectivity Prediction
      ↓
Candidate Prioritization
      ↓
Ligand-Aware 3D Structure Prediction
      ↓
SE(3)-Based Docking / Pose Refinement
      ↓
Binding Affinity Prediction
      ↓
Final Candidate Ranking
```

## Important Files to Include

```text
requirements.txt
.env.example
README.md

src/
├── data/
├── models/
├── inference/
├── explainability/
└── api/

models/
├── feature_columns.pkl
├── feature_scaler.pkl
└── trained_weights/
```

The MSR-ARN implementation specifically relies on the serialized feature configuration and `StandardScaler` to ensure that inference uses the same feature representation and scaling as the trained model.

For the 3D structure component, retain the modular Python source files required by the attention, pairformer, SE(3), and utility implementations.

## What NOT to Include in `src/`

* `.env` files containing real credentials or secrets
* `venv/`, `.venv/`, or `node_modules/`
* `__pycache__/`
* Large raw molecular datasets and experimental archives
* Generated prediction outputs
* Large trained-model binaries that exceed normal Git repository limits
* Temporary notebooks, logs, checkpoints, and intermediate experiment files

Large model weights or datasets should be stored using **Git LFS or external object storage**, while the repository should retain the code, configuration, preprocessing artifacts, and documented download locations.
