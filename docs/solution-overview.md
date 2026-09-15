# Solution Overview

## What We Built

We built an integrated AI platform for **predicting selective MMP-12 inhibitors and evaluating their structural and binding properties in silico**. The platform combines chemical-space analysis with three complementary prediction stages.

First, the **MSR-ARN (Multi-Scale Residual Attentive Regression Network)** learns patterns from molecular representations to predict MMP-12 inhibitory activity and identify compounds with greater predicted selectivity toward MMP-12 over related MMP isoforms. The model uses multi-scale convolution, residual learning, channel attention, dilated processing, self-attention, and feature modulation to capture complex molecular relationships.

Second, the platform predicts a **ligand-conditioned 3D protein structure**, allowing the MMP-12 conformation to adapt to the ligand rather than treating the protein as a fixed structure. This stage combines transformer-based pairwise representations, ligand cross-attention, diffusion-style structure generation, and SE(3)-equivariant coordinate refinement.

Finally, the platform performs **ligand-aware docking and binding-affinity prediction** using an SE(3)-equivariant graph network that refines the ligand pose and estimates binding energy from the resulting protein-ligand geometry.

The result is a unified workflow that moves from **molecular activity and selectivity → ligand-aware protein structure → binding pose and affinity → final candidate ranking**.

## How It Works

1. **Molecular input:** The user provides a ligand SMILES, molecular library, and the MMP-12 protein sequence.
2. **Chemical processing:** Molecular structures are validated and converted into machine-readable representations such as ECFP4 fingerprints.
3. **Activity prediction:** MSR-ARN predicts the inhibitory activity of each molecule against MMP-12.
4. **Selectivity prediction:** Candidate molecules are evaluated for predicted selectivity against related isoforms such as MMP-2 and MMP-13.
5. **Candidate prioritization:** Molecules with favorable predicted activity and selectivity are selected for structure-aware analysis.
6. **3D structure prediction:** The diffusion-transformer module combines the MMP-12 sequence with ligand information and generates a ligand-conditioned 3D protein structure.
7. **Pose refinement:** The SE(3)-equivariant docking model refines the ligand position and molecular geometry within the predicted binding site.
8. **Affinity prediction:** The refined protein-ligand complex is passed to the affinity head to predict binding energy.
9. **Final ranking:** Activity, selectivity, structural compatibility, and binding affinity are combined to rank candidate MMP-12 inhibitors.
10. **Interpretation:** SHAP analysis and 3D visualization provide molecular and structural explanations for the predictions.

## Architecture Diagram

> See [`architecture.md`](architecture.md) for the detailed diagram.

```text
[User]
   ↓
[React Frontend]
   ↓
[FastAPI Backend]
   ↓
[Chemical Processing]
   ↓
[MSR-ARN]
   ├──→ [MMP-12 Activity]
   └──→ [MMP-12 Selectivity]
              ↓
     [Candidate Prioritization]
              ↓
[Diffusion-Transformer Folding]
              ↓
[Ligand-Conditioned 3D MMP-12]
              ↓
[SE(3)-Equivariant Docking]
              ↓
[Ligand-Aware Affinity Score]
              ↓
      [Final Ranking]
         ↙       ↘
    [SHAP]    [3D Visualization]
```

## Key Design Decisions

| Decision                                           | Rationale                                                                                                                                                                                    |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Use MSR-ARN for molecular prediction               | Its multi-scale residual and attention-based design is specifically intended to capture complex nonlinear relationships in molecular feature representations for MMP-12 activity prediction. |
| Add ligand-conditioned 3D structure prediction     | Protein conformation is modelled together with ligand information so that the predicted structure can reflect ligand-dependent binding-site geometry.                                        |
| Use SE(3)-equivariant docking and affinity scoring | The model can refine ligand coordinates while respecting three-dimensional geometric transformations and simultaneously estimate binding energy.                                             |
| Combine the three stages into one pipeline         | Activity/selectivity prediction identifies promising molecules, while structure and affinity analysis provides complementary structural evidence for prioritization.                         |
| Include explainability                             | SHAP identifies molecular features that contribute to activity and selectivity predictions, improving interpretability of the AI outputs.                                                    |

## IBM Technologies Used

No IBM-specific AI service is required by the **core scientific methodology** described in the two reference papers. The proposed solution is implemented around Python-based scientific and deep-learning technologies, including TensorFlow/Keras, PyTorch, RDKit, DeepChem, Scikit-learn, SHAP, and SE(3)-equivariant neural components. The reference implementation describes Python and TensorFlow/Keras as part of its software environment.

For the hackathon submission, any IBM platform technology provided by the event should therefore be described according to its **actual implementation role in the deployed prototype** rather than claiming that IBM services were used inside the scientific models when they were not.
