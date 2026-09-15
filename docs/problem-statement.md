# Problem Statement

## Background

Matrix metalloproteinase-12 (MMP-12), also known as macrophage metalloelastase, is associated with cancer, inflammatory, cardiovascular, pulmonary, and other pathological conditions. Its therapeutic targeting is therefore of significant interest. However, selective MMP-12 inhibition is challenging because MMP family members share highly conserved catalytic domains, making it difficult to distinguish MMP-12 from related isoforms such as MMP-2 and MMP-13.

Modern AI provides an opportunity to combine molecular chemical-space analysis with deep learning and structure-based modelling. The referenced MSR-ARN framework demonstrated that multi-scale convolution, residual learning, attention, and feature modulation can learn complex molecular activity relationships from chemical representations.

## The Problem

Current computational workflows generally handle molecular activity prediction, protein structure prediction, docking, and binding-affinity estimation as separate tasks. This creates a gap between which molecules are predicted to be selective MMP-12 inhibitors and how those molecules interact with the target structure.

Our problem is to develop an integrated AI framework that predicts:

1. MMP-12 inhibitory activity and selectivity from molecular chemical-space information;
2. Ligand-aware 3D MMP-12 protein structure, where the predicted conformation adapts to the ligand; and
3. Ligand-aware binding affinity and pose, using geometric and energetic information from the predicted protein–ligand complex.

## Who is Affected

The problem primarily affects drug-discovery researchers, medicinal chemists, computational biologists, and pharmaceutical R&D teams working on MMP-12-targeted therapeutics. They require computational methods that can rapidly prioritize promising compounds before resource-intensive experimental validation.

## Why It Matters

Poor target selectivity can result in unwanted interaction with related MMP isoforms and reduce the therapeutic usefulness of candidate inhibitors. Efficient computational prioritization can reduce the number of molecules requiring downstream experimental investigation and provide structural insight for lead optimization. AI-based approaches can process large chemical datasets and learn complex molecular interaction patterns more efficiently than purely conventional workflows.

## Why Existing Solutions Fall Short

Ligand-based QSAR and ML models primarily learn relationships between molecular descriptors and biological activity, while conventional docking and scoring methods separately estimate binding poses and energies. Structure prediction can also be performed independently of the ligand. The referenced work highlights the need for integrated approaches that jointly model ligand-conditioned folding, pose refinement, and affinity scoring.

Our proposed solution combines the MSR-ARN multi-scale residual attentive architecture for chemical-space-guided MMP-12 selectivity prediction with a diffusion-transformer ligand-conditioned folding module and an SE(3)-equivariant graph-based docking and affinity module. The folding module incorporates ligand information through cross-attention and iterative SE(3)-equivariant coordinate refinement, while the docking module jointly refines ligand pose and predicts binding energy.
