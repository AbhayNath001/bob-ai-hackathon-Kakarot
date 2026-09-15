import pandas as pd
import numpy as np
import math
from rdkit import Chem
from rdkit.Chem import PandasTools
from mordred import Calculator, descriptors
from mordred.error import Error as MordredError

def load_and_clean_sdf(sdf_path):
    """Loads SDF, finds SMILES and Target (pIC50), and returns cleaned DataFrame."""
    try:
        df = PandasTools.LoadSDF(sdf_path)
    except Exception:
        # Fallback using SDMolSupplier if PandasTools fails
        suppl = Chem.SDMolSupplier(sdf_path)
        rows = []
        for mol in suppl:
            if mol is not None:
                props = dict(mol.GetPropsAsDict())
                props['SMILES'] = Chem.MolToSmiles(mol)
                rows.append(props)
        df = pd.DataFrame(rows)

    # Normalize columns
    df.columns = [c.strip() for c in df.columns]
    
    # Ensure SMILES
    if 'SMILES' not in df.columns:
        # Simple heuristic to find SMILES column
        candidates = [c for c in df.columns if 'smiles' in c.lower()]
        if candidates:
            df['SMILES'] = df[candidates[0]]
        else:
            raise ValueError("SMILES column not found in SDF.")

    # Find Target
    target_col = _find_target_column(df)
    
    # Drop missing
    df = df.dropna(subset=['SMILES', target_col]).reset_index(drop=True)
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    
    return df, target_col

def _find_target_column(df):
    """Internal helper to identify pIC50 or Activity columns."""
    cols_lower = {c.lower(): c for c in df.columns}
    
    # Priority 1: Direct pIC50
    for lower, orig in cols_lower.items():
        if ('p' in lower and 'log' in lower) or 'pic50' in lower or 'pactivity' in lower:
            return orig
            
    # Priority 2: Convert nM Activity
    for lower, orig in cols_lower.items():
        if 'activity' in lower and 'nm' in lower:
            # Clean string data
            df[orig] = pd.to_numeric(df[orig].astype(str).str.replace('[^0-9.eE+-]', '', regex=True), errors='coerce')
            # Create new pIC50 column
            new_col = 'pActivity_computed'
            df[new_col] = df[orig].apply(lambda x: 9.0 - math.log10(x) if (x > 0 and not np.isnan(x)) else np.nan)
            return new_col
            
    raise ValueError("Could not automatically identify a target column (pIC50 or Activity).")

def compute_descriptors(smiles_list):
    """Computes Mordred descriptors for a list of SMILES."""
    calc = Calculator(descriptors, ignore_3D=True)
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    
    # Filter invalid mols
    valid_mols = [m for m in mols if m is not None]
    valid_indices = [i for i, m in enumerate(mols) if m is not None]
    
    if not valid_mols:
        return pd.DataFrame(), []

    # Calculate
    raw_df = calc.pandas(valid_mols)
    
    # Clean non-numeric errors
    def clean_val(v):
        if isinstance(v, (MordredError, list, tuple, np.ndarray, str)):
            return np.nan
        return v
        
    cleaned = raw_df.applymap(clean_val)
    # Ensure float
    cleaned = cleaned.apply(pd.to_numeric, errors='coerce')
    
    # Reset index to match original valid indices
    cleaned.index = valid_indices
    return cleaned, valid_indices

def clean_feature_matrix(X_df, missing_threshold=0.3):
    """Drops sparse/zero-variance columns and imputes median."""
    # Drop high missing
    X_df = X_df.dropna(axis=1, thresh=int((1-missing_threshold)*len(X_df)))
    # Drop zero variance
    X_df = X_df.loc[:, (X_df.var() > 0)]
    # Impute remaining
    X_df = X_df.fillna(X_df.median())
    return X_df