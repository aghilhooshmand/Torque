#!/usr/bin/env python3
"""Export sklearn classification datasets to CSV in data/ for Evolution/Test page."""
import os
import pandas as pd
from sklearn.datasets import load_wine, load_breast_cancer

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Wine: 178 samples, 13 features, 3 classes (non-trivial)
wine = load_wine(as_frame=True)
df_wine = wine.frame
df_wine = df_wine.rename(columns={"target": "class"})
df_wine.to_csv(os.path.join(DATA_DIR, "wine.csv"), index=False)
print("Wrote", os.path.join(DATA_DIR, "wine.csv"), "| shape", df_wine.shape, "| classes", df_wine["class"].nunique())

# Breast cancer (sklearn): 569 samples, 30 features, 2 classes (harder than tiny sets)
bc = load_breast_cancer(as_frame=True)
df_bc = bc.frame
df_bc = df_bc.rename(columns={"target": "class"})
df_bc.to_csv(os.path.join(DATA_DIR, "breast_cancer_sklearn.csv"), index=False)
print("Wrote", os.path.join(DATA_DIR, "breast_cancer_sklearn.csv"), "| shape", df_bc.shape, "| classes", df_bc["class"].nunique())
