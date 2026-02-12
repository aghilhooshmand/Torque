# Meta-Features Guide: Understanding Dataset Complexity

This guide explains the meta-features computed in the **Meta-Feature Explorer** page, how to interpret them, and which ones are most important for understanding dataset complexity.

---

## Quick Start: First Steps to Compare Complexity

**For a quick complexity assessment, start with these measures:**

1. **Basic Stats** → `n_rows`, `n_features_only`, `class_imbalance_ratio_max_min`
2. **PCA Structure** → `pca_k95_over_n_features` (how much redundancy exists)
3. **Intrinsic Dimensionality** → `id_mle` or `id_twonn` (true degrees of freedom)
4. **General (PyMFE)** → `inst_to_attr` (sample-to-feature ratio)
5. **Model-Based (PyMFE)** → `leaves`, `nodes` (decision tree complexity)

**Interpretation:**
- **Low complexity**: Small k95/n_features (< 0.5), low ID (< n_features/2), balanced classes, high inst_to_attr
- **High complexity**: Large k95/n_features (> 0.8), high ID (close to n_features), imbalanced classes, low inst_to_attr

---

## Meta-Feature Groups

### 1. Basic Stats (`basic-stats`)

**Purpose**: Simple, interpretable dataset-level statistics computed directly from the data.

**Measures:**

| Measure | Description | Calculation | Interpretation |
|---------|-------------|-------------|----------------|
| `n_rows` | Number of samples | Count of rows in dataset | More samples → more reliable models (if balanced) |
| `n_columns` | Total columns (features + target) | Count of columns | Total dimensionality |
| `n_features_only` | Number of feature columns | Columns minus target | Input dimensionality |
| `missing_total` | Total missing values | Sum of NaN/None values | Data quality indicator |
| `missing_pct` | Percentage of missing cells | `(missing_total / (rows × cols)) × 100` | > 5% may need imputation |
| `n_classes` | Number of unique target classes | Count of unique labels | Classification complexity |
| `class_imbalance_ratio_max_min` | Max class size / Min class size | `max(counts) / min(counts)` | > 10 indicates severe imbalance |
| `outliers_count_z3` | Rows with any feature |z-score| > 3 | Z-score > 3 on any numeric feature | Potential noise/errors |
| `outliers_pct_z3` | Percentage of outlier rows | `(outliers_count / n_rows) × 100` | > 5% may indicate data quality issues |

**When to use**: Always compute first. Quick sanity check for data quality and basic structure.

**References**: Standard statistical measures.

---

### 2. PCA Structure (`pca-structure`)

**Purpose**: Measures how many principal components are needed to capture most variance, indicating redundancy and intrinsic structure.

**Measures:**

| Measure | Description | Calculation | Interpretation |
|---------|-------------|-------------|----------------|
| `pca_k95` | Components needed for 95% variance | PCA on StandardScaler(X), cumulative variance ≥ 0.95 | Lower = more redundant/easier |
| `pca_k99` | Components needed for 99% variance | PCA on StandardScaler(X), cumulative variance ≥ 0.99 | More conservative estimate |
| `pca_k95_over_n_features` | k95 relative to feature count | `k95 / n_features` | **< 0.5**: Highly redundant<br>**0.5-0.8**: Moderate redundancy<br>**> 0.8**: Low redundancy (complex) |
| `pca_k99_over_n_features` | k99 relative to feature count | `k99 / n_features` | Similar to k95, more conservative |

**How it's calculated:**
```python
scaler = StandardScaler(with_mean=True, with_std=True)
X_scaled = scaler.fit_transform(X)
pca = PCA().fit(X_scaled)
cum_var = np.cumsum(pca.explained_variance_ratio_)
k95 = np.searchsorted(cum_var, 0.95) + 1
k99 = np.searchsorted(cum_var, 0.99) + 1
```

**When to use**: Quick check for redundancy. If k95 << n_features, data is low-dimensional and often easier to model.

**References**: Principal Component Analysis (PCA) - standard dimensionality reduction technique.

---

### 3. Intrinsic Dimensionality (`intrinsic-dim`)

**Purpose**: Estimates the "true degrees of freedom" in the data using geometric/manifold methods.

**Measures:**

| Measure | Description | Calculation | Interpretation |
|---------|-------------|-------------|----------------|
| `id_mle` | Maximum Likelihood Estimator | MLE method on StandardScaler(X) via scikit-dimension | Lower = simpler manifold |
| `id_twonn` | Two Nearest Neighbors estimator | TwoNN method on StandardScaler(X) via scikit-dimension | Alternative geometric estimate |

**How it's calculated:**
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
id_mle = skdim.id.MLE().fit(X_scaled).dimension_
id_twonn = skdim.id.TwoNN().fit(X_scaled).dimension_
```

**Interpretation:**
- **ID << n_features**: Data lies on a low-dimensional manifold (easier)
- **ID ≈ n_features**: Data fills the full space (harder, noisy)
- Compare ID to `pca_k95`: If ID < k95, data has geometric structure beyond linear PCA

**When to use**: For understanding geometric complexity. More accurate than PCA for non-linear manifolds.

**References**: 
- [scikit-dimension documentation](https://scikit-dimension.readthedocs.io/)
- MLE: Levina & Bickel (2005)
- TwoNN: Facco et al. (2017)

---

### 4. General (PyMFE) (`general`)

**Purpose**: Simple dataset characteristics: size, ratios, and variable type counts.

**Key Measures:**

| Measure | Description | Interpretation |
|---------|-------------|---------------|
| `nr_inst` | Number of instances (samples) | Dataset size |
| `nr_attr` | Number of attributes (features) | Input dimensionality |
| `nr_class` | Number of classes | Classification complexity |
| `inst_to_attr` | Samples per feature ratio | **> 10**: Good (enough data)<br>**< 5**: Risk of overfitting |
| `attr_to_inst` | Features per sample ratio | Inverse of inst_to_attr |
| `nr_num` | Number of numeric attributes | Feature type distribution |
| `nr_cat` | Number of categorical attributes | Feature type distribution |
| `cat_to_num` | Categorical to numeric ratio | Mixed data indicator |
| `freq_class` | Class relative frequencies | Class distribution (mean/sd) |

**When to use**: Always useful. Provides baseline dataset characteristics.

**References**: [PyMFE General Group](https://pymfe.readthedocs.io/en/latest/generated/pymfe.general.MFEGeneral.html)

---

### 5. Statistical (PyMFE) (`statistical`)

**Purpose**: Distributional properties of attributes: central tendency, spread, shape, and correlations.

**Key Measures:**

| Measure | Description | Interpretation |
|---------|-------------|---------------|
| `mean`, `sd`, `median`, `min`, `max` | Central tendency and spread | Feature scale and range |
| `iq_range` | Interquartile range | Robust spread measure |
| `mad` | Median Absolute Deviation | Robust dispersion |
| `skewness` | Distribution asymmetry | > 1 or < -1 indicates skew |
| `kurtosis` | Tail heaviness | > 3: heavy tails (outliers) |
| `cov.mean`, `cor.mean` | Mean covariance/correlation | Feature relationships |
| `eigenvalues` | PCA eigenvalues (mean/sd) | Variance distribution |

**When to use**: For understanding feature distributions and relationships. Useful for preprocessing decisions.

**References**: [PyMFE Statistical Group](https://pymfe.readthedocs.io/en/latest/using.html)

---

### 6. Information-Theoretic (PyMFE) (`info-theory`)

**Purpose**: Entropy and mutual information measures capturing uncertainty and attribute-class relationships.

**Key Measures:**

| Measure | Description | Interpretation |
|---------|-------------|---------------|
| `attr_ent` | Attribute entropy | Higher = more uncertainty/variability |
| `class_ent` | Class entropy | Higher = more balanced classes |
| `joint_ent` | Joint entropy (attribute + class) | Combined uncertainty |
| `mut_inf` | Mutual information | Higher = stronger attribute↔class association (predictive power) |

**Interpretation:**
- **High entropy**: More uncertainty, harder to predict
- **High mutual information**: Strong feature-class relationships (good for classification)

**When to use**: For understanding predictive relationships and class separability.

**References**: [PyMFE Information-Theoretic Group](https://pymfe.readthedocs.io/en/latest/using.html)

---

### 7. Model-Based (PyMFE) (`model-based`)

**Purpose**: Complexity measures extracted from fitted decision trees, capturing model complexity and decision boundary structure.

**Key Measures:**

| Measure | Description | Interpretation |
|---------|-------------|---------------|
| `leaves` | Number of leaf nodes | Higher = more complex decision boundaries |
| `nodes` | Total nodes in tree | Overall model complexity |
| `leaves_per_class` | Leaves per class (mean/sd) | Class-specific complexity |
| `nodes_per_attr` | Nodes per attribute | Complexity relative to features |
| `leaves_homo` | Leaf homogeneity | Higher = purer leaves (better separation) |
| `lh_trace` | Leaf homogeneity trace | Cumulative purity measure |

**Interpretation:**
- **High leaves/nodes**: Complex decision boundaries (may need regularization)
- **High leaves_homo**: Good class separation
- Compare across datasets: Higher values = harder to model

**When to use**: For understanding how complex a model needs to be. Good proxy for dataset difficulty.

**References**: [PyMFE Model-Based Group](https://pymfe.readthedocs.io/en/latest/using.html)

---

### 8. Landmarking (PyMFE) (`landmarking`)

**Purpose**: Performance of fast/simple learners (1-NN, Naive Bayes, Linear Discriminant) as meta-features.

**Key Measures:**

| Measure | Description | Interpretation |
|---------|-------------|---------------|
| `one_nn` | 1-Nearest Neighbor accuracy | Baseline non-parametric performance |
| `elite_nn` | Best k-NN accuracy | Best k-NN performance |
| `naive_bayes` | Naive Bayes accuracy | Simple probabilistic model |
| `linear_discr` | Linear Discriminant accuracy | Linear separability indicator |
| `random_node` | Random classifier baseline | Lower bound reference |

**Interpretation:**
- **High landmarking scores**: Easy dataset (simple models work well)
- **Low scores**: Hard dataset (needs complex models)
- Compare across datasets: Relative performance indicates difficulty

**When to use**: For quick difficulty assessment. If simple models perform well, dataset is likely easier.

**References**: [PyMFE Landmarking Group](https://pymfe.readthedocs.io/en/latest/using.html)

---

### 9. Complexity Summary Factor (`complexity_summary_factor`)

**Purpose**: A single category that shows only the meta-features most related to dataset difficulty (easier vs harder). Combines PyMFE **complexity** (l1–l3, n1–n4, f1–f4) and **landmarking** (linear_discr, naive_bayes, one_nn, best_node). Choose this category when you want to focus on complexity interpretation without other groups.

**Measures included:**

| Family | Measures | Easier | Harder |
|--------|----------|--------|--------|
| **Linear separability** | l1, l2, l3 | l1 high; l2, l3 low | l2, l3 high |
| **Neighborhood** | n1, n2, n3, n4 | n1–n4 low | n1, n3, n4 high |
| **Feature overlap** | f1, f2, f3, f4 | f1, f2, f4 low; f3 high | f1, f2, f4 high; f3 low |
| **Landmarking** | linear_discr, naive_bayes, one_nn, best_node | high | low |

**Quick interpretation:**

- **Easier**: l2, l3, n3, n1, f1 **low**; landmarking (lda, naive_bayes, 1nn, dtree) **high**.
- **Harder**: l2, l3, n3, n1, f1 **high**; landmarking **low** → intrinsically hard or noisy.

**If you only watch 6:** l2, l3, n3, n1, f1, and one landmarking (e.g. linear_discr). Low l2/n3/n1/f1 and high landmarking = easier; opposite = harder.

---

#### Comparing two datasets (Part 3: Compare Datasets)

When you compare **Dataset A** vs **Dataset B** for the `complexity_summary_factor` group, the table shows:

- **Columns A and B**: The numeric value of each meta-feature for each dataset.
- **diff (B − A)**: Difference = value of B minus value of A.
  - **B > A** (diff > 0, green): Dataset B has a *higher* value than A for this measure.
  - **B < A** (diff < 0, red): Dataset B has a *lower* value than A.
- **B vs A**: Same as above, in words (↑ B higher / ↓ B lower).
- **Easier**: Which dataset is *easier* (less complex) for this measure.
  - **B** (green): For this feature, B is easier than A.
  - **A** (red): For this feature, A is easier than B.
  - **Same**: Tie.

**Important:** “B > A” does *not* always mean “B is easier.” It depends on the measure:

| Type | Examples | B > A means | B < A means |
|------|----------|-------------|-------------|
| **Hardness** (lower = easier) | l2, l3, n1–n4, f1, f2, f4 | B is *harder* (higher error/overlap) | B is *easier* |
| **Ease** (higher = easier) | l1, f3, linear_discr, naive_bayes, one_nn, best_node | B is *easier* (better performance) | B is *harder* |

So when you read the table:

- For **hardness** metrics (l2, l3, n1, n3, n4, f1, f2, f4): **lower is better**. If B < A, the “Easier” column will show **B**.
- For **ease** metrics (l1, f3, landmarking): **higher is better**. If B > A, the “Easier” column will show **B**.

**Big picture:** Count how many rows show “Easier: B” vs “Easier: A.” If most rows say **Easier: B**, then Dataset B is generally easier (less complex) than A. If most say **Easier: A**, then A is easier. Use that to label datasets (e.g. “A = HARD, B = MODERATE”) and to choose models (simpler models for easier datasets, more complex models for harder ones).

**When to use**: Select **only** this category to see just these complexity-related measures, or combine with other groups for full meta-feature analysis.

**References**: PyMFE complexity and landmarking groups; Lorena et al., “How Complex is your classification problem?” (CSUR 2019).

---

## Recommended Workflow

### Step 1: Quick Complexity Check (5 minutes)
1. Compute **Basic Stats** + **PCA Structure** + **Intrinsic Dimensionality**
2. Check:
   - `pca_k95_over_n_features` < 0.5? → Low redundancy (good)
   - `id_mle` << `n_features`? → Low-dimensional manifold (good)
   - `class_imbalance_ratio_max_min` < 10? → Balanced classes (good)
   - `inst_to_attr` > 10? → Enough samples per feature (good)

### Step 2: Detailed Analysis (if needed)
3. Compute **General** + **Statistical** + **Info-Theory** + **Model-Based** + **Landmarking**
4. Compare across datasets:
   - Lower `leaves`, `nodes` → Simpler decision boundaries
   - Higher `mut_inf` → Better feature-class relationships
   - Higher landmarking scores → Easier dataset

### Step 3: Interpretation
5. **Low complexity indicators**:
   - Low PCA k95/n_features (< 0.5)
   - Low intrinsic dimensionality (< n_features/2)
   - Balanced classes
   - High inst_to_attr (> 10)
   - High landmarking scores
   - Low model-based complexity (leaves, nodes)

6. **High complexity indicators**:
   - High PCA k95/n_features (> 0.8)
   - High intrinsic dimensionality (≈ n_features)
   - Imbalanced classes
   - Low inst_to_attr (< 5)
   - Low landmarking scores
   - High model-based complexity

---

## Comparing Two Datasets

When comparing datasets A and B:

1. **Start with ratios**:
   - `pca_k95_over_n_features`: Lower is simpler
   - `id_mle / n_features`: Lower is simpler
   - `inst_to_attr`: Higher is better (more data per feature)

2. **Check class balance**:
   - `class_imbalance_ratio_max_min`: Lower is better

3. **Compare model complexity**:
   - `leaves`, `nodes`: Lower indicates simpler boundaries
   - `mut_inf`: Higher indicates better predictive relationships

4. **Check landmarking**:
   - Higher scores indicate easier dataset

---

## References

- **PyMFE Documentation**: https://pymfe.readthedocs.io/
- **scikit-dimension**: https://scikit-dimension.readthedocs.io/
- **PCA**: Standard dimensionality reduction (scikit-learn)
- **Intrinsic Dimensionality**: Levina & Bickel (2005), Facco et al. (2017)

---

## Tips

- **Always start with Basic Stats + PCA + Intrinsic Dim** for quick assessment
- **Use comparison view** to see relative differences between datasets
- **Combine multiple measures** for robust complexity assessment
- **Consider your use case**: Classification vs regression may prioritize different measures
