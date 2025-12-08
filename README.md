# Protein Fitness Optimization with RL

**CS557 Final Project - When Does RL Help?**

## Quick Start

### Setup Environment
```bash
conda create -n protein_rl python=3.10
conda activate protein_rl
pip install torch numpy pandas scipy matplotlib seaborn scikit-learn biopython esm
```

### Download Data
```bash
# 1. Download metadata
python src/utils/download_metadata.py

# 2. Select datasets
python src/utils/select_datasets.py

# 3. Download datasets
python src/utils/download_datasets.py

# 4. Extract wild-types
python src/utils/extract_wild_types.py
```

### Validate Oracle
```bash
# Local (slow on CPU, ~10 hours)
python src/utils/validate_oracle.py

# Or use Colab Pro (recommended, ~2.5 hours)
# See PROJECT_STATUS.md for Colab notebook
```

## Project Structure

```
Protein_RL_Project/
├── data/
│   ├── raw/
│   │   ├── dms_datasets/         # 6 DMS datasets (CSV)
│   │   └── wild_types/           # 6 wild-type sequences (FASTA)
│   └── processed/                # Processed features (TBD)
├── src/
│   ├── models/                   # ESM-2 oracle (TODO)
│   ├── baselines/                # Random, Greedy, SA (TODO)
│   ├── rl_methods/               # Contextual Bandits, PPO (TODO)
│   └── utils/                    # Data utilities (✓)
├── experiments/
│   ├── results/                  # Experiment outputs (TODO)
│   └── plots/                    # Visualizations (TODO)
└── tests/                        # Unit tests
```

## Current Status

### ✅ Completed
- [x] Environment setup
- [x] Data download pipeline
- [x] 6 datasets selected and downloaded
- [x] Wild-type sequences extracted
- [x] Oracle validation (ESM-2 vs experimental fitness)

### 🚧 In Progress
- [ ] Implement baselines (Random, Greedy, SA)
- [ ] Implement RL methods (Contextual Bandits, PPO)
- [ ] Run experiments (400 runs)
- [ ] Analysis and visualization

### 📊 Oracle Validation Results

| Dataset | Oracle Quality | Spearman ρ |
|---------|---------------|------------|
| PITX2_HUMAN | HIGH | 0.689 |
| SRC_HUMAN | MEDIUM | 0.474 |
| PAI1_HUMAN | LOW | 0.393 |
| CCR5_HUMAN | LOW | 0.349 |
| DN7A_SACS2 | LOW | 0.351 |
| RFAH_ECOLI | LOW | 0.295 |

**Distribution**: 1 HIGH, 1 MEDIUM, 4 LOW

## Documentation

- **[PROJECT_STATUS.md](PROJECT_STATUS.md)**: Comprehensive project documentation
  - Detailed progress
  - Oracle validation results
  - Next steps and timeline
  - Technical details

## Next Steps

1. **Week 3**: Implement baselines
2. **Week 4**: Implement RL methods  
3. **Week 5**: Run experiments
4. **Week 6**: Analysis and writing

## Contact

**Sam**  
CS557 - Artificial Intelligence  
Emory University

---

**Last Updated**: December 7, 2025
