# CS557 Final Project: Protein Fitness Optimization with RL
## When Does RL Help? Oracle Quality and Sample Efficiency

**Author**: Sam  
**Course**: CS557 - Artificial Intelligence  
**Institution**: Emory University  
**Last Updated**: December 7, 2025

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Research Question & Hypothesis](#research-question--hypothesis)
3. [Methodology](#methodology)
4. [Current Progress](#current-progress)
5. [Oracle Validation Results](#oracle-validation-results)
6. [Next Steps](#next-steps)
7. [Timeline](#timeline)
8. [Technical Details](#technical-details)
9. [References](#references)

---

## Project Overview

### **Core Question**
**When does reinforcement learning provide advantages over simpler optimization methods for protein sequence optimization?**

### **Hypothesis**
RL helps most when oracle quality is **MEDIUM** (Spearman ρ ~ 0.4-0.6), not when it's too good or too bad.

**Intuition:**
- **HIGH quality oracle** (ρ ≥ 0.6): Greedy methods work well, RL overhead unnecessary
- **MEDIUM quality oracle** (0.4 ≤ ρ < 0.6): Oracle is noisy enough that exploration helps, but informative enough to guide search
- **LOW quality oracle** (ρ < 0.4): Too noisy, even RL struggles to extract signal

---

## Research Question & Hypothesis

### **Primary Research Question**
How does oracle quality (measured by correlation with ground truth fitness) affect the relative performance of RL versus classical optimization methods?

### **Specific Hypotheses**

**H1**: RL methods (Contextual Bandits, PPO) will show **largest advantage** over greedy baselines in the MEDIUM oracle quality regime (ρ ~ 0.4-0.6)

**H2**: In HIGH quality regimes (ρ ≥ 0.6), simple greedy hill-climbing will perform **comparably** to RL

**H3**: In LOW quality regimes (ρ < 0.4), **all methods** will struggle due to noise, with minimal differences between RL and baselines

### **Metrics**
- **Sample Efficiency**: Fitness improvement vs number of oracle queries
- **Best-of-N**: Best fitness found within query budget
- **Regret**: Gap between best found and optimal variant
- **Exploration vs Exploitation**: Trade-off analysis

---

## Methodology

### **1. Oracle: ESM-2 Protein Language Model**

**Model**: ESM-2 (Evolutionary Scale Modeling)
- Architecture: Transformer-based protein language model
- Trained on: 250M+ protein sequences
- Scoring Method: Pseudo-log-likelihood (mask each position, predict amino acid)

**Why ESM-2?**
- State-of-the-art zero-shot fitness prediction
- Established correlation with experimental fitness (ProteinGym benchmark)
- Controllable oracle quality across different proteins
- No retraining needed (zero-shot)

### **2. Dataset: ProteinGym**

**Source**: ProteinGym v1.3 - Deep Mutational Scanning (DMS) benchmark
- Total available: 217 protein datasets
- Selected: **6 diverse datasets**
- Size range: 1,008 - 6,137 variants per dataset
- Protein length: 120-352 amino acids

**Selection Criteria**:
- Manageable size (500-8,000 variants)
- Diverse size distribution (small, medium, large)
- Single mutants primarily (for tractability)

### **3. Optimization Methods**

#### **Baselines** (Week 2)
1. **Random Sampling**: Uniform random variant selection
2. **Greedy Hill-Climbing**: Always pick best neighbor
3. **Simulated Annealing**: Probabilistic neighbor acceptance

#### **RL Methods** (Week 3-4)
1. **Contextual Bandits** (Primary)
   - Multi-armed bandit with protein context
   - Thompson Sampling or UCB policy
   - Exploration bonus based on uncertainty
   
2. **PPO** (Secondary, if time permits)
   - Proximal Policy Optimization
   - Policy network: sequence → mutation distribution
   - Value network: sequence → expected fitness

### **4. Experimental Design**

**Variables**:
- **Oracle Quality**: HIGH (ρ ≥ 0.6), MEDIUM (0.4-0.6), LOW (< 0.4)
- **Mutation Budget**: k ∈ {1, 3, 5, 10} simultaneous mutations
- **Query Budget**: Fixed at 500 oracle calls
- **Seeds**: 5 random seeds per configuration

**Total Experiments**: 
- Original Plan: 5 methods × 5 seeds × 4 k-values × 6 datasets = **600 runs**
- Adjusted Plan: ~400 runs (see Oracle Validation Results)

**Evaluation**:
- Sample efficiency curves (fitness vs queries)
- Best fitness achieved within budget
- Regret analysis
- Statistical significance testing (Wilcoxon signed-rank)

---

## Current Progress

### **✅ Completed (Weeks 1-2)**

#### **Week 1: Environment Setup & Data Acquisition**
- [x] Created project structure (`src/`, `data/`, `experiments/`, `tests/`)
- [x] Set up conda environment with dependencies:
  - PyTorch 2.0+ with MPS/CUDA support
  - ESM-2 (`esm` package)
  - Scientific stack (numpy, pandas, scipy, scikit-learn)
  - Visualization (matplotlib, seaborn, plotly)
  - RL libraries (gym, stable-baselines3)
- [x] Downloaded ProteinGym metadata (217 datasets)
- [x] Selected 6 diverse datasets
- [x] Downloaded datasets via official ProteinGym ZIP
- [x] Extracted wild-type sequences for all 6 proteins

#### **Week 2: Oracle Validation**
- [x] Implemented ESM-2 oracle class with pseudo-log-likelihood scoring
- [x] Implemented mutation application utilities
- [x] Validated oracle quality on all 6 datasets (500 samples each)
- [x] Computed Spearman/Pearson correlations
- [x] Categorized datasets into HIGH/MEDIUM/LOW tiers
- [x] Generated correlation plots
- [x] Optimized for Apple Silicon (MPS) and Colab Pro (CUDA)

### **Scripts Created**

#### **Data Pipeline** (`src/utils/`)
1. `download_metadata.py` - Download ProteinGym metadata
2. `select_datasets.py` - Select 6 diverse datasets from 217 available
3. `download_datasets.py` - Download full ProteinGym ZIP, extract selected datasets
4. `extract_wild_types.py` - Extract wild-type sequences from DMS data
5. `validate_oracle.py` - Validate ESM-2 correlation with experimental fitness

#### **Infrastructure**
- `tests/test_installation.py` - Verify dependencies
- Directory structure with proper organization
- README documentation

---

## Oracle Validation Results

### **Validation Setup**
- **Model**: ESM-2 t12_35M (small model for speed)
- **Platform**: Google Colab Pro (Tesla T4 GPU)
- **Samples**: 500 variants per dataset
- **Method**: Spearman correlation between ESM-2 scores and DMS fitness
- **Total Time**: 149.7 minutes (~25 min/dataset)

### **Oracle Quality Distribution**

| Tier | Expected | Actual | Status |
|------|----------|--------|--------|
| HIGH (ρ ≥ 0.6) | 2 | **1** | ⚠️ Under |
| MEDIUM (0.4-0.6) | 2 | **1** | ⚠️ Under |
| LOW (< 0.4) | 2 | **4** | ⚠️ Over |

### **Detailed Results**

| Dataset | Variants | Protein Length | Spearman ρ | Tier | Notes |
|---------|----------|----------------|------------|------|-------|
| **PITX2_HUMAN_Tsuboyama_2023_2L7M** | 1,824 | 271 AA | **0.689** | HIGH | ✓ Good oracle |
| **SRC_HUMAN_Ahler_2019** | 3,372 | 536 AA | **0.474** | MEDIUM | ✓ Target regime |
| **PAI1_HUMAN_Huttinger_2021** | 5,345 | 379 AA | 0.393 | LOW | Near MEDIUM |
| **CCR5_HUMAN_Gill_2023** | 6,137 | 352 AA | 0.349 | LOW | |
| **DN7A_SACS2_Tsuboyama_2023_1JIC** | 1,008 | 144 AA | 0.351 | LOW | |
| **RFAH_ECOLI_Tsuboyama_2023_2LCL** | 1,326 | 162 AA | 0.295 | LOW | Weakest oracle |

### **Analysis**

**Why Unbalanced?**
- Dataset selection prioritized **smallest sizes** for computational efficiency
- Smaller proteins often have **lower ESM-2 correlation** (less evolutionary signal)
- No pre-filtering by oracle quality (metadata lacked correlation scores)
- Random selection was unlucky

**Implications**:
- Have at least **1 representative from each tier** ✓
- Can still test core hypothesis (RL helps most in MEDIUM)
- Limited statistical power for HIGH/MEDIUM tiers (n=1 each)
- Strong evidence possible for LOW tier (n=4)

**Visualization**: 
- Correlation plots saved in: `experiments/oracle_validation/plots/`
- Summary CSV: `experiments/oracle_validation/oracle_validation_summary.csv`

---

## Next Steps

### **Adjusted Experimental Plan**

Given the unbalanced distribution, we have **two options**:

#### **Option A: Proceed with Current Datasets** (Recommended)

**Rationale**: 
- Already validated 6 datasets (2.5 hours of compute)
- Have at least one representative per tier
- Can still test hypothesis with reduced statistical power
- Acknowledge limitation in paper

**Experimental Design**:
```
HIGH tier (n=1):
  - PITX2_HUMAN: 5 methods × 5 seeds × 4 k-values = 100 runs

MEDIUM tier (n=1):
  - SRC_HUMAN: 5 methods × 5 seeds × 4 k-values = 100 runs

LOW tier (n=4, select best 2):
  - PAI1_HUMAN (ρ=0.393): 5 methods × 5 seeds × 4 k-values = 100 runs
  - CCR5_HUMAN (ρ=0.349): 5 methods × 5 seeds × 4 k-values = 100 runs

Total: 400 runs
```

**Limitations to Acknowledge**:
> "Due to ESM-2's variable performance across proteins, our final dataset distribution was 1 HIGH, 1 MEDIUM, and 2 LOW quality oracle datasets. While this limits our ability to make strong statistical claims about the HIGH and MEDIUM tiers individually, it provides sufficient evidence to test our core hypothesis that RL advantage varies with oracle quality."

#### **Option B: Re-Select and Validate More Datasets**

**Process**:
1. Examine ProteinGym metadata for datasets with likely better oracle quality
2. Select 4 new candidates targeting HIGH/MEDIUM
3. Validate on Colab (~1.5-2 hours)
4. Replace weakest LOW datasets

**Pros**: Better balanced distribution, stronger statistical power  
**Cons**: Additional 2 hours validation time, risk of still not finding good matches

---

### **Implementation Timeline**

#### **Week 3: Baselines Implementation** (Dec 8-14)
- [ ] Implement mutation operators (single, multiple simultaneous)
- [ ] Implement Random baseline
- [ ] Implement Greedy hill-climbing baseline
- [ ] Implement Simulated Annealing baseline
- [ ] Test all baselines on 1 dataset
- [ ] Verify correct oracle usage and query counting

**Deliverable**: Working baselines with unit tests

#### **Week 4: RL Methods** (Dec 15-21)
- [ ] Implement Contextual Bandits
  - State representation: ESM-2 embeddings
  - Action space: k-mutation combinations
  - Exploration: Thompson Sampling or UCB
- [ ] (Optional) Implement PPO if time permits
- [ ] Test RL methods on 1 dataset
- [ ] Hyperparameter tuning

**Deliverable**: Working RL methods

#### **Week 5: Experiments** (Dec 22-28)
- [ ] Run all methods on all datasets
  - Parallelization strategy (multiple seeds simultaneously)
  - Use Colab Pro for GPU acceleration
- [ ] Monitor experiments and handle failures
- [ ] Save all results with proper versioning

**Deliverable**: Complete experimental results

#### **Week 6: Analysis & Writing** (Dec 29 - Jan 4)
- [ ] Generate sample efficiency curves
- [ ] Statistical significance testing
- [ ] Create visualizations (plots, tables)
- [ ] Write paper sections
- [ ] Prepare presentation

**Deliverable**: Final paper and presentation

---

## Technical Details

### **Directory Structure**

```
Protein_RL_Project/
├── data/
│   ├── raw/
│   │   ├── dms_datasets/           # 6 DMS CSV files
│   │   ├── wild_types/             # 6 FASTA files
│   │   ├── DMS_substitutions_metadata.csv
│   │   ├── selected_datasets.csv
│   │   └── selected_ids.txt
│   └── processed/                  # Will contain processed features
├── src/
│   ├── models/                     # ESM-2 oracle implementation
│   ├── baselines/                  # Random, Greedy, SA
│   ├── rl_methods/                 # Contextual Bandits, PPO
│   └── utils/                      # Data loading, evaluation
├── experiments/
│   ├── results/                    # Experiment outputs
│   ├── plots/                      # Visualizations
│   └── logs/                       # Training logs
├── tests/                          # Unit tests
└── notebooks/                      # Analysis notebooks
```

### **Key Implementation Details**

#### **ESM-2 Oracle**
```python
class ESM2Oracle:
    """
    Pseudo-log-likelihood scoring:
    - Mask each position i in sequence
    - Predict amino acid at position i
    - Sum log probabilities: Σ log P(x_i | x_{-i})
    """
    
    def score_sequence(self, sequence):
        # Returns: Sum of log probabilities (higher = better fitness)
```

**Models Available**:
- `esm2_t6_8M_UR50D`: 8M params, very fast
- `esm2_t12_35M_UR50D`: 35M params, **fast, used for validation**
- `esm2_t33_650M_UR50D`: 650M params, most accurate, slow on CPU

**Device Support**:
- Apple Silicon: MPS (Metal Performance Shaders)
- NVIDIA GPU: CUDA
- CPU: Fallback (very slow)

#### **Mutation Representation**
- **Format**: "A2V" = position 2 (1-indexed), A→V
- **Multiple**: "A2V:K5R:D10E" (colon-separated)
- **Wild-type**: "WT"
- **Amino acids**: 20 standard (ACDEFGHIKLMNPQRSTVWY)

#### **Experimental Protocol**
1. Start with wild-type sequence
2. For each method:
   - Initialize with wild-type
   - For budget iterations:
     - Select k positions to mutate
     - Query oracle for fitness
     - Update method state
     - Record: query count, fitness, sequence
3. Repeat for 5 random seeds
4. Aggregate results

---

## Computational Resources

### **Hardware Used**

**Local (Development)**:
- MacBook Pro M3 Pro
- 18 GB RAM
- Apple Metal (MPS) GPU
- Used for: script development, small tests

**Cloud (Heavy Compute)**:
- Google Colab Pro
- Tesla T4 GPU (16 GB VRAM)
- 25-50 GB RAM
- Used for: oracle validation (2.5 hours)
- Will use for: full experiments (~10-20 hours)

### **Estimated Compute Requirements**

**Oracle Validation**: ✅ Complete
- 6 datasets × 500 variants × ~352 AA avg
- Tesla T4: ~25 min/dataset
- Total: 2.5 hours

**Full Experiments**: Upcoming
- 400 runs × 500 queries × ~352 AA avg
- Estimated: 10-20 hours on Colab Pro T4
- Can parallelize across seeds

---

## Data Files

### **Downloaded Datasets**

| Dataset | Size | Protein | Length | Function |
|---------|------|---------|--------|----------|
| CCR5_HUMAN_Gill_2023 | 6,137 | C-C chemokine receptor 5 | 352 AA | HIV co-receptor |
| PAI1_HUMAN_Huttinger_2021 | 5,345 | Plasminogen activator inhibitor 1 | 379 AA | Blood clotting |
| SRC_HUMAN_Ahler_2019 | 3,372 | Proto-oncogene tyrosine kinase | 536 AA | Cell signaling |
| PITX2_HUMAN_Tsuboyama_2023 | 1,824 | Pituitary homeobox 2 | 271 AA | Transcription factor |
| RFAH_ECOLI_Tsuboyama_2023 | 1,326 | Transcription antitermination | 162 AA | RNA polymerase |
| DN7A_SACS2_Tsuboyama_2023 | 1,008 | DnaJ homolog subfamily A | 144 AA | Protein folding |

### **File Locations**

**Raw Data**:
- DMS datasets: `data/raw/dms_datasets/*.csv`
- Wild-types: `data/raw/wild_types/*.fasta`
- Metadata: `data/raw/DMS_substitutions_metadata.csv`

**Validation Results**:
- Summary: `experiments/oracle_validation/oracle_validation_summary.csv`
- Plots: `experiments/oracle_validation/plots/*.png`

**Scripts**:
- All utilities: `src/utils/*.py`
- Future baselines: `src/baselines/*.py`
- Future RL: `src/rl_methods/*.py`

---

## Key Decisions Made

### **1. Dataset Selection Strategy**

**Initial Plan**: Select 6 datasets with 2 each from HIGH/MEDIUM/LOW tiers

**Reality**: Metadata lacked pre-computed oracle quality scores

**Solution**: 
- Selected 6 diverse datasets by size (small, medium, large)
- Validated oracle quality post-hoc
- Got unbalanced distribution (1 HIGH, 1 MEDIUM, 4 LOW)

**Lesson**: Should have validated a larger pool (~10-15) then selected best 6

### **2. Validation Sample Size**

**Decision**: Use 500 samples per dataset for validation

**Rationale**:
- Standard error ≈ 1/√500 ≈ 0.045
- Tier boundaries are 0.2 wide (HIGH: 0.6-1.0, MEDIUM: 0.4-0.6)
- 0.045 << 0.2, so 500 samples sufficient for tier assignment

**Alternative**: Could use fewer (200) for speed, but 500 is gold standard

### **3. ESM-2 Model Size**

**Decision**: Use small model (35M) for validation, potentially large (650M) for experiments

**Validation**: Used `esm2_t12_35M_UR50D`
- Faster: ~25 min/dataset on T4
- Accurate enough: correlation difference ±0.02-0.03 vs large model
- Won't change tier assignments

**Experiments**: TBD
- Could use small model for speed (400 runs × 500 queries = 200K oracle calls)
- Or large model for accuracy (if Colab Pro time permits)

### **4. Hardware Platform**

**Development**: Local M3 Pro with MPS
- Pros: Convenient, no upload/download
- Cons: Slow for large model (15-20 min per dataset)

**Heavy Compute**: Google Colab Pro
- Pros: Fast T4 GPU, no local battery drain, reproducible notebooks
- Cons: Setup time, file transfers

**Decision**: Hybrid approach
- Local for development and testing
- Colab for validation and full experiments

---

## Potential Challenges & Mitigations

### **Challenge 1: Unbalanced Oracle Quality Distribution**

**Impact**: Limited statistical power for HIGH/MEDIUM tiers (n=1 each)

**Mitigations**:
1. **Accept limitation**: Acknowledge in paper, focus on LOW tier (n=4)
2. **Re-validate**: Spend 2 more hours finding better datasets
3. **Adjust hypothesis**: Frame as exploratory study of oracle quality effect

**Chosen**: Option 1 (pragmatic, saves time for actual experiments)

### **Challenge 2: Computational Budget**

**Impact**: 400 runs × 500 queries × 352 AA avg = ~70M forward passes

**Mitigations**:
1. **Small model**: Use esm2_t12_35M for speed
2. **Parallelization**: Run multiple seeds simultaneously
3. **Colab Pro**: Leverage cloud GPU
4. **Caching**: Cache sequence scores to avoid recomputation

**Estimate**: ~10-20 hours on Colab Pro T4 (acceptable)

### **Challenge 3: RL Implementation Complexity**

**Impact**: Contextual bandits with protein sequences is non-trivial

**Mitigations**:
1. **Start simple**: Thompson Sampling with ESM-2 embeddings as context
2. **Use libraries**: Leverage stable-baselines3 for PPO
3. **Focus on bandits**: PPO is secondary, can skip if time-limited
4. **Test early**: Implement on 1 dataset before full experiments

### **Challenge 4: Time Constraint**

**Impact**: ~4 weeks remaining (with winter break)

**Mitigations**:
1. **Prioritize**: Baselines + Contextual Bandits (skip PPO if needed)
2. **Parallel work**: Can implement baselines while Colab runs validation
3. **Accept scope**: 400 runs on 4 datasets is sufficient for CS557 project
4. **Plan buffer**: Aim for completion by Dec 28, leaving Jan 1-4 for writing

---

## Questions for Discussion

### **Scientific**
1. Should we re-validate more datasets to get better HIGH/MEDIUM representation?
2. Is 500 oracle queries sufficient, or should we test with 1000?
3. Should we include PPO or focus on Contextual Bandits?
4. What mutation operators make sense? (random k-point, learned distributions?)

### **Technical**
1. Small vs large ESM-2 model for experiments?
2. How to best represent sequences as context for bandits?
3. Parallelization strategy for 400 runs?
4. Where to cache oracle scores for efficiency?

### **Experimental**
1. How many k-values? Currently 4 {1,3,5,10}, could reduce to 3 {1,5,10}
2. Statistical tests: Wilcoxon signed-rank sufficient?
3. What baselines are essential? Is simulated annealing needed?

---

## References

### **Papers**

1. **ProteinGym Benchmark**
   - Notin et al. (2024) "ProteinGym: Large-Scale Benchmarks for Protein Design and Fitness Prediction"
   - Source of DMS datasets and experimental fitness measurements

2. **ESM-2 Model**
   - Lin et al. (2023) "Evolutionary-scale prediction of atomic-level protein structure with a language model"
   - Pseudo-log-likelihood scoring for zero-shot fitness prediction

3. **Protein RL**
   - Joshi et al. (2023) "Contextual Bandits for Protein Design"
   - Biswas et al. (2021) "Low-N protein engineering with data-efficient deep learning"

### **Code & Tools**

- **ESM**: https://github.com/facebookresearch/esm
- **ProteinGym**: https://github.com/OATML-Markslab/ProteinGym
- **Stable-Baselines3**: https://stable-baselines3.readthedocs.io/

### **Datasets**

- **ProteinGym v1.3**: https://marks.hms.harvard.edu/proteingym/
- **Download URL**: https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_substitutions.zip

---

## Acknowledgments

- **Course**: CS557 Artificial Intelligence, Emory University
- **Instructor**: [Instructor Name]
- **Resources**: Google Colab Pro, ProteinGym, ESM-2

---

## Contact

**Sam**  
Email: [your email]  
GitHub: [your github]

---

**Last Updated**: December 7, 2025, 10:40 PM  
**Status**: Oracle validation complete, ready to implement baselines
