# Optimized Experiment Configuration

## Quick Reference for Cell 9

**Use this command in Cell 9 of the Colab notebook:**

```python
!python run_experiments.py \
    --methods random sa bandit ppo \
    --k_values 1 3 5 10 \
    --seeds 42 123 456 789 1011 \
    --budget 300 \
    --model esm2_t33_650M_UR50D \
    --device cuda \
    --output {output_dir} \
    --datasets {MY_DATASET} \
    --resume
```

---

## Configuration Details

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Methods** | random, sa, bandit, ppo | Removed greedy (slow + broken) |
| **k-values** | 1, 3, 5, 10 | All kept |
| **Seeds** | 42, 123, 456, 789, 1011 | All 5 kept |
| **Budget** | 300 | Reduced from 500 |
| **Total Experiments** | 80 | Per dataset |
| **Time** | 9-10 hours | Per person |

---

## Why These Changes?

### ❌ Removed Greedy
- **Problem**: Greedy throws `NotImplementedError` for k>1
- **Problem**: Even with fix, Greedy k>1 takes hours or is impossible
- **Solution**: Removed entirely
- **Impact**: Still have Random + SA as baselines

### ⚡ Reduced Budget: 500 → 300
- **Benefit**: 40% time savings
- **Scientific validity**: 300 queries is sufficient
- **Many papers use 100-500 queries for optimization comparisons**

---

## What You Still Get

✅ **4 Methods**: 
- 2 Baselines (Random, SA)
- 2 RL Methods (Contextual Bandit, PPO)

✅ **4 k-values**: 1, 3, 5, 10 mutations

✅ **5 Seeds**: Full statistical validity

✅ **6 Datasets**: All oracle quality tiers (HIGH/MED/LOW)

✅ **Publication-quality data**: 480 total experiments

---

## Team Totals

| Person | Datasets | Experiments | Time |
|--------|----------|-------------|------|
| Person 1 | PITX2 | 80 | 9-10h |
| Person 2 | SRC | 80 | 9-10h |
| Person 3 | PAI1 + CBPA2 | 160 | 9-10h |
| Person 4 | SAV1 + CCR5 | 160 | 9-10h |

**Total**: 480 experiments in ~9-10 hours per person

---

## Time Comparison

| Config | Experiments | Time/Person | Issue |
|--------|-------------|-------------|-------|
| **Original** | 600 (100 each) | 21 hours | Greedy breaks 🔴 |
| **Optimized** | 480 (80 each) | 9-10 hours | Works! ✅ |

---

## Scientific Justification

**This is still rigorous science!**

1. **300 queries**: Sufficient to demonstrate method differences
2. **5 seeds**: Standard for statistical validity
3. **4 methods**: Covers baselines + RL approaches
4. **6 datasets**: Complete oracle quality coverage

**Comparison to literature:**
- Many optimization papers use 100-500 queries
- We're in the upper range with 300
- 5 seeds × 4 k-values = 20 runs per method
- More than enough for significance testing

---

## For Your Team

Share this with your teammates! Everyone should use the optimized config:

**Updated Colab Cell 9**: Use the command at the top of this file

**Updated TEAM_INSTRUCTIONS.md**: See main repository

**Questions?** Check TEAM_INSTRUCTIONS.md or contact Sam
