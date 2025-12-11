# Team Parallel Execution Guide - CS557 Final Project

## 🎯 Quick Overview

**Project**: Protein RL Optimization Experiments  
**Total**: 480 experiments, divided among 4 people (OPTIMIZED)  
**Time**: ~9-10 hours per person (all finish together!)  
**GPU**: L4 (available in Colab Pro)  

**⚡ OPTIMIZED CONFIGURATION:**
- 4 methods (Greedy removed for speed)
- Budget reduced: 500 → 300 queries
- Still scientifically rigorous!

---

## 📋 Dataset Assignments

| Person | Dataset(s) | Tier | Experiments | Time |
|--------|-----------|------|-------------|------|
| **Person 1** | PITX2_HUMAN | HIGH | 80 | ~9-10h |
| **Person 2** | SRC_HUMAN | MEDIUM | 80 | ~9-10h |
| **Person 3** | PAI1_HUMAN + CBPA2_HUMAN | LOW + HIGH | 160 | ~9-10h (2 parallel notebooks) |
| **Person 4** | SAV1_MOUSE + CCR5_HUMAN | MEDIUM + LOW | 160 | ~9-10h (2 parallel notebooks) |

**Person 3 & 4 Special Instructions**: 
- Run **2 notebooks simultaneously** (one per dataset)
- Open 2 browser tabs with the same Colab notebook
- Configure each tab with a different dataset
- Both will run in parallel on the same Colab account

---

## 🚀 Step-by-Step Instructions

### **Step 1: Open the Colab Notebook**

**Colab Link**: https://colab.research.google.com/drive/15I9yJqePIOWkACRwCiyhkhqRlmw2ahAE?usp=sharing

**Important**: Make a copy for yourself!
- File → Save a copy in Drive

### **Step 2: Select L4 GPU**

1. Click **Runtime** → **Change runtime type**
2. Hardware accelerator: **GPU**
3. GPU type: **L4** (premium GPU)
4. Click **Save**

### **Step 3: Configure Your Dataset**

In **Cell 1**, change these two lines:

**Person 1**:
```python
MY_DATASET = "PITX2_HUMAN_Tsuboyama_2023_2L7M"
MY_NAME = "Person1"
```

**Person 2**:
```python
MY_DATASET = "SRC_HUMAN_Ahler_2019"
MY_NAME = "Person2"
```

**Person 3 - Notebook A** (Tab 1):
```python
MY_DATASET = "PAI1_HUMAN_Huttinger_2021"
MY_NAME = "Person3A"
```

**Person 3 - Notebook B** (Tab 2):
```python
MY_DATASET = "CBPA2_HUMAN_Tsuboyama_2023_1O6X"
MY_NAME = "Person3B"
```

**Person 4 - Notebook A** (Tab 1):
```python
MY_DATASET = "SAV1_MOUSE_Tsuboyama_2023_2YSB"
MY_NAME = "Person4A"
```

**Person 4 - Notebook B** (Tab 2):
```python
MY_DATASET = "CCR5_HUMAN_Gill_2023"
MY_NAME = "Person4B"
```

### **Step 4: Run All Cells**

1. Click **Runtime** → **Run all**
2. When prompted, allow Google Drive access
3. Wait for experiments to complete (~4-5 hours)

**Important**: Keep the Colab tab open! Colab may disconnect if you close the tab or if you're inactive for too long.

---

## ⚡ Why the Optimizations?

**Original Plan**: 600 experiments × 500 queries = 21+ hours per person 🔴  
**Optimized Plan**: 480 experiments × 300 queries = 9-10 hours per person ✅

**What we removed:**
- ❌ **Greedy baseline**: Slow (6h) + implementation issues for k>1
- ⚠️ **Budget 500→300**: Still enough to show method differences

**What we kept (the important stuff):**
- ✅ **4 methods**: Random + SA (baselines) + Bandit + PPO (RL)
- ✅ **All 4 k-values**: 1, 3, 5, 10
- ✅ **5 seeds**: Full statistical validity
- ✅ **6 datasets**: Complete oracle quality coverage

**Scientific validity**: 300 queries is sufficient to demonstrate RL advantages. Many papers use 100-500 queries.

---

## ⚠️ Important Notes

### **Keep Colab Active**
- **Don't close the browser tab**
- **Move your mouse occasionally** to prevent disconnection
- If using Colab Pro: you can run in background
- If disconnected: Just run Cell 9 again (it will resume automatically)

### **Auto-Save & Resume**
- ✅ Results save after **each experiment** (safe!)
- ✅ If disconnected, rerun Cell 9 with `--resume` flag (already included)
- ✅ Won't redo completed experiments

### **Monitor Progress**
- Run **Cell 10** anytime to see progress
- Check your Google Drive folder for saved results

### **Person 3 & 4 - Running 2 Notebooks Each**

**Person 3**:
1. Open the Colab link
2. Make a copy: "Person3A_PAI1"
3. Configure: `MY_DATASET = "PAI1_HUMAN_Huttinger_2021"`, `MY_NAME = "Person3A"`
4. In a new tab, open the link again
5. Make another copy: "Person3B_CBPA2"
6. Configure: `MY_DATASET = "CBPA2_HUMAN_Tsuboyama_2023_1O6X"`, `MY_NAME = "Person3B"`
7. Run both simultaneously!

**Person 4**:
1. Open the Colab link
2. Make a copy: "Person4A_SAV1"
3. Configure: `MY_DATASET = "SAV1_MOUSE_Tsuboyama_2023_2YSB"`, `MY_NAME = "Person4A"`
4. In a new tab, open the link again
5. Make another copy: "Person4B_CCR5"
6. Configure: `MY_DATASET = "CCR5_HUMAN_Gill_2023"`, `MY_NAME = "Person4B"`
7. Run both simultaneously!

---

## 📊 Expected Time Estimates

**With L4 GPU** (recommended - OPTIMIZED CONFIG):
- Single experiment: ~6-8 minutes
- 80 experiments: ~9-10 hours
- 160 experiments (parallel): ~9-10 hours (Person 3 & 4)

**With T4 GPU** (free tier):
- Single experiment: ~10-12 minutes
- 80 experiments: ~12-15 hours
- May timeout on free tier

**Recommendation**: Use Colab Pro for L4 access!

**Why 9-10 hours?**
- ⚡ Removed slow Greedy method
- ⚡ Reduced budget: 500 → 300 queries
- ✅ Still scientifically valid (4 methods, 5 seeds, 300 queries)

---

## 💾 Results Location

Your results will be saved in Google Drive:

```
Google Drive/
└── Protein_RL_Results/
    ├── Person1_PITX2/
    │   ├── result_PITX2_HUMAN_..._random_k1_seed42_....pkl
    │   ├── result_PITX2_HUMAN_..._bandit_k1_seed42_....pkl
    │   └── ... (100 files)
    ├── Person2_SRC/
    │   └── ... (100 files)
    ├── Person3A_PAI1/
    │   └── ... (100 files)
    ├── Person3B_CBPA2/
    │   └── ... (100 files)
    ├── Person4A_SAV1/
    │   └── ... (100 files)
    └── Person4B_CCR5/
        └── ... (100 files)
```

---

## 🐛 Troubleshooting

### **"No GPU detected"**
→ Runtime → Change runtime type → GPU → L4

### **"Colab disconnected"**
→ Just run **Cell 9** again (it uses `--resume` flag automatically)

### **"Out of memory"**
→ Shouldn't happen on L4 (24GB VRAM)  
→ If it does: Runtime → Restart runtime, then run all cells again

### **"Dataset not found in ZIP"**
→ Check the dataset name in Cell 1  
→ Make sure it matches exactly (copy from instructions above)

### **"Cell takes forever to run"**
→ Check you selected **L4 GPU** (not CPU!)  
→ Runtime → Change runtime type → verify GPU = L4

### **"Cannot access Google Drive"**
→ Cell 4 will prompt you to authorize  
→ Click the link and sign in with your Google account

---

## 📈 Cell-by-Cell Guide

| Cell | Purpose | Time | Action |
|------|---------|------|--------|
| 1 | Configuration | <1s | ✏️ **EDIT: Change dataset & name** |
| 2 | Check GPU | <1s | ✓ Verify L4 detected |
| 3 | Install dependencies | ~2 min | ✓ Wait |
| 4 | Mount Google Drive | ~10s | ✓ Authorize access |
| 5 | Clone repository | ~10s | ✓ Wait |
| 6 | Download dataset | ~5-10 min | ✓ Wait (downloads 500MB) |
| 7 | Extract wild-type | ~5s | ✓ Wait |
| 8 | Quick test | ~1-2 min | ✓ Verify it works |
| 9 | **RUN EXPERIMENTS** | **~9-10 hrs** | ⏰ **This is the main run!** |
| 10 | Monitor progress | - | Optional: Check progress |
| 11 | View results | ~10s | After completion |

---

## ✅ Checklist Before Starting

- [ ] Opened Colab notebook
- [ ] Made a copy in my Drive
- [ ] Selected L4 GPU runtime
- [ ] Changed `MY_DATASET` in Cell 1
- [ ] Changed `MY_NAME` in Cell 1
- [ ] Ready to run for 4-5 hours
- [ ] Won't close browser tab

---

## 🎯 Success Criteria

**You're done when:**
- Cell 9 shows: "✓✓✓ ALL EXPERIMENTS COMPLETE!"
- Cell 11 shows: "Completed experiments: 80/80"
- Your Google Drive folder has 80+ files

---

## 📞 Contact

**Questions?** Contact Sam

**Issues?** 
1. Check troubleshooting section above
2. Take a screenshot of the error
3. Share with Sam

---

## 🎉 After Completion

1. **Don't delete anything!** Results are needed for final analysis
2. Notify team that your dataset is complete
3. Share your Google Drive folder link with Sam
4. Take a screenshot of Cell 11 (results summary)

---

## 📊 What Gets Measured

Each of your 80 experiments measures:
- **4 methods**: Random, Simulated Annealing, Contextual Bandit, PPO (optimized)
- **4 k-values**: 1, 3, 5, 10 mutations
- **5 seeds**: 42, 123, 456, 789, 1011
- **300 queries** per experiment (optimized from 500)

**Total data points**: 80 experiments × 300 queries = 24,000 measurements per person!

**Why no Greedy?** Removed for speed - it's slow and has implementation issues for k>1. You still have Random + SA as baselines.

---

## 🏆 Team Progress Tracker

Update this as you complete:

- [ ] Person 1: PITX2_HUMAN (80 experiments)
- [ ] Person 2: SRC_HUMAN (80 experiments)  
- [ ] Person 3: PAI1_HUMAN + CBPA2_HUMAN (160 experiments)
- [ ] Person 4: SAV1_MOUSE + CCR5_HUMAN (160 experiments)

**Total**: 0/480 experiments complete (OPTIMIZED)

**Breakdown by Dataset**:
- [ ] PITX2_HUMAN (HIGH) - Person 1 - 80 exp
- [ ] SRC_HUMAN (MEDIUM) - Person 2 - 80 exp
- [ ] PAI1_HUMAN (LOW) - Person 3A - 80 exp
- [ ] CBPA2_HUMAN (HIGH) - Person 3B - 80 exp
- [ ] SAV1_MOUSE (MEDIUM) - Person 4A - 80 exp
- [ ] CCR5_HUMAN (LOW) - Person 4B - 80 exp

---

## 🚀 Let's Go!

**Goal**: All 4 people finish in ~9-10 hours  
**Strategy**: Parallel execution + Optimized config  
**Timeline**: Start together, finish together!

**⚡ Optimizations:**
- 4 methods (no Greedy - saved 6 hours!)
- Budget: 300 queries (still rigorous)
- Total: 480 experiments across 6 datasets

**This is publication-quality data!** 🎓

Good luck! 🎯
