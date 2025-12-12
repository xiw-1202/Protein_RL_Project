# Next Steps: Implementation Roadmap

**Current Phase**: Oracle Validation Complete ✅  
**Next Phase**: Baseline Implementation (Week 3)

---

## Immediate Action Items (This Week)

### 1. Implement ESM-2 Oracle Class
**File**: `src/models/esm_oracle.py`

```python
class ESM2Oracle:
    """
    ESM-2 oracle for protein fitness prediction
    """
    def __init__(self, model_name="esm2_t12_35M_UR50D", device="auto"):
        # Load model, setup device
        pass
    
    def score_sequence(self, sequence: str) -> float:
        # Pseudo-log-likelihood scoring
        # Returns: fitness score (higher = better)
        pass
    
    def score_batch(self, sequences: List[str]) -> np.ndarray:
        # Batch scoring for efficiency
        pass
```

**Tasks**:
- [ ] Copy validation code as starting point
- [ ] Add batch scoring for efficiency
- [ ] Add caching to avoid re-scoring same sequences
- [ ] Test on sample sequences
- [ ] Benchmark speed (sequences/sec)

**Estimated Time**: 2-3 hours

---

### 2. Implement Mutation Operators
**File**: `src/utils/mutations.py`

```python
def get_neighbors(sequence: str, k: int = 1) -> List[str]:
    """
    Generate all k-point mutant neighbors
    
    Args:
        sequence: Wild-type sequence
        k: Number of simultaneous mutations
    
    Returns:
        List of mutant sequences (k mutations from WT)
    """
    pass

def apply_mutation(sequence: str, mutation: str) -> str:
    """
    Apply mutation string to sequence
    
    Args:
        sequence: Starting sequence
        mutation: Format "A2V" or "A2V:K5R"
    
    Returns:
        Mutated sequence
    """
    pass

def parse_mutation(mutation: str) -> List[Tuple[int, str, str]]:
    """
    Parse mutation string into components
    
    Args:
        mutation: "A2V:K5R:D10E"
    
    Returns:
        [(1, 'A', 'V'), (4, 'K', 'R'), (9, 'D', 'E')]
        (positions are 0-indexed)
    """
    pass
```

**Tasks**:
- [ ] Implement mutation parsing
- [ ] Implement single-point mutations
- [ ] Implement k-point mutations (all combinations)
- [ ] Add validation (check wild-type matches)
- [ ] Write unit tests

**Estimated Time**: 3-4 hours

---

### 3. Implement Random Baseline
**File**: `src/baselines/random_baseline.py`

```python
class RandomBaseline:
    """
    Random sampling baseline
    """
    def __init__(self, oracle, k=1, seed=42):
        self.oracle = oracle
        self.k = k  # Number of simultaneous mutations
        self.rng = np.random.RandomState(seed)
    
    def optimize(self, wt_sequence: str, budget: int = 500) -> Dict:
        """
        Random search
        
        Args:
            wt_sequence: Wild-type starting sequence
            budget: Number of oracle queries
        
        Returns:
            {
                'best_sequence': str,
                'best_fitness': float,
                'history': List[Tuple[sequence, fitness]],
                'queries_used': int
            }
        """
        history = []
        best_seq = wt_sequence
        best_fitness = self.oracle.score_sequence(wt_sequence)
        
        for i in range(budget):
            # Generate random k-point mutant
            mutant = self._random_mutant(wt_sequence)
            fitness = self.oracle.score_sequence(mutant)
            
            history.append((mutant, fitness))
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_seq = mutant
        
        return {
            'best_sequence': best_seq,
            'best_fitness': best_fitness,
            'history': history,
            'queries_used': budget
        }
    
    def _random_mutant(self, sequence: str) -> str:
        # Randomly select k positions
        # Randomly select amino acids
        # Return mutated sequence
        pass
```

**Tasks**:
- [ ] Implement random mutation generation
- [ ] Test on one dataset
- [ ] Verify query counting is correct
- [ ] Save results to file
- [ ] Plot fitness vs queries

**Estimated Time**: 2-3 hours

---

### 4. Implement Greedy Hill-Climbing
**File**: `src/baselines/greedy.py`

```python
class GreedyHillClimbing:
    """
    Greedy hill-climbing baseline
    """
    def __init__(self, oracle, k=1):
        self.oracle = oracle
        self.k = k
    
    def optimize(self, wt_sequence: str, budget: int = 500) -> Dict:
        """
        Greedy search: always pick best neighbor
        
        Returns same format as RandomBaseline
        """
        history = []
        current_seq = wt_sequence
        current_fitness = self.oracle.score_sequence(wt_sequence)
        queries_used = 1
        
        history.append((current_seq, current_fitness))
        
        while queries_used < budget:
            # Get all k-point neighbors
            neighbors = get_neighbors(current_seq, k=self.k)
            
            # Score all neighbors (batch if possible)
            neighbor_scores = [
                self.oracle.score_sequence(n) for n in neighbors
            ]
            queries_used += len(neighbors)
            
            # Pick best
            best_idx = np.argmax(neighbor_scores)
            best_neighbor = neighbors[best_idx]
            best_neighbor_fitness = neighbor_scores[best_idx]
            
            # Record all evaluations
            for n, f in zip(neighbors, neighbor_scores):
                history.append((n, f))
            
            # Accept if better (greedy)
            if best_neighbor_fitness > current_fitness:
                current_seq = best_neighbor
                current_fitness = best_neighbor_fitness
            else:
                # Stuck in local optimum
                break
        
        return {
            'best_sequence': current_seq,
            'best_fitness': current_fitness,
            'history': history,
            'queries_used': queries_used
        }
```

**Tasks**:
- [ ] Implement neighbor generation
- [ ] Implement greedy selection
- [ ] Handle local optima (early stopping)
- [ ] Test on one dataset
- [ ] Compare to random baseline

**Estimated Time**: 3-4 hours

---

### 5. Implement Simulated Annealing
**File**: `src/baselines/simulated_annealing.py`

```python
class SimulatedAnnealing:
    """
    Simulated annealing baseline
    """
    def __init__(self, oracle, k=1, temp_init=1.0, temp_final=0.01, seed=42):
        self.oracle = oracle
        self.k = k
        self.temp_init = temp_init
        self.temp_final = temp_final
        self.rng = np.random.RandomState(seed)
    
    def optimize(self, wt_sequence: str, budget: int = 500) -> Dict:
        """
        Simulated annealing search
        """
        history = []
        current_seq = wt_sequence
        current_fitness = self.oracle.score_sequence(wt_sequence)
        best_seq = current_seq
        best_fitness = current_fitness
        queries_used = 1
        
        for i in range(budget - 1):
            # Temperature schedule (linear annealing)
            temp = self._temperature(i, budget)
            
            # Generate random neighbor
            neighbor = self._random_neighbor(current_seq)
            neighbor_fitness = self.oracle.score_sequence(neighbor)
            queries_used += 1
            
            history.append((neighbor, neighbor_fitness))
            
            # Acceptance criterion
            delta = neighbor_fitness - current_fitness
            if delta > 0 or self.rng.random() < np.exp(delta / temp):
                current_seq = neighbor
                current_fitness = neighbor_fitness
                
                if current_fitness > best_fitness:
                    best_seq = current_seq
                    best_fitness = current_fitness
        
        return {
            'best_sequence': best_seq,
            'best_fitness': best_fitness,
            'history': history,
            'queries_used': queries_used
        }
    
    def _temperature(self, step: int, total_steps: int) -> float:
        # Linear annealing schedule
        alpha = step / total_steps
        return self.temp_init * (1 - alpha) + self.temp_final * alpha
    
    def _random_neighbor(self, sequence: str) -> str:
        # Generate random k-point mutant
        pass
```

**Tasks**:
- [ ] Implement temperature schedule
- [ ] Implement acceptance criterion
- [ ] Test different temp schedules
- [ ] Compare to greedy and random

**Estimated Time**: 3-4 hours

---

### 6. Create Experiment Runner
**File**: `src/utils/experiment_runner.py`

```python
def run_single_experiment(
    method,
    dataset_name: str,
    k: int,
    budget: int,
    seed: int,
    output_dir: str
) -> Dict:
    """
    Run a single experiment configuration
    
    Args:
        method: Optimization method instance
        dataset_name: Name of DMS dataset
        k: Number of simultaneous mutations
        budget: Oracle query budget
        seed: Random seed
        output_dir: Where to save results
    
    Returns:
        Results dictionary
    """
    # Load wild-type
    wt_seq = load_wild_type(dataset_name)
    
    # Run optimization
    results = method.optimize(wt_seq, budget=budget)
    
    # Save results
    save_results(results, output_dir, dataset_name, k, seed)
    
    return results

def run_experiment_suite(
    methods: List,
    datasets: List[str],
    k_values: List[int],
    seeds: List[int],
    budget: int = 500
):
    """
    Run full experimental suite
    
    Parallelizes across seeds if possible
    """
    pass
```

**Tasks**:
- [ ] Implement single experiment runner
- [ ] Add result saving (JSON, CSV)
- [ ] Add progress tracking
- [ ] Test on 1 dataset × 1 method × 1 seed

**Estimated Time**: 2-3 hours

---

### 7. Implement Evaluation Metrics
**File**: `src/utils/evaluation.py`

```python
def compute_metrics(results: Dict, ground_truth_optimum: float = None) -> Dict:
    """
    Compute evaluation metrics
    
    Args:
        results: Output from optimization method
        ground_truth_optimum: If known (from DMS data)
    
    Returns:
        {
            'best_fitness': float,
            'queries_to_best': int,
            'regret': float (if optimum known),
            'auc': float (area under fitness curve)
        }
    """
    pass

def plot_sample_efficiency(results_list: List[Dict], output_path: str):
    """
    Plot fitness vs queries for multiple runs
    
    Shows mean ± std across seeds
    """
    pass

def compare_methods(
    method_results: Dict[str, List[Dict]], 
    output_dir: str
):
    """
    Statistical comparison between methods
    
    - Wilcoxon signed-rank test
    - Effect sizes
    - Visualization
    """
    pass
```

**Tasks**:
- [ ] Implement basic metrics
- [ ] Implement sample efficiency plots
- [ ] Implement statistical tests
- [ ] Test on dummy data

**Estimated Time**: 3-4 hours

---

## Week 3 Checklist

### Monday-Tuesday (Dec 9-10)
- [ ] Implement ESM2Oracle class (2-3 hrs)
- [ ] Implement mutation operators (3-4 hrs)
- [ ] Write unit tests for mutations (1 hr)

### Wednesday-Thursday (Dec 11-12)
- [ ] Implement Random baseline (2-3 hrs)
- [ ] Implement Greedy baseline (3-4 hrs)
- [ ] Test both on 1 dataset (1-2 hrs)

### Friday-Saturday (Dec 13-14)
- [ ] Implement Simulated Annealing (3-4 hrs)
- [ ] Implement experiment runner (2-3 hrs)
- [ ] Run small test experiment (1 method × 1 dataset × 3 seeds) (1-2 hrs)

### Sunday (Dec 15)
- [ ] Implement evaluation metrics (3-4 hrs)
- [ ] Generate plots for test experiment (1-2 hrs)
- [ ] Debug any issues (buffer time)

---

## Testing Strategy

### Unit Tests (`tests/`)

```python
# tests/test_mutations.py
def test_single_mutation():
    seq = "MKTAYIAK"
    mutant = apply_mutation(seq, "K1A")
    assert mutant == "MATAYIAK"

def test_multiple_mutations():
    seq = "MKTAYIAK"
    mutant = apply_mutation(seq, "K1A:T2V")
    assert mutant == "MAVAYIAK"

# tests/test_oracle.py
def test_oracle_scoring():
    oracle = ESM2Oracle()
    wt_seq = "MKTAYIAK"
    score = oracle.score_sequence(wt_seq)
    assert isinstance(score, float)

# tests/test_baselines.py
def test_random_baseline():
    oracle = MockOracle()
    baseline = RandomBaseline(oracle, k=1, seed=42)
    results = baseline.optimize("MKTAYIAK", budget=10)
    assert results['queries_used'] == 10
    assert 'best_fitness' in results
```

### Integration Tests

Run on **smallest dataset** (DN7A_SACS2, 1008 variants):
- 3 methods (Random, Greedy, SA)
- k=1 only
- 3 seeds
- budget=100 (small for speed)

**Total**: 9 runs, should complete in ~30 minutes

---

## File Organization

After Week 3, your structure should be:

```
src/
├── models/
│   ├── __init__.py
│   └── esm_oracle.py          ✓ NEW
├── baselines/
│   ├── __init__.py
│   ├── random_baseline.py     ✓ NEW
│   ├── greedy.py              ✓ NEW
│   └── simulated_annealing.py ✓ NEW
└── utils/
    ├── mutations.py            ✓ NEW
    ├── experiment_runner.py    ✓ NEW
    └── evaluation.py           ✓ NEW

tests/
├── test_mutations.py           ✓ NEW
├── test_oracle.py              ✓ NEW
└── test_baselines.py           ✓ NEW

experiments/
└── baselines_test/             ✓ NEW
    ├── results/
    └── plots/
```

---

## Success Criteria for Week 3

By end of week, you should have:

1. ✅ **Working ESM-2 oracle** that can score sequences
2. ✅ **3 baseline methods** (Random, Greedy, SA) fully implemented
3. ✅ **Unit tests** passing for all components
4. ✅ **Test experiment** completed on 1 dataset
5. ✅ **Plots** showing sample efficiency curves
6. ✅ **Code review**: Clean, documented, ready for Week 4 RL implementation

---

## Tips & Best Practices

### Code Quality
- Write docstrings for all functions
- Use type hints: `def foo(x: int) -> str:`
- Keep functions < 50 lines
- Extract repeated code into utilities

### Debugging
- Start with small examples (short sequences, small budgets)
- Print intermediate values
- Use assertions liberally
- Test each component independently before integration

### Performance
- Batch oracle calls when possible
- Cache sequence scores (use dict with sequence as key)
- Profile code to find bottlenecks
- Consider using numba for speed-critical loops

### Version Control
- Commit after each major component works
- Use meaningful commit messages
- Tag releases: `git tag v0.1-baselines`

---

## Questions to Answer This Week

1. How many k-point mutant neighbors exist?
   - k=1: 19 × L (where L = sequence length)
   - k=3: C(L, 3) × 19³ (combinatorially large!)
   - **Decision**: For k>1, sample neighbors instead of exhaustive?

2. Should greedy explore all neighbors or sample?
   - All neighbors: Slow for k>1
   - Sample neighbors: Faster but less optimal
   - **Recommendation**: Sample for k≥3

3. What temperature schedule for SA?
   - Linear: temp(t) = T_0 (1 - t/T_max)
   - Exponential: temp(t) = T_0 × α^t
   - **Try both**, compare empirically

4. How to handle query budget for greedy?
   - Each iteration uses 19×L queries (for k=1)
   - May exceed budget quickly
   - **Solution**: Track queries, stop when budget reached

---

## Resources

### Code Examples
- ESM-2 usage: https://github.com/facebookresearch/esm
- Hill climbing: Standard algorithm, easy to find examples
- Simulated annealing: https://en.wikipedia.org/wiki/Simulated_annealing

### Papers to Reference
- Greedy for proteins: "Directed evolution" literature
- Simulated annealing: Original Kirkpatrick et al. (1983)

---

**Next Document**: After Week 3 completes, create `WEEK4_RL_IMPLEMENTATION.md`

---

**Last Updated**: December 7, 2025
