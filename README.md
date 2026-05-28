# ALNS for Heterogeneous Fleet VRPTW (HFVRPTW) — Solomon C20 / R20

A complete Adaptive Large Neighborhood Search implementation for the
**Heterogeneous Fleet Vehicle Routing Problem with Time Windows**, with
**soft time windows** (late arrivals penalized linearly by lateness).

---

## 1. Problem Setup

| Parameter | Value |
|-----------|-------|
| **Depot** | Node 0 (Solomon coordinates) |
| **Customers** | First 20 of Solomon C101 (clustered) and R101 (random) |
| **Fleet — Large (L)** | Q = 200, F = 200, α = 1.0, max 3 vehicles |
| **Fleet — Small (S)** | Q = 100, F = 100, α = 1.5, max 4 vehicles |
| **Lateness penalty** | λ = 50 per unit of late time |
| **Capacity / fleet-size** | Hard constraints |

### Cost Function

```
cost(S) = Σ_routes [ F_v + α_v · dist(R) ]       (vehicle + variable cost)
        + λ_late · Σ_customers max(0, w_i − b_i)  (lateness penalty)
```

---

## 2. Algorithm Components

### 2.1 Initial Solution Constructors (5 variants)

| Constructor | Description |
|-------------|-------------|
| `type_aware` | Vehicle-type-aware sequential insertion (prefers L) |
| `clarke_wright` | Classic Clarke-Wright savings |
| `clarke_wright_tw_simple` | CW with temporal compatibility penalty |
| `clarke_wright_tw_full` | CW with full cost-delta evaluation |
| `random` | Random customer insertion |

### 2.2 ALNS Operators

| Destroy (5) | Repair (4) |
|-------------|------------|
| Random removal | Greedy insertion |
| Worst removal | Regret-2 insertion |
| Shaw/related removal | Regret-3 insertion |
| Route removal | Vehicle-type-aware insertion |
| Vehicle-type-aware removal | |

### 2.3 Adaptive Selection & Acceptance

- **Selection:** Roulette-wheel with segment-based weight updates (Θ = 100)
- **Scores:** σ₁ = 33 (new best), σ₂ = 9 (better current), σ₃ = 13 (accepted worse)
- **Acceptance:** Simulated annealing, geometric cooling
- **T₀ calibration:** Warm-up sampling (30 random destroy/repair moves)
- **Stopping:** `min(2000 iterations, 60 seconds)`

### 2.4 Tuned Parameters

| Parameter | Value | Sensitivity |
|-----------|-------|-------------|
| `cooling_rate` | 0.998 | **High** — 0.995 too aggressive, 0.999 too slow |
| `segment_length` | 100 | Low |
| `eta_max_frac` | 0.40 | Medium — 0.25 underperforms |
| `reaction_factor` | 0.10 | Low |

---

## 3. Alternative Heuristics (Baselines)

| Method | Description |
|--------|-------------|
| **Standalone SA** | Simulated Annealing with 2-opt*, relocate, or-opt, intra-2-opt, swap; CW-TW init; 50K iter |
| **Nearest Neighbour (NN)** | Greedy construction with TW-aware tie-breaking |
| **CW-TW** | Clarke-Wright TW-simple construction only |
| **Greedy** | Type-aware sequential insertion only |

---

## 4. Project Layout

```
alns_hfvrptw/
├── data/
│   ├── C20.txt                 # Solomon C20 (20 customers + depot)
│   └── R20.txt                 # Solomon R20
├── src/
│   ├── instance.py             # Solomon parser, FleetConfig, distance matrix
│   ├── solution.py             # Route, Solution, soft-TW cost, feasibility
│   ├── construct.py            # 5 initial solution constructors
│   ├── operators.py            # 5 destroy + 4 repair operators
│   ├── alns.py                 # Main ALNS loop, adaptive weights, SA acceptance
│   └── baselines.py            # Alternative heuristics (SA, NN, CW-only)
├── results/                    # All generated outputs
├── run_final.py                # Final ALNS run (type_aware, 2000 iter / 60s)
├── benchmark.py                # ALNS vs alternatives comparison
├── parameter_tuning.py         # Parameter sensitivity analysis
├── compare_initials.py         # Initial constructor comparison
├── generate_report.py          # LaTeX tables + text summary generator
├── run_experiments.py           # Full experiment run
├── requirements.txt
└── README.md
```

---

## 5. How to Run

```bash
pip install -r requirements.txt

# Full pipeline (recommended)
python run_final.py             # Final ALNS solution (3 seeds)
python benchmark.py             # ALNS vs SA vs NN vs CW vs Greedy
python parameter_tuning.py      # Sensitivity analysis
python generate_report.py       # Generate LaTeX tables & summary

# Individual experiments
python compare_initials.py      # Compare 5 initial constructors
python run_experiments.py       # Detailed ALNS run with operator stats
```

---

## 6. Results

### 6.1 Final ALNS (type_aware, 2000 iter / 60s, 3 seeds)

| Instance | Best | Mean | Worst | Lateness | L used | S used | Time |
|----------|-----:|-----:|------:|---------:|:------:|:------:|-----:|
| **C20** | **650.23** | 650.72 | 651.71 | 0.00 | 1 / 3 | 2 / 4 | ~3.5s |
| **R20** | **1461.86** | 1461.86 | 1461.86 | 0.00 | 2 / 3 | 4 / 4 | ~5.0s |

### 6.2 Benchmark — ALNS vs Alternatives

| Instance | Method | Best | Mean | Lateness | Time |
|----------|--------|-----:|-----:|---------:|-----:|
| C20 | **ALNS** | **650** | **651** | **0.00** | 3.5s |
| C20 | SA | 685 | 685 | 0.00 | 0.7s |
| C20 | CW-TW | 685 | 685 | 0.00 | <0.01s |
| C20 | Greedy | 107,450 | 107,450 | 2,136 | <0.01s |
| C20 | NN | 298,232 | 298,232 | 5,952 | <0.01s |
| R20 | **ALNS** | **1,462** | **1,462** | **0.00** | 5.0s |
| R20 | SA | 8,126 | 8,126 | 137 | 0.8s |
| R20 | CW-TW | 8,126 | 8,126 | 137 | <0.01s |
| R20 | Greedy | 124,557 | 124,557 | 2,474 | <0.01s |
| R20 | NN | 294,247 | 294,247 | 5,868 | <0.01s |

### 6.3 Best Routes

**C20** (cost = 650.23):
```
[L] load=190/200  dist= 95.88  | 0 → 13 → 17 → 18 → 19 → 15 → 16 → 14 → 12 → 0
[S] load=100/100  dist= 61.57  | 0 → 20 → 11 →  9 →  6 →  4 →  2 →  1 → 0
[S] load= 70/100  dist= 41.33  | 0 →  5 →  3 →  7 →  8 → 10 → 0
```

**R20** (cost = 1461.86):
```
[L] load=60/200  dist=145.00  | 0 → 14 → 15 →  3 →  4 → 0
[S] load=71/100  dist= 68.08  | 0 →  5 → 16 →  6 → 13 → 0
[S] load=16/100  dist= 77.76  | 0 →  7 →  8 → 17 → 0
[L] load=45/200  dist= 81.11  | 0 → 11 → 19 → 10 → 0
[S] load=54/100  dist= 83.40  | 0 → 12 →  9 → 20 →  1 → 0
[S] load=19/100  dist= 61.27  | 0 →  2 → 18 → 0
```

---

## 7. Key Findings

1. **ALNS dramatically outperforms standalone SA** — destroy/repair operators
   can restructure entire routes, while SA's small-neighbourhood moves
   (swap, relocate, 2-opt) get trapped in local optima.

2. **CW-TW is the best construction heuristic** — for C20 it reaches 685
   (only 5% above ALNS optimal); SA with 50K iterations cannot improve it.

3. **ALNS achieves 0 lateness** on all instances — all time windows satisfied
   without soft-TW penalties.

4. **Parameter sensitivity:** `cooling_rate` has the strongest impact;
   0.997–0.998 optimal for 2000 iterations. Other parameters are robust.

5. **Convergence is fast:** 2000 iterations (~3-5s) sufficient for n=20.
   The 60-second time limit is never reached.

6. **Initial solution quality matters less with enough iterations** — all 5
   constructors converge to similar final costs when ALNS has budget.

---

## 8. Output Files

| File | Description |
|------|-------------|
| `results/final_*_routes.png` | Route visualization (best solution) |
| `results/final_*_convergence.png` | Convergence curves (3 seeds) |
| `results/benchmark_*.png` | ALNS vs alternatives (4-panel) |
| `results/tuning_*.png` | Parameter sensitivity (4 parameters) |
| `results/final_summary.json` | Detailed JSON results |
| `results/benchmark.json` | Benchmark comparison data |
| `results/parameter_tuning.json` | Tuning experiment data |
| `results/report_tables.tex` | LaTeX tables for embedding |
| `results/report_summary.txt` | Plain-text executive summary |
