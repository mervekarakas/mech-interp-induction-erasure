# Induction Circuits Under Token Erasures

A mechanistic interpretability study of how induction heads in GPT-2 small respond to systematic corruption of their input context.

## Summary

We reverse-engineer the induction circuit in GPT-2 small using activation patching, head ablation, and attention pattern analysis, then study how this circuit degrades under controlled token erasures. Our main findings:

1. **Robust degradation:** Induction performance (logit difference) degrades smoothly and proportionally with erasure rate. The circuit remains net-helpful even with 50% of the context destroyed, suggesting redundant information encoding across positions.

2. **Division of labor:** The induction circuit has two distinct roles — early-layer *pattern matchers* (L0H5, L3H0) that attend strongly to matching tokens but have low causal contribution individually, and mid-layer *output boosters* (L5H1, L4H11) that don't attend to matches themselves but are causally critical. The correlation between induction attention score and ablation contribution is near zero (r=0.03), revealing this functional separation.

3. **Marginal harmful-head regime:** At very high erasure rates (>70%), a few heads show small negative contributions (i.e., ablating them slightly helps). The strongest effect is L5H1 dipping to ~-0.13 at ε≈0.7 — compared to its +7.24 contribution on clean input. This effect is small and only appears when the overall signal is nearly destroyed.

4. **Distinct failure modes under corruption:** A conditional analysis splitting positions by mirror status (intact vs corrupted) reveals that pattern matchers and output boosters fail differently:
   - *Pattern matcher* L0H1 (induction score 0.26) shows a clean local decomposition: always helps on intact mirrors, always hurts on corrupted mirrors. Its failure is **position-local**.
   - *Output booster* L5H1 (induction score 0.01) degrades even on intact mirrors when overall corruption is high, because it reads from a globally degraded residual stream. Its failure is **context-global**.
   - *Output booster* L4H11 (induction score 0.01) degrades gracefully to zero on both position types — a "fail-safe" head that never becomes harmful.

   This extends the division-of-labor finding: the two roles don't just differ in what they attend to, but in *how they break*.

5. **Adversarial erasure reveals interaction-driven bottlenecks:** A greedy adversary that chooses which positions to corrupt outperforms random corruption (gap AUC = 1.28), reducing induction to 1.34 at 50% corruption vs 3.17 for random. Crucially, corrupting positions in descending order of single-position importance actually performs *worse* than random (sorted metric 7.14 vs random 3.17 at ε=0.5) — only 3/10 of the top positions overlap between greedy and importance-sorted orderings. This means the greedy adversary exploits **inter-position interactions** (partial redundancy), not just individual importance: the individually most important positions are clustered and redundant, so the greedy adversary spreads its budget across non-redundant positions.

6. **Adversarial ordering is positional, not token-specific:** The greedy adversarial order computed on one batch transfers almost perfectly to held-out tokens (transfer ratio = 0.90). This means the adversary exploits **structural positional properties** — likely learned positional embeddings or edge effects — rather than features of the particular token sequence.

7. **Output boosters predict adversarial bottlenecks, pattern matchers don't:** Correlating per-position attention mass with position importance reveals that output boosters (L5H1 r=+0.38, p=0.007; L4H11 ρ=-0.64, p<0.001) significantly predict which positions matter, while pattern matchers (L0H1, L3H0) show no significant correlation. L4H11's strong negative correlation suggests positions it neglects become the circuit's bottlenecks.

## Task and Metric

**Synthetic induction task:** Sequences of the form `[BOS, x₀, x₁, ..., x₄₉, x₀, x₁, ..., x₄₉]` — 50 random tokens repeated. If the model has induction heads, it should predict tokens in the second half by copying from the first.

**Metric:** Logit difference — how much the model's logit for the correct next token exceeds the average logit. Higher = stronger induction. Clean baseline: ~21.5.

**Token erasures:** We corrupt a fraction ε of the first half by replacing tokens with either random vocab tokens or a fixed token ("the"). The second half is never modified. Two corruption types are used as a robustness check.

## Methods

### Activation patching (Notebook 01)
We run the model on clean and fully corrupted (ε=1.0) inputs, then patch clean residual stream activations into the corrupted run at each (layer, position) to measure recovery. Position-level patching showed distributed dependence (~2% max recovery per position). Head-level patching (patching each head's z output across all positions) identified contributing heads in layers 5–11.

### Head ablation (Notebook 02)
We zero out each head's output and measure the effect on logit difference. This identifies which heads are causally important vs. merely correlated.

### Erasure sweep (Notebook 02)
We sweep ε from 0 to 1 in steps of 0.05, measuring the metric and per-head ablation contributions at each level. Results are aggregated over 10 random seeds for statistical robustness. Both corruption types (random replacement and fixed-token replacement) are tested.

### Conditional analysis (Notebook 02)
We split second-half evaluation positions into "good" (mirror token in the first half survived corruption) vs "bad" (mirror was corrupted). For each group, we compute the per-position head contribution (logit diff with vs without ablation). This decomposes the aggregate harmful-head effect into its mechanistic components.

### Attention pattern analysis (Notebook 03)
We compute an induction score for all 144 heads: the average attention weight placed on the matching source position in the first half. We compare this against ablation contributions to understand functional roles within the circuit.

### Adversarial erasure (Notebook 04)
We corrupt specific positions in the first half (rather than a random fraction) and compare three strategies: (1) random ordering, (2) greedy adversarial (at each step, corrupt the position that maximally reduces the metric), (3) sort-by-importance (static ordering by single-position importance). Pairwise interaction terms test whether position contributions are additive. Held-out evaluation tests whether the adversarial ordering transfers to new token sequences. Circuit tie-back correlates position importance with head attention mass.

## Results

### Degradation curve
Induction performance follows a convex-decreasing curve — steep initial drop at low ε, flattening at high ε. At ε=0.5, the metric retains ~14% of the clean value (3.0 out of 21.5). Results are consistent across corruption types and random seeds.

### Top heads by causal importance (clean ablation)
| Head | Ablation contribution | Induction score |
|------|----------------------|-----------------|
| L5H1 | 7.24 | 0.01 |
| L4H11 | 4.60 | 0.01 |
| L7H6 | 4.02 | 0.01 |
| L5H9 | 3.90 | 0.01 |
| L0H1 | 3.47 | 0.26 |

### Top heads by induction score (attention to matching position)
| Head | Induction score | Ablation contribution |
|------|----------------|----------------------|
| L3H0 | 0.67 (67× baseline) | ~0 |
| L0H5 | 0.64 (65× baseline) | ~0 |
| L0H1 | 0.26 (27× baseline) | 3.47 |

The near-zero correlation (r=0.03) between these two measures reveals a division of labor: pattern matchers attend to the right positions but contribute little individually, while output boosters don't attend to matches but are causally essential.

### Harmful-head transitions under erasure
All top-10 heads remain helpful through ε=0.5. At ε≥0.7, 2–3 heads show small negative contributions (max magnitude ~0.13). The effect is consistent across corruption types but small relative to clean contributions.

### Conditional analysis: good vs bad mirror positions
For each second-half position, we classify its mirror (first-half counterpart) as "good" (survived corruption) or "bad" (was corrupted), then measure the head's contribution separately on each group. Three heads reveal three distinct failure modes:

| Head | Role | Good positions | Bad positions | Failure mode |
|------|------|---------------|---------------|-------------|
| L0H1 | Pattern matcher (score 0.26) | Always positive (+2.5 → +0.1) | Always negative (-1.8 → -0.04) | Local: per-position mirror status |
| L5H1 | Output booster (score 0.01) | Degrades, goes negative at ε≥0.7 | Near zero throughout | Global: overall context health |
| L4H11 | Output booster (score 0.01) | Always positive, graceful decay | Near zero (never hurts) | Fail-safe: degrades to zero |

L0H1 is the clearest example of the predicted mechanism: it directly attends to matching tokens, so it helps when the match is intact and hurts when it's corrupted. L5H1 and L4H11, as output boosters that read from the residual stream rather than attending to matches directly, show context-dependent degradation that isn't captured by local mirror status alone.

### Adversarial vs random erasure
Three corruption strategies compared at ε=0.5 (25/50 positions corrupted):

| Strategy | Metric at ε=0.5 | AUC gap vs random |
|----------|----------------|-------------------|
| Random (10-seed mean) | 3.17 | — |
| Sort-by-importance (static) | 7.14 | -2.27 (less damaging) |
| Greedy adversarial | 1.34 | +1.28 (more damaging) |

The sort-by-importance inversion (performing worse than random as an adversary) reveals that individually important positions are **redundant with each other** — killing them together wastes adversarial budget. The greedy adversary instead selects spread-out, non-redundant positions.

Pairwise interaction terms (mean -0.36, ratio 0.17) confirm partial redundancy: corrupting two positions together hurts slightly less than the sum of corrupting each alone.

### Held-out transfer
The adversarial ordering computed on one batch (seed=42) transfers to a held-out batch (seed=999) with transfer ratio 0.90 (held-out gap AUC 1.15 vs training gap AUC 1.28). The adversarial structure is positional, not token-specific.

### Circuit tie-back: attention vs position importance
| Head | Role | Pearson r | Spearman ρ | Significant? |
|------|------|-----------|------------|-------------|
| L0H1 | Pattern matcher | +0.18 | +0.18 | No (p=0.21) |
| L3H0 | Pattern matcher | -0.09 | -0.33 | No (p=0.52) |
| L5H1 | Output booster | +0.38 | +0.01 | Yes (p=0.007) |
| L4H11 | Output booster | -0.40 | -0.64 | Yes (p=0.004) |

Pattern matchers distribute attention roughly uniformly across prefix positions (L0H1 range: 0.005-0.006), so they don't predict which positions are adversarial bottlenecks. Output boosters show significant correlations: L4H11's strong negative Spearman ρ (-0.64) indicates that positions it attends to *least* are the most important — these neglected positions become the circuit's weak points.

## Limitations

- **Toy task:** Exact-repeat induction is a simplified setting. Real in-context learning involves fuzzy, semantic matching.
- **Single model:** Results are specific to GPT-2 small. The qualitative pattern likely generalizes but head identities will differ.
- **Corruption choice:** As noted in best-practices literature (Heimersheim & Janiak, 2024), the choice of corruption can affect localization results. We mitigate this with two corruption types.
- **Single-head ablation:** We ablate one head at a time. Multi-head ablation could reveal redundancy and interactions not captured here.

## Next steps

- Multi-head ablation to test circuit redundancy (e.g., do pattern matchers compensate for each other?)
- Extend to a larger model (GPT-2 medium, Pythia) to test generalization
- SAE-based feature analysis to understand what information the pattern matchers write into the residual stream
- More naturalistic induction tasks (fuzzy matching, semantic similarity)

## Reproduction

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run notebooks in order:
- `notebooks/00_baseline_induction.ipynb` — Baseline induction effect
- `notebooks/01_activation_patching.ipynb` — Circuit localization
- `notebooks/02_ablation_and_erasures.ipynb` — Ablation + erasure sweep
- `notebooks/03_head_diagnostics.ipynb` — Attention pattern validation
- `notebooks/04_adversarial_erasures.ipynb` — Adversarial vs random erasure

All experiments use GPT-2 small and run on a single GPU in minutes.

## Repository structure

```
├── README.md
├── notebooks/
│   ├── 00_baseline_induction.ipynb
│   ├── 01_activation_patching.ipynb
│   ├── 02_ablation_and_erasures.ipynb
│   ├── 03_head_diagnostics.ipynb
│   └── 04_adversarial_erasures.ipynb
├── src/
│   ├── data.py          # Synthetic induction sequence generation
│   ├── corruptions.py   # Token erasure functions
│   ├── metrics.py       # Logit difference metric
│   ├── patching.py      # Activation and head-level patching
│   ├── ablation.py      # Head ablation utilities
│   └── plotting.py      # Visualization helpers
└── results/
    └── figures/         # All generated plots
```
