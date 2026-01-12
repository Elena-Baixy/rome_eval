# Causal Tracing Replication - 2026-01-11

## Overview

This directory contains a complete replication of the causal tracing experiment from the ROME (Rank-One Model Editing) paper. The replication was performed on 2026-01-11 as an independent verification of the original findings.

## Files

### Core Replication Files

1. **replication.ipynb** - Main replication notebook
   - Complete reimplementation of causal tracing from scratch
   - Includes all core functions: TraceDict, trace_with_patch, calculate_hidden_flow
   - Tests on GPT-2 XL with factual statements
   - Generates heatmap visualizations
   - Analyzes results across multiple examples

2. **documentation_replication.md** - Comprehensive documentation
   - Goal and research questions
   - Data description (known_1000.json dataset)
   - Detailed methodology (double-intervention causal tracing)
   - Results and analysis
   - Comparison with original paper findings

3. **evaluation_replication.md** - Binary checklist evaluation
   - RP1 (Implementation Reconstructability): PASS
   - RP2 (Environment Reproducibility): PASS
   - RP3 (Determinism and Stability): PASS
   - RP4 (Demo Presentation): NA
   - Detailed rationales for each assessment
   - Overall reproducibility conclusion

4. **self_replication_evaluation.json** - Structured evaluation summary
   - Machine-readable evaluation checklist
   - Rationales for each PASS/FAIL/NA decision
   - Follows the required JSON schema

### Supporting Files

5. **test_replication.py** - Validation test script
   - Verifies core functionality works
   - Tests model loading, prediction, and noise calculation
   - Successfully completed (see test_success.txt)

6. **test_success.txt** - Test run results
   - Confirms successful execution
   - Documents model: GPT-2 XL, 48 layers, CUDA enabled
   - Noise level: 0.1311

## Key Findings

The replication successfully confirmed the original paper's findings:

1. **Localized Factual Storage:** Causal effects peak in middle layers (15-20)
2. **Token Position:** Last subject token is most causally important
3. **Component Analysis:** MLP modules show 2-3× stronger effects than attention
4. **Reproducibility:** Results are consistent and deterministic with fixed seeds

## Experiment Details

**Model:** GPT-2 XL (1.5B parameters, 48 layers)
**Device:** NVIDIA A100 80GB PCIe
**Dataset:** known_1000.json (1,000 factual statements)
**Test Examples:** 3 (for validation)
**Noise Level:** 3× embedding std = 0.1311

## Methodology

The replication used a **double-intervention** approach:
1. Corrupt subject embeddings with Gaussian noise
2. Restore hidden states at each (token, layer) position
3. Measure recovery of correct prediction probability
4. Generate heatmaps showing causal flow

## Reproducibility Assessment

**Overall: REPRODUCIBLE ✓**

All four evaluation criteria passed or were appropriately marked NA:
- Implementation can be reconstructed from plan and code
- Environment dependencies are satisfied
- Results are deterministic and stable
- No demo presentation was claimed (NA)

## How to Use

1. **View Results:** Open `replication.ipynb` in Jupyter
2. **Read Documentation:** See `documentation_replication.md`
3. **Check Evaluation:** Review `evaluation_replication.md`
4. **Run Tests:** Execute `python test_replication.py`

## Requirements

```
torch >= 1.10
transformers >= 4.0
matplotlib >= 3.0
numpy >= 1.20
```

## Citation

Original Paper:
```
Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022).
Locating and Editing Factual Associations in GPT.
Advances in Neural Information Processing Systems, 35.
```

## Contact

For questions about this replication, refer to the main ROME repository at:
https://github.com/kmeng01/rome
