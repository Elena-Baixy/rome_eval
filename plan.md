# Plan
## Objective
Analyze the storage and recall of factual associations in autoregressive transformer language models, finding evidence that these associations correspond to localized, directly-editable computations.

## Hypothesis
1. Factual associations in GPT correspond to a localized computation mechanism where each midlayer MLP module accepts inputs encoding a subject, then produces outputs recalling memorized properties about that subject, with middle layer MLP outputs accumulating information that is copied to the last token by attention at high layers.
2. Factual associations are localized in the MLP modules at specific middle layers, specifically at the processing of the subject's last token.
3. MLP layers in transformers can be modeled as linear associative memory where weights act as key-value stores.

## Methodology
1. Develop a causal intervention method (Causal Tracing) using causal mediation analysis to identify neuron activations that are decisive in a model's factual predictions by running the network with corrupted subject embeddings and selectively restoring hidden states.
2. Modify feed-forward weights using Rank-One Model Editing (ROME) to update specific factual associations. ROME inserts a new key-value association into a single MLP layer by computing a rank-one weight update that minimizes interference with existing memories.
3. Evaluate ROME on both a standard zero-shot relation extraction (zsRE) benchmark and a new COUNTERFACT dataset of difficult counterfactual assertions, measuring efficacy, generalization (paraphrase), and specificity (neighborhood).
4. Experimental setting: GPT-2 XL (1.5B parameters) and GPT-J (6B parameters) autoregressive transformer models tested on factual statements and counterfactual edits.

## Experiments
### Causal Tracing of Factual Associations
- What varied: Layer and token position of hidden state mediators; corruption applied to subject embeddings
- Metric: Average Indirect Effect (AIE) measuring contribution of hidden states to factual prediction restoration
- Main result: MLP modules at middle layers (around layer 15-18) at the last subject token have strong causal effects (AIE=6.6% for MLP vs 1.6% for attention at early site), revealing a distinct early site in middle-layer feed-forward modules.

### ROME Evaluation on Zero-Shot Relation Extraction (zsRE)
- What varied: Model editing method (ROME vs FT, FT+L, KE, MEND, KE-zsRE, MEND-zsRE)
- Metric: Efficacy, Paraphrase accuracy, Specificity on 10,000 zsRE records
- Main result: ROME achieves 99.8% efficacy and 88.1% paraphrase accuracy with maintained specificity (24.2%), competitive with hypernetwork methods despite simplicity.

### ROME Layer and Token Sweep on COUNTERFACT
- What varied: Target layer (0-47) and token position for ROME intervention
- Metric: Efficacy (EM), Generalization (PM), Specificity (NM), Score (S)
- Main result: Performance peaks at middle layers (around layer 18) at the last subject token, confirming causal analysis. Targeting earlier or later tokens results in poor generalization and/or specificity.

### ROME Evaluation on COUNTERFACT Dataset (GPT-2 XL)
- What varied: Model editing method (ROME vs FT, FT+L, KN, KE, KE-CF, MEND, MEND-CF) on counterfactual assertions
- Metric: Score (S), Efficacy Score/Magnitude (ES/EM), Paraphrase Score/Magnitude (PS/PM), Neighborhood Score/Magnitude (NS/NM), Generation Entropy (GE), Reference Score (RS)
- Main result: ROME achieves best overall Score (89.2) with 100% efficacy, 96.4% paraphrase success, and 75.4% neighborhood preservation. Other methods exhibit overfitting (FT: high generalization but 40.4% specificity) or underfitting (FT+L, KE, MEND).

### ROME Evaluation on COUNTERFACT Dataset (GPT-J)
- What varied: Model editing method (FT, FT+L, MEND, ROME) on GPT-J 6B model
- Metric: Score (S), Efficacy, Generalization, Specificity metrics on 2000-record test set
- Main result: ROME achieves Score of 91.5 with 99.9% efficacy, 99.1% paraphrase success, and 78.9% neighborhood preservation. FT shows model damage with only 10.3% neighborhood success.

### Human Evaluation of Generated Text Quality
- What varied: Model editing method (ROME vs FT+L) on 50 counterfactual scenarios
- Metric: Human rater judgments on consistency with counterfactual and fluency (15 volunteers, 150 evaluations)
- Main result: ROME rated 1.8 times more likely to be consistent with inserted fact than FT+L, but 1.3 times less likely to be more fluent, suggesting some fluency loss not captured by automated metrics.