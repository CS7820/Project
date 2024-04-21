# Introduction 
KGs allow for representing structured data providing a source of information.  <!-- (Type another 1-2 sentences) -->

KGE combine ML techniques to allow KGs with the ability to infer knowledge that are not explicitly stated in the KG.

## Problem Statement 
KGEs suffer from susceptibility similar to a machine learning model through AA/Perturbations of the data sources to trained models, in this case the KG.

... something later...

## State-of-the-Art Analysis/Related Work

## Research Contributions
AA on a KG has already been investigated and these reports reveal negative impacts on KGE performances to which affect KGE models' ability to perform KGE related tasks (KGC, inferring insight).

We hope to continue researching this particular domain by perform AA on a real world dataset, motivating further research for causal effects of AA on KGE and methods for defending a KGE model.

# Proposed Methodology
[Cite CRIAGE] implemented a gradient based approach to calculate influence score (define influence score + purpose).  While a brute force approach can be computationally expensive, There are other approaches to this. <!-- (Insert more + cite) -->

We hope to target facts related to Judge entity types -- isolating the search space for AA attacks, reducing the search space for an attack to be implemented <!--<!-- (maybe specific to charge still). -->
<!-- (Insert reason -- tie back to intro) -->

## Expected Results
Our proposed implementation would yield an attack on the embedding space relative to judge entities (unless this changes) .

We would like to explore perturbed embedding space to identify if KGE models can incorrectly infer judge ruled sentencing.  This investigation hopes to understand if attacked KGE can link to causing judges to appear as sentencing towards a bias ruling. A bias ruling is understandable by analyzing similar rulings on similar cases, sometimes with differring judges. We hope to see the perturbed KG consistently inferring a judge towards sentencing more aggressive charges which have harsher jail time/death sentencing. <!-- (Smartify, include typing of entities somehow) -->

We would also like to identify a maximal threshold, perhaps in a percentile, of number of facts to perturb in discovering effectiveness of AA in KGE.
<!-- (Verify if Declan does this -- if yes, toy v real data, if no, new research) -->

## Proposed Evaluation 
While KGE metrics analyzes the plausibility of inferred facts and the presence of existing facts, the validation of incorrectly inferred facts can be validated with unseen valid KG facts. <!-- (This probably needs group-discussed) -->

# Conclusion
<!-- New Hook, Summarize above sections in 1-2sentences per section, New closing remarks -->
