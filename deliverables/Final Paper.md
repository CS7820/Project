# Introduction 
Knowledge graphs (KGs) are multi-relational directed graphs widely used to represent knowledge in the form of triplets also known as facts. The edges within the graph represent relations between the entities or nodes they connect. KGs serve as a source of information. <!-- (Type another 1-2 sentences) -->

KGE combine ML techniques to allow KGs with the ability to infer knowledge that are not explicitly stated in the KG.

## Problem Statement 
Public datasets are susceptible to attacks that involve manipulation of the data. Data can be easily added, modified, or removed from a dataset without a user's awareness. One possible attack includes introducing a bias into a dataset. Our research demonstrates the modification of the Cook County litigation dataset specifically to introduce bias into the dataset regarding the sentence decided upon by the judge given the charge and guilty verdict.

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
Because CRIAGE did not work as expected, we coded the adversarial attack from scratch. For this topic, the strategy of the attack involved adding and deleting relationships to inject false information into the graph. Both versions of the attack could be targeted in the sense that we inject or remove triples with certain relationships. For example, we oculd remove a triple (s, p, o) that had a certain targeted relationship. Additionally, we could create another triple by injectiong a targeted relationship between nodes. 

The performance of these attacks could be measured through classification accuracy, to ensure that the level of the attack was effective enough to then move to measure the desired metric. 

## Expected Results
Our proposed implementation would yield an attack on the embedding space relative to judge entities (unless this changes) .

We would like to explore perturbed embedding space to identify if KGE models can incorrectly infer judge ruled sentencing.  This investigation hopes to understand if attacked KGE can link to causing judges to appear as sentencing towards a bias ruling. A bias ruling is understandable by analyzing similar rulings on similar cases, sometimes with differring judges. We hope to see the perturbed KG consistently inferring a judge towards sentencing more aggressive charges which have harsher jail time/death sentencing. <!-- (Smartify, include typing of entities somehow) -->
We would meaure this be observing if the model predicts targeted sentencing with greater or lesser severity as the embeddings change.  

We would also like to identify a maximal threshold, perhaps in a percentile, of number of facts to perturb in discovering effectiveness of AA in KGE.
<!-- (Verify if Declan does this -- if yes, toy v real data, if no, new research) -->

## Proposed Evaluation 
While KGE metrics analyzes the plausibility of inferred facts and the presence of existing facts, the validation of incorrectly inferred facts can be validated with unseen valid KG facts. <!-- (This probably needs group-discussed) -->

# Conclusion
<!-- New Hook, Summarize above sections in 1-2sentences per section, New closing remarks -->
