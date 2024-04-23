

Important Comment - I believe we should be measuring whether the poisoning attack introduces bias into the dataset. This requires calculations of the inital dataset to be compared to the modified dataset. For the poisoning attack to introduce bias the probabilities of the intial dataset will be much different from the modified dataset for the bias.




# Introduction 
Knowledge graphs (KGs) are multi-relational directed graphs widely used to represent knowledge in the form of triplets also known as facts. The edges within the graph represent relations between the entities or nodes they connect. KGs serve as a source of information about a particular domain. Datasets represented in KGs are accepted as truth but unfortunately they are susceptible to attacks. During the lifetime of the dataset, data can be easily added, modified, or removed without a user's awareness. One possible attack poisoning attack includes introducing bias within the dataset. Our research demonstrates the modification of the Cook County litigation dataset specifically to introduce bias into the dataset regarding the sentence decided upon by the judge given the charge and guilty verdict.

KGE combine ML techniques to allow KGs with the ability to infer knowledge that are not explicitly stated in the KG.

## Problem Statement 


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



Bias within a dataset may focus on gender, race, sentencing, etc. We chose to focus on the sentence decided upon by the judge given the charge and guilty verdict. To determine if a judge is biased towards a charge the probability of each sentence assigned by this judge for each charge can be calculated for the entire dataset. This process should be repeated for all judges. It is vital to calculate these probabilities for the original dataset as well as the modified dataset to witness whether bias was introduced. A histogram can be created with the appropriate groupings to visually show the differences between each judge. 

## Expected Results
Our proposed implementation would yield an attack on the embedding space relative to judge entities (unless this changes) .

We would like to explore perturbed embedding space to identify if KGE models can incorrectly infer judge ruled sentencing.  This investigation hopes to understand if attacked KGE can link to causing judges to appear as sentencing towards a bias ruling. A bias ruling is understandable by analyzing similar rulings on similar cases, sometimes with differring judges. We hope to see the perturbed KG consistently inferring a judge towards sentencing more aggressive charges which have harsher jail time/death sentencing. <!-- (Smartify, include typing of entities somehow) -->
We would meaure this be observing if the model predicts targeted sentencing with greater or lesser severity as the embeddings change.  

We would also like to identify a maximal threshold, perhaps in a percentile, of number of facts to perturb in discovering effectiveness of AA in KGE.
<!-- (Verify if Declan does this -- if yes, toy v real data, if no, new research) -->


## Results
To demonstrate whether bias is introduced into the Cook County dataset after the poisoning attack the new dataset must be compared to the initial dataset. For each judge, the probability of each charge and the sentences assigned must be calculated. These calculations should be repeated for the modified dataset to allow the differences in the probabilities to be determined. Histograms with the proper groupings for judge, charge, and sentence can be created to visually show the difference or bias added to the dataset.


## Proposed Evaluation 
While KGE metrics analyzes the plausibility of inferred facts and the presence of existing facts, the validation of incorrectly inferred facts can be validated with unseen valid KG facts. <!-- (This probably needs group-discussed) -->

# Conclusion
<!-- New Hook, Summarize above sections in 1-2sentences per section, New closing remarks -->
