# Articles
This directory is used to store articles related to the project scope for CS7820

## [Article Title](www.link.to.article)
Author(s)  
Annotated Notes/Summary  
* Problem Statement/Proposed Solution/Future Work

## [Adversarial Explanations for Knowledge Graph Embeddings](https://www.ijcai.org/proceedings/2022/0391.pdf)
Patrick Betz, Christian Meilicke, Heiner Stuckenschmidt
* Find a logical exlanation for KGE model predections
    * apply rule learning approach to learn a logical theory that describes general regulations
    * abductive reasoning to find the triple that together with the theory is the best explanation for the prediction
    * the triple is used as the triple that is attacked
- Black box method used for adversarial attacks
- Abductive reasoning: find an explanation for an observation given a theory
- Either delete or add triples for the attack
- Delete: supress explanation by deleteing the triple
- Addition: perturb the true explanation for the target to a senseless statement about one of the entities in the target 


## [Poisoning Knowledge Graph Embeddings via Relation Inference Patterns](https://arxiv.org/abs/2111.06345)
Peru Bhardwaj, John Kelleher, Luca Costabello, Declan O'Sullivan

I am the annotated bibliography.


## [Adversarial Attacks on Knowledge Graph Embeddings via Instance Attribution Methods](https://arxiv.org/abs/2111.03120)
Peru Bhardwaj, John Kelleher, Luca Costabello, Declan O'Sullivan

I am the annotated bibliography.

## [Evaluation Framework for Poisoning Attacks on Knowledge Graph Embeddings](https://link.springer.com/chapter/10.1007/978-3-031-44693-1_15)
Dong Zhu, Yao Lin, Le Wang, Yushun Xie, Jie Jiang & Zhaoquan Gu 

- Design Toxicity and Stealthiness in Data D for poisoning attack
- Toxicity: Quantifiable by the decrease in MRR given target triples
- Stealthiness: The degree to which added poisoned triples interfere with the unselected triples
- Harmonic Mean of Toxicit and Stealthiness ensures consideration of both to better represent overall impact of data poisoning attacks.
- Random-n attack vs Random-g
    - Adding attacks to target sample vs all graph entities
    - FB15k-237 with embedding TransE, DistMult, ComplEx, ConvE
    - WN18RR was used as a control model to evaluate/support findings on FB15k-237 attacks

## [Data Poisoning Attack against Knowledge Graph Embedding](https://arxiv.org/abs/1904.12052)
Hengtong Zhang, Tianhang Zheng, Jing Gao, Chenglin Miao, Lu Su, Yaliang Li, Kui Ren

- First study on KGE vulnerabilities and proposes a family of effective data poisoning attack strategies, manipulating the training data of KGE with addition and/or deletion of facts (triples).
- Supports KGE analysis for robustness against attacks (specifically poisoning attack)
- KG designed on unreliable and public data sources, user-submitted wiki contributions
- Introduction to Direct Attacks through addition and deletion
- Introduction to Indirect Attacks
- FB15k and WN18 datasets were tested embedded with TransE, TransR, and RESCAL
- Metric evaluation with MRR and Hits@10.

## [Investigating Robustness and Interpretability of Link Prediction via Adversarial Modifications](https://arxiv.org/abs/1905.00563)
Pouya Pezeshkpour, Yifan Tian, Sameer Singh

CRIAGE, Completion Robustness and Interpretability via Adversarial Graph Edits, studies the affects of added and removed facts from a knowledge graph. The study performs two experiments to evaluate a KG's robustness and sensitivity to fact changes: 1) removing neighboring links, which can be identified as the most influential related facts to a targeted fact, and 2) adding new facts.  In order to gauge the changes in a graph, the authors propose calculating a Taylor function to approximate node influences. 
CRIAGE is experimented with the KGE benchmark datasets WN18 and YAGO3-10. The experiment is evaluated from two tests with the traditional KGE metric evaluation methods, MRR and Hits@k, however only when k is 1.  

The first test targets all test nodes for node modifications.  
The second test implements attacks on a subset of test data that fit two criterias.  The subset consists of nodes that the model best (most correctly) predicts and where the difference between the subset's scoring function and the highest scoring function of the negative samples are the lowest. CRIAGE is also evaluated against two baseline methods: Random Attack, where random target facts are modified; and Opposite Attack, where target facts are modified is is calculated and calculated and when a decoder is fed the subtraction of a fact based on fixed embedding space of the subject and relationship.

The authors also include to CRIAGE an inverter in decoding the embedding space to support tractability of the search space for fact generation. This was performed by analyzing subgraph patterns where R<sub>1</sub>(a,c) and R<sub>2</sub>(c,b) with respect to a target triple and extracting rules that appear more than 90% of the times in the target triple's graph neighborhood. The rule for R<sub>2</sub>(c,bb) would then be removed. This method is supported with YAGO3-10 embedded with DistMult as the original publication included extracted rules that CRIAGE was able to also replicate.

The authors conclude that CRIAGE better performs on multiplicative scoring function-based KGE models, choosing to use DistMult and ConvE simply based on resulting in the highest accuracies; however, points out that the research reflects into additive-based models as well.

## [Exploiting Semantics for Explaining Link Prediction Over Knowledge Graphs](https://link.springer.com/chapter/10.1007/978-3-031-43458-7_50)
Yashrajsinh Chudasama 

I am the annotated bibliography.
