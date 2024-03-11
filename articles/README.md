# Articles
This directory is used to store articles related to the project scope for CS7820

## [Article Title](www.link.to.article)
Author(s)  
Annotated Notes/Summary  
* Problem Statement/Proposed Solution/Future Work

## [Quantifying and Defending against Privacy Threats on Federated Knowledge Graph Embedding](https://arxiv.org/pdf/2304.02932.pdf)
* Three attacks proposed to infer the existence of specific KG triples held by victim clients in federated knowledge graph embedding 
* Attack settings involve server and client adversaries with different background information and capabilities
* Attack types include server-initiated inference attack, client-initiated passive inference attack, and client-initiated active inference attack.
* Attack:
   * Server-initiate Inference Attack: Involves calculating entity embeddings, inferring relations, and inferring specific triple types using an auxiliary dataset
   * Client-initiate Passive Inference Attack: Adversary follows standard FKGE protocol but identifies target entity embeddings and approximates relation embeddings to infer triples
   * Client-initiate Active Inference Attack: Malicious client modifies local embeddings to gain better attack capability by reversing entity embeddings and inferring triple existence

* Defense: 
   * Baseline defense: involves applying differentially private stochastic gradient descent to FKGE
   * Advanced defense: addresses challenges of model size and sparse gradient property of FKGE improving utility
   * Adaptive privacy budget allocation dynamically adjusts noise scale based on validation accuracy during training

* The defense can defend against inference attacks 

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

In this work, the authors propose to exploit the inductive abilities of KGE models to craft poisoned examples against the model. The inductive abilities are captured through the relationship patterns including symmetry, inversion, and composition in the KG. These are referred to as inference patterns. The authors propose to focus on link prediction using KGE models and consider the adversarial goal of degrading the predicted rank of target missing facts. 

DATA - Evaluate four models with varying inductive abilities - DistMult, ComplEx, ConvE, and TransE on two publicly available benchmark datasets for link prediction - WN18RR and FB15k-237.

The proposed adversarial attacks outperform the random baselines and the state-of-the-art poisoning attacks for all KGE models on both datasets. The attacks based on symmetry inference pattern perform the best across all model dataset combinations.


## [Adversarial Attacks on Knowledge Graph Embeddings via Instance Attribution Methods](https://arxiv.org/abs/2111.03120)
Peru Bhardwaj, John Kelleher, Luca Costabello, Declan O'Sullivan

The authors study the security vulnerabilities of KGE models through data poisoning attacks.
    - To select adversarial deletions, the authors propose to use the model-agnostic instance attribution methods from Interpretable Machine Learning.
    - A heuristic method is proposed to replace one of the two entities in each influential triple to generate adversarial additions.

Instance Attribution Methods:
    - Instance Similarity - Estimate the influence of training triple x on the prediction of target triple z based ont he similarity of their feature representations.
    - Gradient Similartiy - Gradient similarity metrics compute similarity between the gradients due to target triple z and the gradients due to training triple x. The intuition is to assign higher influence         to training triples that have similar effect on the model's parameters as the target triple; and are therefore likely to impact the prediction on target triple. 
    - Influence Functions - To estimate the effect of a training point on a model's predictions, it first approximates the effect of removing the training point on the learned model parameters.

  Adversarial Additions: 
    - Using the Instance Attribution Methods, the training triple x is selected that is most influential to the prediction of z. An adversarial addition can be created by replacing the initial x with a               dissimilar entity x'. The replacing triple will have a different object. The influence of the influential triple will be reduced.

The degradation in predictive performance is more significant on WN18RR than on FB15k-237. This is likely due to the sparser graph structure of WN18RR. The model learns its predictions from few influential triples in WN 18RR.

DATA - Four KGE models - DistMult, ComplEx, ConvE, and TransE on two benchmark datasets - WN18RR and FB15k-237


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


## [Stealthy Targeted Data Poisoning Attack on Knowledge Graphs](https://ieeexplore.ieee.org/document/9458733)
Prithu Banerjee, Lingyang Chu, Yong Zhang, Laks V.S. Lakshmanan, Lanjun Wang

Is it possible to conduct a stealthy targeted data poisoning attack, while keeping the exposure risk low?

Major contributions of the paper:
    Introduce the novel problem of finding targeted knowledge graph attack under an exposure risk constraint. 
         - Attack the KG by means of perturbations where the goal is to maximize the manipulation of the target fact's plausibility while keeping the risk of exposure under a given budget.
    Develop a principled framework, RATA(Risk called Aware Targetted Attacker), for attacks, based on hierarchical deep Q-learning, by making use of the natural match between reinforcement learning and MDP.           The attack approach succeeds in exploiting the non-linear structure of the KG efficiently. RATA learns to use low-risk perturbations without compromising on the performances.
    Our experiments on large real world benchmark KG datasets, demonstrate that RATA achieves good attack performance, while staying within a given exposure risk budget. The experiments show that RATA                 achieves state-of-the-art performance even while using a fractrion of the risk.


## [MaSS: Model-agnostic, Semantic and Stealthy Data Poisoning Attack on Knowledge Graph Embedding ](https://arxiv.org/abs/2209.03303)
Xiaoyu You, Mi Zhang, Fuli Feng, Beina Sheng, Xudong Pan, Daizong Ding, Min Yang

Despite the benefits of public data collection, this mechanism opens up a new attack window for malicious users. Attackers may submit poisoned triplets to manipulate the KG, leading to biased KGEs and wrong decisions of the downstream applications. Through extensive evaluation of benchmark datasets and 6 typical knowledge graph embedding models as the victims, the authors validate the effectiveness in terms of attack success rate(ASR) under opaque-box setting and stealthiness.

The research presented aims to launch a data poisoning attack against KGE models that satisfies the following requirements:
    - Opaque-box Setting - The effectiveness of the proposed data poisoning attack should be promised without the full knowledge of KGE models.
    - Semantic Constraints - The inserted triplets should contain correct semantical information to evade the error detection methods of KG.
    - Stealthiness - The infected model should maintain good performance of clean triplets, preventing the undergoing attack from exposing by clean performance degradation.

Develop a framework that satisfies the constraints above.
    Model-Agnostic Semantic and Stealthy (MaSS)
        - insert indicative paths to mislead the target KGE model - inserting indicative paths composed of triplets to make the model predict the fact. The motivation behind the such design is that                    indicative paths on KGs represent how one entity is related to another semantically by some explicit relations and can be learned by various KGE models with different architectures. 
        
Evaluate the proposed attack on three benchmark datasets, attacking six state-of-the-art KGE models. The experiment is designed to answer the following research questions:
    Can MaSS successfully attack in opaque-box settings?
    Can MaSS conduct a stealthy attack?
    Do the injected triplets of MaSS contain semantical information?
    How do settings influence performance and stealthiness?




## [Poisoning attacks against knowledge graph-based recommendation systems using deep reinforcement learning](https://link.springer.com/article/10.1007/s00521-021-06573-8)
Zih-Wun Wu, Chiao-Ting Chen, Szu-Hao Huang

Introducing KGs into recommendation systems can improve recommendation accuracy. However, KGs are usually based on third-party data that may be manipulated by malicious individuals. In this study, we developed a poisoning attack strategy applied on a KG-based recommendation system to analyze the influence of fake links.

This study aims to attack a KG-based recommendation system using reinforcement learning approaches(Q-learning). Attackers influence the KG by adding or forging facts. During our research, we discovered that
the number of possible attack combinations was excessively high, and each attack combination must interact with the recommendation system under a poisoning attack setting. The correct attack effect of all combinations could not be obtained. 

