# Adversarial Attacks on Cook County Litigation Dataset
Author(s): Brandon Dave, Calvin Greenewald, Stacie Severyn

# Introduction
Knowledge graphs (KGs) are multi-relational directed graphs widely used to represent knowledge in the form of triplets (or facts) and act as a source of information (or knowledge) based on the domain to which they represent [1]. Graph nodes and edges are respectively implemented as entities and relations between entities.

Datasets represented in KGs are commonly accepted as truth, but, unfortunately, they are susceptible to attacks [2, 3]. These attacks, perturbing facts of a KG, can appear as easily adding, modifying, or removing facts without a user’s awareness. A current research in KG application includes the combination of semantic tools (KGs) and machine learning techniques to train Knowledge Graph Embedding (KGE) models. While new information (or facts of a KG) can be inferred by domain experts, KGE models allow for machines to attain the ability to learn and evaluate the likelihood of a fact’s existence provided the KG as a source of training data [4], otherwise simply known as Knowledge Graph Completion (KGC). There exists research that reveals that a perturbed KG does indeed worsen KGE model performances (or their ability to perform KGC tasks) [5,6,7].

The proposed research in this paper intends to implement a poisoning attack [5] on a real-world dataset transformed to a KG, originating from the Cook County Sentencing data<sup>1</sup>. Similar to other research, we expect implementing perturbed facts to the KG will worsen a trained KGE model's ability to perform KGC-related tasks. Whereas a benchmark KG, used in research in this field, can be demonstrated as a toy dataset, or a dataset designed to be manipulated and studied, our implementation of a real-world KG motivates additional research to promote security measures in defending publicly available KGs. Our newly created KG (unaffected by attacks) will serve as the baseline to evaluate against with the perturbed KG.

While worsening a KGE model’s performance is the ideal goal of attacking the source KG, our research also hopes to understand whether a degree of social bias exists within the Cook County KG. While demonstrating social bias is frowned upon, the practice of bias can sometimes be present unconsciously. A secondary goal from our attack is to cause the presence of social bias to become more pronounced. While it is unfortunate that individuals are afflicted by social bias in the court system [8], our research aims to attack solely facts of the judge of a sentencing case itself, rather than the agents performing the roles of plaintiff or defendant of a case, to demonstrate judges falsely practicing bias towards a particular sentencing, regardless of the aforementioned agents and severity of the crime(s) committed.

<!--
****
PARAGRAPH BELOW IS MODIFICATION TO PARAGRAPH ABOVE - DELETE ONE OF THEM
****
While worsening a KGE model’s performance is the ideal goal of attacking the source KG, a tertiary goal from our attack is to cause the presence of social bias to become more pronounced within the perturbed knowledge graph. Our research aims to attack solely facts of the sentencing case itself, rather than the plaintiff or defendants of a cause, to demonstrate judges practicing bias towards a particular sentencing, regardless of the severity of the crime committed.
-->

The contributions to this field of research includes:
1. The formation of the Cook County Sentencing data as a KG.
2. A homebrew poisoning attack producing and removing facts of the KG.
4. An analysis towards understanding social bias in the embedding space of KGE models from real-world data


# Related Works
[I am the related works.](https://github.com/CS7820/Project/tree/main/articles) 

# Proposed Methodology
[7] implements a gradient based approach to calculate influence score, or a metric gauging the importance of neighboring facts to the targeted fact<sup>2</sup>. While a targeted perturbation attack can have a higher likelihood of worsening the affects of a target model’s performance, we propose seeding the exploration of attacks and their effects through the implementation of semi-random selection-based attacks [9].
<!-- 
******
QUESTION ABOUT THE SENTENCE ABOVE REGARDING WORDING "SEMI-RANDOM" -  IS THE ATTACK TARGETTED IF WE ARE TRYING TO ADD BIAS(ATTACKING FACTS ABOUT THE SENTENCING)?
******
-->

As our research intends to target the judge and their ruling over a case, we propose improving the computational complexity of our attack by reducing the search space of our attack by beginning our perturbation of facts with facts related to judge entities in our KG. We will implement a breadth-first search (BFS) [10] from the judge entities to gain a better understanding of neighboring relations and entities. The result of BFS will provide a tree-representation of facts and their distance away from the root fact (the root node can be represented as the judge's entity typing), represented by the tree’s depth levels. This will allow for the experiment to explore both an attack on direct relations from the judge entities (where the depth level is closer to 0) and on indirect relations from the judge entities (where the depth level is farther from 0).

<!-- (Insert reason -- tie back to intro) -->
Our adversarial attack will need to be coded from scratch. The strategy of the attack will also involve adding (injecting) false information and deleting (removing) relationships from the graph. The performance of these attacks will be measured through classification accuracy, to ensure that the level of the attack was effective enough to then move to measure the desired metric. 

## Expected Results
Our proposed implementation would yield an attack on the embedding space relative to judge entities (unless this changes) .

We would like to explore perturbed embedding space to identify if KGE models can incorrectly infer judge ruled sentencing.  This investigation hopes to understand if attacked KGE can link to causing judges to appear as sentencing towards a bias ruling. A bias ruling is understandable by analyzing similar rulings on similar cases, sometimes with differring judges. We hope to see the perturbed KG consistently inferring a judge towards sentencing more aggressive charges which have harsher jail time/death sentencing. <!-- (Smartify, include typing of entities somehow) -->
We would meaure this be observing if the model predicts targeted sentencing with greater or lesser severity as the embeddings change.  

<!--
***
DELETE PARAGRAPH ABOVE OR BELOW...WHICH ONE DO YOU LIKE?
***

We would like to explore perturbed embedding space to identify if KGE models can incorrectly infer judge ruled sentencing.  This investigation hopes to understand if attacked KGE can link to causing judges to appear as sentencing towards a bias ruling. We hope to see the perturbed KG consistently inferring a judge towards sentencing more or less aggressive charges which respectively have harsher or limited jail time/death sentencing. <!-- (Smartify, include typing of entities somehow) --#>
-->

We would measure this by observing if the model predicts targeted sentencing with greater or lesser severity as the embeddings change.
 


We would also like to identify a maximal threshold, perhaps in a percentile, of number of facts to perturb in discovering effectiveness of AA in KGE.
<!-- (Verify if Declan does this -- if yes, toy v real data, if no, new research) -->

## Proposed Evaluation 
While KGE metrics analyzes the plausibility of inferred facts and the presence of existing facts, the validation of incorrectly inferred facts can be validated with unseen valid KG facts. <!-- (This probably needs group-discussed) -->

[11] introduces the notion of toxicity and stealthiness of KGE model attacks by analyzing the decreased change in MRR metrics as a reflection of the toxicity of an attack and the degree to which added poisoned triples interfere with non-targeted triples as stealthiness. These notions act as investigating starting points of analyses of our proposed implementation. 

# Conclusion
<!-- New Hook, Summarize above sections in 1-2sentences per section, New closing remarks -->
As an established tool for gaining insight and knowledge within a particular domain, KGs used to train KGE models suffer from the same susceptibilities of datasets when training machine learning models. The area of research in attacking and defending these particular datasets have become an arms race in both studying and practicing attack and defense techniques. THere exists real-world consequences from attacking KGs, which can result in producing falsified knowledge through the perturbed facts of an KG. Information particular to the legal domain and its inner-workings still exists as a mystery to the general public, who are a subset of users that benefit from the exploration and querying of a legal-domain focused KG.

Our proposed research incorporates techniques similar to many of the studied lessons of previous researchers to apply into a real-world dataset, the Cook County Sentencing dataset. With this newly formed KG, we propose attacking the KG before it is trained to KGE models with the expectation of worsening the models' abilities to perform link prediction or KGC-related tasks. We will explore a tree-based structure, generated by performing BFS on the KG, where the root node acts as a targeted fact, and the following levels and leaf nodes act as neighboring facts. We will explore both neighboring facts that appear on higher levels and lower levels to evaluate attacks on a targeted fact and a KGE models ability to infer new insight (or knowledge) from this targeted fact. An overall gauge of KGE model performance can be evaluated with the traditional metrics used in KGE model evaluation; such as, hits@k, mean rank, and mean reciprocal rank.

This research aims to motivate continual research in the fields of neural-symbolic (NeSy) AI, applications of KGs, and research with perturbations in computer science.

# Footnotes
1. https://www.infoq.com/news/2019/12/tensorflow-eol-python-2/
2. It has since been discovered that the CRIAGE framework introduced in [7] was designed with Python 2 when the TensorFlow package still supported Python 2. As of the date of this writing, the authors of TensorFlow have stopped support for Python 2. https://twitter.com/TensorFlow/status/1202271346396405762

# References
1. [1] M. Kejriwal, C. Knoblock, P. Szekely, Knowledge Graphs: Fundamentals, Techniques, and Applications, Adaptive Computation and Machine Learning series, MIT Press, 2021. URL: https://books.google.com/books?id=iqvuDwAAQBAJ.
2. [2] P. Banerjee, L. Chu, Y. Zhang, L. V. Lakshmanan, L. Wang, Stealthy targeted data poisoning attack on knowledge graphs, in: 2021 IEEE 37th International Conference on Data Engineering (ICDE), 2021, pp. 2069–2074. doi:10.1109/ICDE51399.2021.00202.
3. [3] Y. Chudasama, Exploiting semantics for explaining link prediction over knowledge graphs, in: C. Pesquita, H. Skaf-Molli, V. Efthymiou, S. Kirrane, A. Ngonga, D. Collarana, R. Cerqueira, M. Alam, C. Trojahn, S. Hertling (Eds.), The Semantic Web: ESWC 2023 Satellite Events, Springer Nature Switzerland, Cham, 2023, pp. 321–330.
4. [4] A. Bordes, N. Usunier, A. Garcia-Durán, J. Weston, O. Yakhnenko, Translating embeddings for modeling multi-relational data, in: Proceedings of the 26th International Conference on Neural Information Processing Systems - Volume 2, NIPS’13, Curran Associates Inc., Red Hook, NY, USA, 2013, p. 2787–2795.
5. [5] H. Zhang, T. Zheng, J. Gao, C. Miao, L. Su, Y. Li, K. Ren, Data poisoning attack against knowledge graph embedding, 2019. arXiv:1904.12052.
6. [6] P. Bhardwaj, J. Kelleher, L. Costabello, D. O’Sullivan, Adversarial attacks on knowledge graph embeddings via instance attribution methods, 2021. arXiv:2111.03120.
7. [7] P. Pezeshkpour, Y. Tian, S. Singh, Investigating robustness and interpretability of link prediction via adversarial modifications, CoRR abs/1905.00563 (2019). URL: http://arxiv. org/abs/1905.00563. arXiv:1905.00563.
8. [8] J. Fisher, D. Palfrey, C. Christodoulopoulos, A. Mittal, Measuring social bias in knowledge graph embeddings, 2020. arXiv:1912.02761.
9. [9] A. Dziedzic, S. Krishnan, Analysis of random perturbations for robust convolutional neural networks, 2020. arXiv:2002.03080.
10. [10] R. E. Korf, Depth-first iterative-deepening: An optimal admissible tree search, Artificial Intelligence 27 (1985) 97–109. URL: https://www.sciencedirect.com/science/article/pii/ 0004370285900840. doi:https://doi.org/10.1016/0004- 3702(85)90084- 0.
11. [11] Zhu, D., Lin, Y., Wang, L., Xie, Y., Jiang, J., and Gu, Z. Evaluation framework for poisoning attacks on knowledge graph embeddings. In Natural Language Processing and Chinese Computing (Cham, 2023), F. Liu,N. Duan, Q. Xu, and Y. Hong, Eds., Springer Nature Switzerland, pp. 184–196.
