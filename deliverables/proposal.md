# Attacking Litigation Knowledge Graph with Falsified Triples

## Administrivia
* **Super Duper Group group**:

* **Team Members**:
    * Brandon Dave
    * Calvin Greenewald
    * Stacie Severyn

#  Problem Statement
Knowledge graphs (KGs) are an established tool used to represent structured data[1]. Knowledge Graph Embedding (KGE) combines techniques from machine learning to understand semantics in the data and also to complete knowledge graphs by performing link predictions, inferring relationships between represented concepts[2]. These learned models share vulnerabilities similar to traditional machine learning models as the data source, the KG in this case, is prone to adversarial and poisoning attacks that decrease the performance of embedding models to handle their tasks[3].

As knowledge graphs are developed as a source of truth in their respective domains, if a KG were to be attacked, the queries within this domain would, thus, return falsified facts, invalidating the KGs usage. 

We propose performing an attack on a KG focused in the legal domain which would affect learned embedding models and their handling of KG completion tasks.

Data.gov provides a non-federal dataset covering information on sentencing data of guilty verdicts in cases from Cooke County in Illinois[5]. The developed knowledge graph will be able to answer the following competency questions:
1) Who are the agents and their respective role for Case [insert]?
2) What charges has Judge [insert] sentenced?

From this new KG, we hope to explore the following research question(s):
1) At what rate of perturbed data does the reliability of the knowledge graph in representing the Case, the Court, the corresponding Agents to a Case, and the Sentencing Charge start to diminish?
2) Can perturbed KGE models demonstrate biasness in court sentencing?

To experiment RQ1, we intend to implement CRIAGE[4], which is a framework designed to challenge the robustness and interpretability of link prediction tasks of KGE models. [4] introduces both the ability to add false facts and also remove targeted facts -- both based on an influence score (a metric gauging the change of a prediction score of an observed fact when perturbed).

In exploring RQ 2, we hypothesize that a corrupted KG after a targeted adversarial attack can skew a KGs facts about judges and their ruled sentencing. We hope to demonstrate these judges sentencing similar charges (in the sentencing commitment) despite varying severities.

# Justification
##  Why NeSy AI
We are using NeSy AI to train KGE models that are able to perform link prediction. The KGE models will be trained particularly with legal data  where the reliability and robustness of the KG provides a source of truth in analyzing information pertaining to court cases. NeSy AI combines symbolic representations with subsymbolic representations learned by neural networks. This combination allows for more robust handling of structured data while also utilizing the power of neural networks to capture the  patterns and semantics. The combination of reasoning and symbolic knowledge better equip the model to combat adversarial attacks. This will make it interesting to attack the model in an attempt to falsify predicted links, as it is natuarlly more difficult to manipulate the data. 

We expect that KGE models will embed data about case's assigned judge and the judge's verdict ruling, and we hope to analyze the underlying semantic between these two concepts. We hope that our analysis will reveal a relationship that points towards a biasness in the judge's verdict ruling regardless of the severity of a crime.

## Intellectual Merit
Previous research results suggest the decreased performance in KGE models after the KG has been perturbed. The KGs used in these research are benchmarking datasets, designed to challenge the KGE models, like FB15k-237 and WN18RR. Our research hopes to demonstrate an attack on a realistic, non-toy dataset. The dataset includes litigation data from Cooke County of Illinos. We will investigate the robustness and interpretability of link prediction via adversarial modification utilizing the CRIAGE framework. CRIAGE is designed to determine the fact(s) to add or remove from the knowledge graph that changes the prediction for a target after the model is retrained. 

## Broader Impacts:
The primary interest in this research is to identify whether there exists a biasness in a judge and their sentencing towards a crime regardless of the crime's severity. If a bias is identified, the results of this research could impose real-world consequences in the legal community. As the judge should be a source of just rulings held to extreme ethical standards, if a judge is found favoring a certain ruling, the case could be disputed for a mistrial due to unfair circumstances, and the ruling judge would have to be held accountable for the unjust acts. 

# References
* [1] Schad, J. Bridging the gap: Integrating knowledge graphs and large language models, Oct 2023.
* [2] S. Ji, S. Pan, E. Cambria, P. Marttinen and P. S. Yu, "A Survey on Knowledge Graphs: Representation, Acquisition, and Applications," in IEEE Transactions on Neural Networks and Learning Systems, vol. 33, no. 2, pp. 494-514, Feb. 2022, doi: 10.1109/TNNLS.2021.3070843. 
* [3] Bhardwaj, P., Kelleher, J., Costabello, L., and O’Sullivan, D. Adversarial attacks on knowledge graph embeddings via instance attribution methods, 2021.
* [4] Pezeshkpour, P., Tian, Y., and Singh, S. Investigating robustness and interpretability of link prediction via adversarial modifications. CoRR abs/1905.00563 (2019).
* [5] https://catalog.data.gov/dataset/sentencing. Updated on 24 Feb., 2024. Accessed on 20 Mar., 2024.