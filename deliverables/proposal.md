# Attacking Litigation Knowledge Graph with Falsified Triples

## Administrivia
* **Super Duper Group group**:

* **Team Members**:
    * Brandon Dave
    * Calvin Greenewald
    * Stacie Severyn

#  Problem Statement
Knowledge graphs (KGs) are an established tool used to represent structured data[1]. Knowledge Graph Embedding (KGE) combines techniques from machine learning to understand semantics in the data and also to complete knowledge graphs by performing link predictions, inferring relationships between represented concepts[2]. These learned models share vulnerabilities similar to traditional machine learning models as the data source, the KG in this case, is prone to adversarial and poisoning attacks that decrease the performance of embedding models to handle their tasks[3].

As knowledge graphs are developed as a source of truth in their respective domains, if a KG were to be attacked, a user who is quering within this domain may falsified facts returned to them. We propose performing an attack on a KG focused in the legal domain which would not only affect a learned embedded model but also now corrupted with false facts affecting any querying for factual results, invalidating the KGs application.  

Data.gov provides a non-federal dataset covering information on sentencing data of guilty verdicts in cases from Cooke County in Illinois[5]. From this new KG, we hope to explore the following research question(s):
1) At what rate of perturbed data does the reliability of the knowledge graph in representing the Case, the Court, the corresponding Agents to a Case, and the Sentencing Charge start to diminish?
2) Can KGE demonstrate biasness in court sentencing?

To experiment RQ1, we intend to implement CRIAGE[4], which is a framework designed to challenge the robustness and interpretability of link prediction tasks of KGE models. [4] introduces both the ability to add false facts and also remove targeted facts -- both based on an influence score (a metric gauging the change of a prediction score of an observed fact when perturbed).

In exploring RQ 2, we hypothesize that a corrupted KG after a targeted adversarial attack can skew a KGs facts. To display this, we will target judges and their ruling sentence and hope to demonstrate these judges sentencing similar charges (in the sentencing commitment) in varying severities.

# Justification
##  Why NeSy AI
Utilizing KGE to perform link prediction

## Intellectual Merit
Utilizing Real Data vs Toy Data (as seen in Declan's papers)

## Broader Impacts:
User querying on judge for sentence biasness

# References
* [1] Schad, J. Bridging the gap: Integrating knowledge graphs and large language models, Oct 2023.
* [2] S. Ji, S. Pan, E. Cambria, P. Marttinen and P. S. Yu, "A Survey on Knowledge Graphs: Representation, Acquisition, and Applications," in IEEE Transactions on Neural Networks and Learning Systems, vol. 33, no. 2, pp. 494-514, Feb. 2022, doi: 10.1109/TNNLS.2021.3070843. 
* [3] Bhardwaj, P., Kelleher, J., Costabello, L., and O’Sullivan, D. Adversarial attacks on knowledge graph embeddings via instance attribution methods, 2021.
* [4] Pezeshkpour, P., Tian, Y., and Singh, S. Investigating robustness and interpretability of link prediction via adversarial modifications. CoRR abs/1905.00563 (2019).
* [5] https://catalog.data.gov/dataset/sentencing. Updated on 24 Feb., 2024. Accessed on 20 Mar., 2024.

=======
# Problem Statement
When trained with Knowledge Graphs (KG), Artificial Intelligence (AI) reports higher accuracy while reducing hallucinogenic effects due to the added semantic understanding to the data[1]. As with other machine learning models, the knowledge graph for a KG-powered AI acts as another source of vulnerability, open to adversarial and poisoning attacks targeting the data source, which in this case act as KG facts (or triplets).

The SCALES-OKN knowledge graph provides a tool to the public in understanding the federal judiciary inner-workings, which as a whole can be considered as constructed from the written opinions of less than 10% of cases, as written on their home page[2]. As a knowledge tool to the public, the organization's next goal is to provide an AI powered with their litigation KG to better serve the public in dissolving the mystery of this particular domain.

In our proposed research, we want to explore adversarial attack impact on a non-toy dataset, which if realistically implemented would impose real-world consequences to the verdict ruling for individuals.

% Formulated RQs
1) What metrics or indicators can be used to quantify the level of unreliability in the knowledge graph resulting from the manipulation of data, and at what point do these metrics reach a critical threshold necessitating corrective action or reevaluation of the graph's integrity?

2) Can the results of the charge be specifically changed to another type of charge. 


We expect an attacked KG, that is to say one that has had falsified facts added and true facts removed, will begin answering queries with false claims - thus invalidating the knowlege graph's performance. An interesting angle for this research is to also obtain an understanding of a threshold of applied adversarial attacks in order to cause the decreased performance on the knowledge graph.

# Background and Relevance
## Ontology and LLM

## Adversarial and Poisoning Attacks


# Proposed Methodology and Expected Results
To explore a KG's query performance, we will implement the CRIAGE framework which provides both an analysis in optimal candidates as a list of facts which can be attacked with the highest likelihood of impacting the KG's ability to query and the attack itself. A set of competency questions (CQ), which can be inspired by SCALE's Satyrn Notebooks as an investigative starting point, will be prompted by both KGs with the expectation that the corrupted KG corrupted by adversarial attack results in both omission of facts that would be true or added false claims.

We hypothesize the results of the following CQs will be affected by a targeted attack on the knowledge graph:
1) Who are the agents and their respective role for Case [insert]?
```sql
# 1) Who are the agents and their respective role for Case [insert]?
SELECT ?Case ?Agent ?Role ?Name WHERE {
#  ?Case a scales:CaseCriminal .
  ?Case scales:hasAgent ?Agent .
  ?Agent scales:hasName ?Name .
  ?Agent scales:hasRoleInCase ?Role .
}   
OrderBy ?Case ?Agent ?Role ?Name
```

2) What charges has Judge [insert] sentenced?
```sql
# 2) What charges has Judge [insert] sentenced?
SELECT ?Agent ?Role ?Name ?Charge ?Content WHERE {
  ?Agent scales:hasRoleInCase ?Role .
  ?Agent scales:hasName ?Name .
  ?Charge a scales:Charge .
  ?Charge scales:hasContents ?Content .
  Filter(?Role = "Assigned Judge")
}   
```
3) What circuit was Case [insert] held in?
```sql
# 3) What circuit was Case [insert] held in?
SELECT ?Case ?CaseType ?Court ?Name ?Circuit WHERE {
  ?Case a ?CaseType .
  ?Case scales:isInCourt ?Court .
  ?Court scales:hasName ?Name .
  ?Court scales:isInCircuit ?Circuit .
}   
OrderBy ?Case ?Court ?Circuit
```

To explore a KG's overall performance in Knowledge Graph Completion (KGC) tasks, enabled by Knowledge Graph Embedding (KGE), we will train KGE models with both the provided SCALES KG, as a base, and an attacked SCALES KG, as a corrupted KG. We expect both graphs to still maintain the functionality for link prediction, thus inferring facts in order to complete a KG; however, we hope to observe the evaluation upon a targetted fact that had been attacked to have differing results, specifically being more difficult to find within a top-k ranking system. 

# Significance
## Cogan: Why NeSy AI?
The SCALES OKN exists as a knowledge base (knowledge graph). This is publicly available data that can be attacked via adding or removing facts from the knowledge base which directly affects KGE performance for predictive tasks in KG Completion (KGC).

The research is focused on the impacts of these attacks.

## Cogan: Intellectual Merit
Research into KGE and adversarial attacks on KGs.

## Broader Impacts

Societal:  KG attacks can affect the outcome of an individual's verdict of their case.


