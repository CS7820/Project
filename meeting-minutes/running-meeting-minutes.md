# Running Meeting Minutes
## Date: 4/11/24
### Attending 
* Brandon Dave
* Calvin Greenewald
* Stacie Severyn

### Agenda 
* Meeting 4/11/24 at **6:30pm** in person

### Actions 
* **Complete presentation slides**
* Due: **4/13/24** 
* Assigned Everyone 

### Notes 


## Date: 4/4/24
### Attending
* Brandon Dave
* Calvin Greenewald
* Stacie Severyn

### Agenda
* Build and run singularity container on fry for CRIAGE implementation on subgraph 

### Actions
* **Split the data into train-test-validate**
* Assigned to: Calvin
* Due 4/7

* **Create presentation in Google Slides, including Introduction**
* Assigned to: Stacie
* Due 4/9

### Notes 
* Group discussion day 4/9 (during class hours)


## Date: 4/2/24
### Attending
* Brandon Dave
* Calvin Greenewald
* Stacie Severyn

### Agenda
* Discuss sub-graph now materialized
* Discuss schema-diagrams
* Discuss CRIAGE

### Actions
* CRIAGE implementation on subgraph
    * Assigned to: Everyone
    * Report back 4/4 on findings

### Notes 
* Next meeting is 4/4 (during CS7820 hours)

## Date: 3/28/24
### Attending
* Brandon Dave
* Calvin Greenewald

### Regrets
* Stacie Severyn

### Agenda
* Discuss next stops from sub-graph formation

### Actions
* Full-graph Diagramming (yEd) 
Assigned to: Calvin
Due by: Tuesday, 4/2
* Sub-graph materialization using Python rdflib
Assigned to: Brandon
Due by: Tuesday, 4/2
* Investigation to CRIAGE
    * Where is CRIAGE framework hosted?
Assigned to: Stacie, support from Brandon and Calvin (if available)
Due by: Friday, 4/2

### Notes 
* Next meeting is 4/2 (during CS7820 hours)
* Full-graph materialization after sub-graph formulation

## Date: 3/12/24
### Attending
* Brandon Dave
* Calvin Greenewald
* Stacie Severyn

### Agenda
* Read the current state of the proposal
* Discussed meeting during "no class" periods
* What to bring up with Cogan come Tuesday, 3/19

### Actions
* Sub-graph Diagramming (yEd) 
Assigned to: Stacie
Due by: Monday, 3/18
* Sub-graph materialization using Python rdflib
Assigned to: Calvin and Brandon
Due by: Thursday, 3/21
* Proposal modifications
Assigned to: Everyone
Due by: Friday, 3/22

### Notes 
* Next meeting is 3/14
* Ask Cogan about Hosting Data on GH

## Date: 3/12/24
### Attending
* Brandon Dave
* Calvin Greenewald
* Stacie Severyn

### Agenda
* Nothing concrete - open discussion

### Actions
* CQ by Wednesday
Assigned to: Everyone 
* Begin sub-graph materialization 
Assigned to: Everyone

### Notes 
* Next meeting is 3/14
* Ask Cogan about Hosting Data on GH

## Date: 3/7/24
### Attending
* Brandon Dave
* Calvin Greenewald
* Stacie Severyn
* Cogan Shimizu

### Agenda
* Get all members of the same page with regards to research ideas
* Present project scope and ideas to Cogan

### Actions
* Look into CRIAGE implementation
  Assigned to: Everyone 

### Notes 
* Create misinformaiton on a subset of data from SCALES, and then determine if the
  nature of the misinformation carries over to the original SCALES as well. 

## Date: 3/6/24

### Attending
* Brandon Dave
* Calvin Greenewald
* Stacie Severyn
* Cogan Shimizu

### Regrets
* Lingwei Chen

### Agenda
* Presenting current status of CS7820 Project idea wrt SCALES OKN
* Receive feedback from Advisors on next-steps

### Action
* Update Lingwei; CC Everyone
Assigned to: Calvin

* Review SCALES data and come up with RQ relative to data
Assigned to: Everyone

### Additional Notes
* Demographic adversarial impacted by a system that would recommend a verdict/sentencing
    * Federal:  Recurrence rarely happens

## Date: 3/5/24

### Attending
* Brandon Dave
* Calvin Greenewald
* Stacie Severyn

### Agenda
* Formulating Problem Statement in preparation for 3/6/24 meeting with Advisors

### Action
*     

### Additional Notes
* 

## Date: 02/22/24

### Attending
* Brandon Dave
* Calvin Greenewald
* Stacie Severyn

### Agenda
* Discussion of problem statement
* Solidify yes/no with going with Cogan and Lingwei's Preliminary research pushing for NeSy 2024 deadline (April 5, 12)
* Planning on Spring Break: Cogan out of office Sunday-to-Sunday, but remotely reachable


### Action
* Research knowledge gaps
    * Assigned to: Calvin, Stacie
    * Due by: 3/6
* Coordinate a meeting with Cogan and Lingwei post-spring break to present concrete ideas
    * Assigned to: Everyone
    * Due by: 3/5

### Additional Notes
* Datasets have been provided on Discord
* Next meeting: 3/6
* 3pm Lingwei <> Calvin meeting will solidify preliminary research decision

## Date: 02/20/24

### Attending
* Brandon Dave
* Calvin Greenewald
* Stacie Severyn

### Agenda
* Next meeting 1-2pm Thursday
* Discussion of findings from annotated biblio

### Action
* During Lit Review, figure out AA methodologies
    * Assigned to: Everyone
    * Due by: 2/22

* Continue lit review
    * Assigned to: Everyone
    * Due by: 2/23
    
### Additional Notes
* Using LaTeX for any paper writing
* Plan around Spring Break, maybe virtual meeting looping in faculty members on overall total ideas gathered

## Date: 02/13/24

### Attending
* Brandon Dave
* Calvin Greenewald
* Stacie Severyn

### Agenda
* Discuss Lit Review stage

### Action
* Complete reading of 3-5 annotated bibliograph (including Cogan suggested)
    * Assigned to: Everyone
    * Due by: 2/20
    

### Additional Notes
* Public KBs can be found from one of the appendices of Brandon's paper when we get that far
* The actual output of KB falsified facts are not necessarily the purpose, but analyzing the performance changes in KGs after training on AA data points
* Use found paper references for hints on connecting relevance
* Narrative: 
    * From Cogan email:
        * > **Declan O'Sullivan** is a part of two of these. Maybe it's worth also digging further into his work more extensively. Anyway, these papers were relatively quick to consume.
        * > Anyway, there is research that says that **dilution of true facts with adversarial or false facts can obscure true facts within the dataset (kg)**. So perhaps one way of moving forward is – **what is the threshold or ratio to which true vs. decoy becomes problematic in link prediction**. After that, the steps become more nebulous. 
            * What mechanisms are there to protect against poisoning? 
            * Does choosing a training set more carefully (or differently) matter? 
            * To what extent does the incorporation of false negative triples as couterfactuals during training negatively impact performance? 
            * Is there a way to make the KGE system "doubt itself" so to speak?
