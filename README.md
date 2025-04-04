# Neura-Scholar

[1] Our business or service is to provide an easier way for users to find relevant research papers from a vast corpus using semantic search.
[2] The existing Non-ML status quo is arXiv's keywords based search over title, author names and abstract.
[3] Business metric would be to Minimize the click through rate over entries from search results, click through rate through to the paper text from these results. Overall, improving search efficiency that is to reduc the time spent on search and serve highly relevant papers based on the search query.

### Contributors

<!-- Table of contributors and their roles. 
First row: define responsibilities that are shared by the team. 
Then, each row after that is: name of contributor, their role, and in the third column, 
you will link to their contributions. If your project involves multiple repos, you will 
link to their contributions in all repos here. -->

| Name                            | Responsible for | Link to their commits in this repo |
|---------------------------------|-----------------|------------------------------------|
| All team members                | The overall Project | |
| Preetham Rakshith Prakash | DevOps | |
| Pranav Bhatt | Model Serving | |
| Riya Gargh | Data Pipeline | |
| Yugesh Panta | Training at Scale | |

###Contributors

| Name                            | Responsible for | Link to their commits in this repo |
|---------------------------------|-----------------|------------------------------------|
| Preetham Rakshith Prakash    |                 |   https://github.com/Yugesh1620/Neura-Scholar/commits/main/?author=th-blitz                                 |
| Riya Garg                  |                 |                                    |
| Pranav Bhatt                  |                 |                                    |
| Yugesh Panta |                 |    https://github.com/Yugesh1620/Neura-Scholar/commits/main/?author=Yugesh1620             |


### System diagram


![System Diagram](System%20Diagram.png)


### Summary of outside materials

<!-- In a table, a row for each dataset, foundation model. 
Name of data/model, conditions under which it was created (ideally with links/references), 
conditions under which it may be used. -->

|              | How it was created | Conditions of use |
|--------------|--------------------|-------------------|
| arXiv Dataset (PDFs) | [From arXiv's bulk access API](https://info.arxiv.org/help/bulk_data/index.html) | [arxiv's nonexclusive-distrib/1.0 license](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html) for individual papers
| arXiv Dataset (Metadata) | [From arXiv's kaggle dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv) | [Creative Commons CC0 1.0 Universal Public Domain Dedication](https://creativecommons.org/publicdomain/zero/1.0) for the metadata
| Script to generate utf-8 encoded plain Text from arXiv PDFs | A third party script called [mattbierbaum/arxiv-public-datasets](https://github.com/mattbierbaum/arxiv-public-datasets) to pull in arXiv Dataset in bulk from arXiv's bulk access API or Google bucket or AWS bucket and to generate plain text | [MIT License](https://github.com/mattbierbaum/arxiv-public-datasets/blob/master/LICENSE) |
| Base model 1 | [Linq-AI-Research/Linq-Embed-Mistral](https://huggingface.co/Linq-AI-Research/Linq-Embed-Mistral) | Creative Commons Attribution Non Commercial 4.0 |
| BART (facebook/bart-large-cnn or philschmid/bart-large-cnn-samsum) | Pretrained by Facebook on CNN/DailyMail and/or SAMSum datasets, a Large language model with encoder-decoder structure. Hosted on [huggingface](https://huggingface.co/facebook/bart-large
). | MIT License (Samsum variant) or Fairseq license for BART. Free for research use. |
| etc          | | |


### Summary of infrastructure requirements

<!-- Itemize all your anticipated requirements: What (`m1.medium` VM, `gpu_mi100`), 
how much/when, justification. Include compute, floating IPs, persistent storage. 
The table below shows an example, it is not a recommendation. -->

| Requirement     | How many/when                                     | Justification |
|-----------------|---------------------------------------------------|---------------|
| `m1.medium` VMs | 1 for the entire project duration, 2 more during final setup | For hosting 1 conventional DB, 1 vector DB and 1 as a proxy |
| `compute_liqid` node with 2 GPUs OR `gpu_mi100` node with 2 GPUs | 1 node with 2 a100s for 20 hours a week ( 2x 4 hour blocks and 2x 6 hour blocks a week ) | For training and serving models |
| `compute_liqid` or `gpu_mi100` node with 1 GPU | For the entire project duration | For training and serving models | 
| Persistent Storage | 120 GB for entire project duration | Needed to persist models, database, training datasets, metrics etc |
| Floating IPs | 2 for entire project duration | 1 for everything ( gateway, dashboards etc ), and 1 as a backup |

<!--
| `m1.medium` VMs | 3 for entire project duration                     | ...           |
| `gpu_mi100` GPUs | 4 hour block four times a week |               |
| Persistent Storage | 120 GB for project duration | Needed to persist datasets, models, metrics, etc. |
| Floating IPs    | 1 for entire project duration, 1 for sporadic use |               |
| etc             |                                                   |               |
-->

### Detailed design plan

## Unit 4:
### Train and Re-train:

1. **Embedding Model**  
   - Trained on arxiv meta data and abstracts for semantic similarity  
   - Retrained with updated queries  

2. **Summarization Model**  
   - Trained on top-k paper summaries  
   - Retrained with better hyperparameters or more diverse examples  


 
### Models:
**Embedding Model**: MiniLM(sentence-transformers/all-MiniLM-L6-v2)
**Summary Model**: Bart(bart-large-cnn-samsum)
 
### Extra Difficulty points we will try:
1.Training strategies for large models

2.Use distributed training to increase velocity


## Unit5:

### Experiment Tracking:
We will dedicate 1 m1.medium for MLflow and log:
  Model architecture, version
   Hyperparamters
   Training loss, validation loss
    GPU and memory usage
    Checkpoint paths
 
### Ray cluster for Scheduling Jobs:
Schedule, launch jobs, and configure Ray worker's nodes to execute training jobs.
 
### Extra Difficulty points we will try:
1. Using Ray Train
2. Scheduling hyperparameter tuning jobs

### Detailed design plan

<!-- In each section, you should describe (1) your strategy, (2) the relevant parts of the 
diagram, (3) justification for your strategy, (4) relate back to lecture material, 
(5) include specific numbers. -->

#### Model training and training platforms

<!-- Make sure to clarify how you will satisfy the Unit 4 and Unit 5 requirements, 
and which optional "difficulty" points you are attempting. -->

#### Model serving and monitoring platforms

<!-- Make sure to clarify how you will satisfy the Unit 6 and Unit 7 requirements, 
and which optional "difficulty" points you are attempting. -->

#### Data pipeline

<!-- Make sure to clarify how you will satisfy the Unit 8 requirements,  and which 
optional "difficulty" points you are attempting. -->

#### Continuous X

<!-- Make sure to clarify how you will satisfy the Unit 3 requirements,  and which 
optional "difficulty" points you are attempting. -->



