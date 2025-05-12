## Semantic Search over Research Corpus
(Project_22)
<!--
Discuss: Value proposition: Your will propose a machine learning system that can be
used in an existing business or service. (You should not propose a system in which
a new business or service would be developed around the machine learning system.)
Describe the value proposition for the machine learning system. What’s the (non-ML)
status quo used in the business or service? What business metric are you going to be
judged on? (Note that the “service” does not have to be for general users; you can
propose a system for a science problem, for example.)
-->

1. Our business or service is to provide an easier way for users to find relevant research papers from a vast corpus using semantic search.
2. The existing Non-ML status quo is arXiv's keywords based search over title, author names and abstract.
3. Business metric would be to Minimize the click through rate over entries from search results, click through rate through to the paper text from these results. Overall, improving search efficiency that is to reduc the time spent on search and serve highly relevant papers based on the search query.


### Contributors

| Name                            | Responsible for | Link to team members' commits in this repo |
|---------------------------------|-----------------|------------------------------------|
| Preetham Rakshith Prakash | Continuous X | https://github.com/Yugesh1620/Neura-Scholar/commits/main/?author=th-blitz |
| Riya Garg | Data pipeline | https://github.com/Yugesh1620/Neura-Scholar/commits/main/?author=riyagarg30 |
| Pranav Bhatt | Model serving and monitoring platforms | https://github.com/Yugesh1620/Neura-Scholar/commits/main/?author=pranav-bhatt |
| Yugesh Panta | Model training and training platforms | https://github.com/Yugesh1620/Neura-Scholar/commits/main/?author=Yugesh1620 |

### Per Responsibility README Sections
- [Model Serving, Monitoring & Eval README](serving_eval/README.md)
- [Data Pipeline] (read-me-data-pipeline.md)

### System diagram

<!-- Overall digram of system. Doesn't need polish, does need to show all the pieces.
Must include: all the hardware, all the containers/software platforms, all the models,
all the data. -->

![System Diagram](System%20Diagram.png)

### Summary of outside materials

<!-- In a table, a row for each dataset, foundation model.
Name of data/model, conditions under which it was created (ideally with links/references),
conditions under which it may be used. -->

|              | How it was created | Conditions of use |
|--------------|--------------------|-------------------|
| arXiv Dataset (PDFs) | [From arXiv's bulk access API](https://info.arxiv.org/help/bulk_data/index.html) | [arxiv's nonexclusive-distrib/1.0 license](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html) for most individual papers, unknown for the rest
| arXiv Dataset (Metadata) | [From arXiv's kaggle dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv) | [Creative Commons CC0 1.0 Universal Public Domain Dedication](https://creativecommons.org/publicdomain/zero/1.0) for the metadata only
| Script to pull in arXiv PDFs in bulk | A third party script called [mattbierbaum/arxiv-public-datasets](https://github.com/mattbierbaum/arxiv-public-datasets) to pull in arXiv Dataset in bulk from arXiv's bulk access API or Google bucket or AWS bucket and to generate plain text | [MIT License](https://github.com/mattbierbaum/arxiv-public-datasets/blob/master/LICENSE) |
| Base Embed Model | [Linq-AI-Research/Linq-Embed-Mistral](https://huggingface.co/Linq-AI-Research/Linq-Embed-Mistral) | Creative Commons Attribution Non Commercial 4.0 |
| BART (facebook/bart-large-cnn or philschmid/bart-large-cnn-samsum) | Pretrained by Facebook on CNN/DailyMail and/or SAMSum datasets, a Large language model with encoder-decoder structure. Hosted on [huggingface](https://huggingface.co/facebook/bart-large). | MIT License (Samsum variant) or Fairseq license for BART. Free for research use. |
| etc          | | |


### Summary of infrastructure requirements

<!-- Itemize all your anticipated requirements: What (`m1.medium` VM, `gpu_mi100`),
how much/when, justification. Include compute, floating IPs, persistent storage.
The table below shows an example, it is not a recommendation. -->

| Requirement     | How many/when                                     | Justification |
|-----------------|---------------------------------------------------|---------------|
| `m1.medium` VMs | 1 for the entire project duration, 2 more during final setup | For hosting a 1 conventional DB, 1 for a vector DB and 1 as a proxy |
| `compute_liqid` node with 2 GPUs OR `gpu_mi100` node with 2 GPUs | 1 node with 2 a100s for 20 hours a week ( 2x 4 hour blocks and 2x 6 hour blocks a week ) | Hosting a Ray cluster for training and serving models |
| `compute_liqid` or `gpu_mi100` or `rtx8000` node with 1 GPU | For the entire project duration | For training and serving models under development |
| Persistent Storage | 120 GB for entire project duration | Needed to persist models, database, training datasets, metrics etc |
| Floating IPs | 1 for entire project duration | 1 for everything ( gateway, dashboards etc ) |

<!--
| `m1.medium` VMs | 3 for entire project duration                     | ...           |
| `gpu_mi100` GPUs | 4 hour block four times a week |               |
| Persistent Storage | 120 GB for project duration | Needed to persist datasets, models, metrics, etc. |
| Floating IPs    | 1 for entire project duration, 1 for sporadic use |               |
| etc             |                                                   |               |
-->

### Detailed design plan

<!-- In each section, you should describe (1) your strategy, (2) the relevant parts of the
diagram, (3) justification for your strategy, (4) relate back to lecture material,
(5) include specific numbers. -->



#### Model training and training platforms

## Unit 4 and Unit 5: Training

## Steps to run:
 
Once inside a jupyter notebook mounted on block storage and Ray cluster running using gpu
In Neura-Scholar/queries/openchat3.5 folder
Run Genrating_Storing _Queries_OpenChat3.5.ipynb to generate queries from chunks by using Openchat-3.5 model.
Then run phrases.ipnyb to convert queries into query_phrases


Next, in Neura-Scholar/Training/main folder
Run train_longformer_final_run.ipynb this will submit a ray train job and longformer-base-4096 model will be registered on MLflow and stored in minio and Postgres as well as checkpoingts being stored on /mnt/ directory. The registered model is getting stored with being named 'final' and version 1 

When retrain is triggered from argo workflows, a ray job is being submit using retrain_without_ray.py in Neura-Scholar/Retraining/ directory. The retrained model is is getting stored with being named 'final' and version 2.


Note: Please change the floating ip and port numbers according to your setup.


## Additional runs and experiments information:

### I) queries folder: queries generation
#### 1)Neura-Scholar/queries/openchat3.5
Queries generated by OpenChat3.5 and Query_phrases by keybert are saved in Neura-Scholar/queries/openchat3.5/Data/ 
References are stored in refrences.txt

#### 2)Neura-Scholar/queries/t5-large/
t5-large.ipynb and t5-large_storing _queries.ipynb is used to generate queries from chunks by using t5-large model
Queries generated are saved Neura-Scholar/queries/t5-large/t5-large.txt 
References are stored in refrences.txt

#### 3)Neura-Scholar/queries/t5-small/
t5 -small_queries_creating_2new tables.ipynb is used to generate queries from chunks by using t5-small model
Queries generated are saved Neura-Scholar/queries/t5-large/t5-small.txt 
References are stored in refrences.txt

#### 4)Neura-Scholar/queries/t5-xl/
t5-xl_storing_queries.ipynb is used to generate queries from chunks by using t5-xl model
Queries generated are saved Neura-Scholar/queries/t5-large/t5-xl.txt 
References are stored in refrences.txt

### II)Training folder:

#### In Neura-Scholar/Training/main->

Ray used
code->longformer_final_run.py
-jupyter notebook used to submit ray job->longformer_final_run
- version '1' of model 'final'
- experiment->final
- run->clumsy-yak-807
- outputs_screenshots has logs and screenshots of ray dashboard
- References are stored in refrences.txt


#### Neura-Scholar/Training/experiments/:

##### openchat3.5:

folder openchat3.5:
In Neura-Scholar/Training/experiments/openchat3.5/mlflow/
###### 1)mlflow  without ray
- jupyter notebook ->Train_mlflow
- version '1' of model 'arxiv-bi-encoder-distilbert'
- experiment->arxiv-bi-encoder-distilbert
- run->carefree-wren-174
- Used distilbert-uncased model
- References are stored in refrences.txt
###### 2)mlflow  without ray

a)mlflow  without ray
- jupyter notebook ->Train_mlflow_2
- version '1' of model 'distilbert-arxiv-bi-encoder1'
- experiment->distilbert-arxiv-bi-encoder1
- run->carefree-wren-174
- Used distilbert-uncased model

b)mlflow  without ray
- jupyter notebook ->Train_mlflow_2
- version '2' of model 'distilbert-arxiv-bi-encoder1'
- experiment->distilbert-arxiv-bi-encoder1
- run->vaunted-shark-785
- Used distilbert-uncased model
- Outputs are in Training/experiments/openchat3.5/mlflow/output
- References are stored in refrences.txt

##### folder2:
###### In Neura-Scholar/Training/experiments/openchat3.5/ray/
1)Ray used
In Training/experiments/openchat3.5/ray/1

a)Ray used
- code->longformer_8.py
- jupyter notebook used to submit ray job->run_ray_longformer_experiment
b)Ray used
- code->longformer_9.py
- jupyter notebook used to submit ray job->run_ray_longformer_experiment
c)Ray used
- code->longformer_10.py
- jupyter notebook used to submit ray job->run_ray_longformer_experiment

References are stored in refrences.txt

2)Ray used
In Training/experiments/openchat3.5/ray/2


a)Ray used

code->check1.py
- jupyter notebook used to submit ray job->run_ray_longformer_experiment_2
(small data to test code, it did not register properly as i was logging arttifacts twice)
- version '1' of model 'check'
- experiment->arxiv-bi-encoder-longformer-ray
- run->efficient-finch-1
- Used longformer-base-4096 model

b)Ray used
- code->final.py
- jupyter notebook used to submit ray job->run_ray_longformer_experiment_2
- (complet data, 3 epochs but doubted it will register properly as i was logging arttifacts twice, so stopped midway)
- version '2' of model 'check'
- experiment->final
- run->amazing-sloth-18
- Used longformer-base-4096 model
- References are stored in refrences.txt

t5-small:
data got deleted from mlflow but code is there 


Ray_tune folder:

Ray used
- code->train_tune_2.py
- jupyter notebook used to submit ray job->train_tune_2
- version '2' and '3' of model 'final'
- experiment->ray_tune
- run->enchanting-pug-21(lr=3e-05)('3)'   and  upbeat-boar-917(1e-05)('2') for different lr


### III)Retraining folder:

Ray job is being submited  ad ray train is being used is getting triggered by argo workflow 

code->retrain_with_ray.py
- version '2' of model 'final'
- experiment->final
- run->gentle-mule-713

- Output folder has logs and screenshot so ray dashboard
- References are stored in refrences.txt


#### Model serving and monitoring platforms

<!-- Make sure to clarify how you will satisfy the Unit 6 and Unit 7 requirements,
and which optional "difficulty" points you are attempting. -->

The embed and summarizer models are served as an API endpoint within a internal network running on GPU nodes ( or CPU nodes ) with a Triton backend. This endpoint is only accessible internally.

Both conventional DB ( i.e. SQL ) and Vector DB ( i.e. Faiss ) are served on CPU nodes as an API endpoint within the network.

These internal endpoints can be used to handle the system flow such as generating embeddings, vector DB lookups, SQL queries, and generating summaries.

A gateway node exposed to the public is used to handle requests to and from within the system.

Platforms used:
1. Docker to containerize software (i.e. DBs, models, monitoring platforms etc)
2. Triton backend for serving both models seperately in onnx format.
3. FastAPI to implement a gateway that interacts with serving models, and databases.

#### Data pipeline

<!-- Make sure to clarify how you will satisfy the Unit 8 requirements,  and which
optional "difficulty" points you are attempting. -->

A datapipeline that handles requests and transfer of information between Model endpoints, and the databases with low latency.

1. A persistent storage of 120 GB for the complete project.
2. A conventional DB ( i.e. SQL ) running on a m1.medium sharing a partition with the persistent storage of around 40GB ( Offline data ).
3. This conventional DB will house all of our dataset that is (a) the corpus of research papers in pdfs, and (b) it's metadata in text.
4. A Vector DB with it's partition of 30 GB to store embeddings of the corpus metadata. ( Offline data )
5. A second partition in our Vector DB of around 20 GB for gradual replacement of embeddings from a new model. ( Offline data )
6. Dataset for re-training ( Offline data ) will be used from the conventional DB itself.
7. The only Online data in our pipeline would be the user quries coming in as requests from the gateway, and the summaries generated by our summarizer model.

#### Continuous X

<!-- Make sure to clarify how you will satisfy the Unit 3 requirements,  and which
optional "difficulty" points you are attempting. -->

We plan to use PythonCHI for installation, configuration and deployment of our infrastructure, versioned as Git.

Implementation of a CI/CD pipeline to re-train, run tests and deploy our models.

Finally, setting up staging environment for the model deployment.



