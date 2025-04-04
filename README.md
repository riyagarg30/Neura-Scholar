# Neura-Scholar

There are traditional search engines for papers like google Scholar which use indexing and word pair matching. While, our Machine Learning system ranks papers based on user query by using embedding model and Vector DB and also provides summaries.
The business metric to judge non ML systems with our ML system are click through rate and time taken to find desired result. 


### System diagram


![System Diagram](system%20diagram.png)


### Summary of outside materials
| Name | How it was Created | Conditions of Use |
|----------|----------|----------|
| arXiv Dataset | From arxiv-public-datasets. Derived from arXiv.org metadata and fulltext. Includes paper titles, abstracts, and fulltext. Link: https://github.com/mattbierbaum/arxiv-public-datasets | It is publicly available under arXiv’s terms of use. Free for academic/research purposes. |
| MiniLM (all-MiniLM-L6-v2)  | Trained by Microsoft using distillation of larger BERT models on general sentence-pair tasks. Available via Hugging Face. Link: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 | Apache 2.0 License. Free for commercial and academic use.  |
|BART (facebook/bart-large-cnn or philschmid/bart-large-cnn-samsum) |Pretrained by Facebook on CNN/DailyMail and/or SAMSum datasets. Large language model with encoder-decoder structure. Link: https://huggingface.co/facebook/bart-large |MIT License (Samsum variant) or Fairseq license for BART. Free for research use. |

### Summary of infrastructure requirements

### 🧠 Infrastructure Requirements

| **Requirement**     | **How Many / When**                                                                                                                                                          | **Justification**                                                                                                               |
|---------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| **m1.medium VMs**    | 2 till project presentation                                                                                                                                                   | One for Ray head node and one for experiment tracking (`MLflow` or `wandb` server)                                              |
| **gpu_mi100**        | - Session 1: MiniLM Train → 4 hours  <br> - Session 2: MiniLM Retrain → 4 hours  <br> - Session 3: BART Train → 4 hours  <br> - Session 4: BART Retrain → 4 hours  <br> - Session 5: Distributed Training Experiments → 6 hours  <br> - Session 6: Ray Tune → 4 hours  <br> - Session 7: Final run → 4 hours | Train and Retrain embedding model (MiniLM) and Summarization model (BART). <br> Also Distributed Training Experiments (DDP, FSDP, etc) <br> And Ray Tune <br> Final attempt to test everything |
| **Persistent Storage** | 120 GB for project duration                                                                                                                                                   | Needed to persist datasets, models, metrics, etc.                                                                                |
| **Floating IP**       | 2 until presentation                                                                                                                                                          | - VM 1: Ray Head Node + Ray Dashboard <br> - VM 2: `MLflow` (or `wandb`) Tracking Server                                        |

| Row2 C1  | Row2 C2  | Row2 C3  |
