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

