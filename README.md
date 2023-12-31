# Offensive Language Detection

## Overview

The pervasive influence of social media in our daily lives has brought to the critical need for effective mechanisms to identify and mitigate offensive language within online discourse. This project focuses on the development of robust tools for offensive language detection in online content, particularly tweets, with the goal of creating safer and more respectful online environments.

## Motivation

The task of offensive language identification has garnered significant attention from researchers. In 2019, the "Shared Task on Offensive Language Identification" sparked a collective effort to address the challenges of categorizing and understanding offensive language within social media contexts. Various researchers participated in this task.

While most previous results achieved by researchers were around 70%, we aimed to find a more effective method for offensive language detection. We experimented with different classical models, LSTM, and large language models, and after fine-tuning, we achieved promising results, with accuracy scores of 80%, 82.7%, and 84.2% respectively.

## Research Questions

Based on the knowledge that large pre-trained language models have cross-lingual capabilities and that monolingual models can be transferred to low-resource languages effectively, we aimed to explore the relationship between different languages and their performance of corresponding monolingual models in offensive language detection. We compared the results of monolingual and multilingual models trained on 10%, 50%, and 100% of the training data.

## Data Download

You can download the data used in this project from the following links, click on the links to download the files:

- [train.tsv](data/train.tsv)
- [dev.tsv](data/dev.tsv)
- [test.tsv](data/test.tsv)

## Getting Started

To get started with this project, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/MvHofslot/LfD_final.git

2. Set up your Python environment:
   
   ```bash
   pip install tensorflow
   pip install transformers

3. For the LSTM model, you will need to download pretrained embeddings from GloVe or FastText. You can download these embeddings from the following links:

- [Glove pretrained embeddings](https://nlp.stanford.edu/projects/glove/)
- [fastText pretrained embeddings](https://fasttext.cc/docs/en/english-vectors.html)

4. Run the provided scripts to train and evaluate the models.

Additionally, 'preprocess.py' is used to remove all the '@USER' in the text and 'randomSelect.py' is used to randomly select 10% and 50% of the data in the training text for experiments. They are placed in the 'data preprocessing' folder.

Running LM.py with the argument -l <language> allows you to choose which language model to use (for example, "python LM.py -l it" runs the Italian model).

## Team Contributions

- **Matthijs:** Primarily responsible for the 'classic models' aspect.
- **Yuwen:** Focused on the 'LSTM' section.
- **Xuanyi:** Led the 'Language Models' section.

For the research question, the three of us collaborated on the training of 2-3 monolingual and multilingual models respectively.
