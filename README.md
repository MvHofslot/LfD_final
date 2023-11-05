# Offensive Language Detection Project

## Overview

The pervasive influence of social media in our daily lives has brought to the critical need for effective mechanisms to identify and mitigate offensive language within online discourse. This project focuses on the development of robust tools for offensive language detection in online content, particularly tweets, with the goal of creating safer and more respectful online environments.

## Motivation

The task of offensive language identification has garnered significant attention from researchers. In 2019, the "Shared Task on Offensive Language Identification" sparked a collective effort to address the challenges of categorizing and understanding offensive language within social media contexts. Various researchers participated in this task.

While most previous results achieved by researchers were around 70%, we aimed to find a more effective method for offensive language detection. We experimented with different classical models, LSTM, and large language models, and after fine-tuning, we achieved promising results, with accuracy scores of 80%, 82.7%, and 84.2% respectively.

## Research Questions

Based on the knowledge that large pre-trained language models have cross-lingual capabilities and that monolingual models can be transferred to low-resource languages effectively, we aimed to explore the relationship between different languages and their performance in offensive language detection. We hypothesized that languages more similar to English, such as Dutch and German, would outperform languages like Chinese and Japanese. To test this, we compared the results of monolingual and multilingual models trained on 10%, 50%, and 100% of the training data.

## Data Download

You can download the data used in this project from the following links, click on the links to download the files:

- [train.tsv](train.tsv)
- [dev.tsv](dev.tsv)
- [test.tsv](test.tsv)

## Getting Started

To get started with this project, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/MvHofslot/LfD_final.git

2. Set up your Python environment:
   
   ```bash
   pip install tensorflow
   pip install transformers

3. Run the provided scripts to train and evaluate the models.

## Team Contributions

- **Matthijs:** Primarily responsible for the 'classic models' aspect.
- **Yuwen:** Focused on the 'LSTM' section.
- **Xuanyi:** Led the 'Language Models' section.

For the research question, the three of us collaborated on the training of 2-3 monolingual and multilingual models respectively.
