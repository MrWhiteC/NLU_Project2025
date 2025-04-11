# AI System For MITRE ATT&CK Threat Classification And Organizational Impact Analysis

## Overview

This project introduces an **AI-powered Natural Language Processing (NLP) system** that automates the **classification** of **cybersecurity news articles** into **MITRE ATT&CK techniques**. A fine-tuned **BERT-based model**, specifically **CTI-BERT**, is used for the classification task, providing better performance than baseline models and traditional rule-based methods. The system is designed to support cybersecurity teams by identifying relevant ATT&CK techniques, **reducing the time, effort, and inconsistency associated with manual threat analysis**.


## Key Features

- **Automated Classification:** The BERT-based model automatically classifies cybersecurity news articles into relevant MITRE ATT&CK techniques, enhancing the efficiency of threat analysis.

- **Fine-tuning for Cybersecurity:** The model is fine-tuned specifically on cybersecurity-specific dataset, ensuring high relevance and accuracy in threat classification.

- **Improved Efficiency:** Reduces the time and effort spent on manual classification, helping security teams react faster to threats.

- **Evaluation Against Baseline Models:** The system's performance will be assessed against traditional keyword-based and rule-based methods, ensuring that it offers significant improvements in both accuracy and relevance.

- **Human Feedback Loop:** Feedback from cybersecurity experts will be integrated into the model evaluation, further enhancing its capabilities for practical, real-world applications.


## Related Works

This research builds on several notable works in the field of cybersecurity threat intelligence and natural language processing (NLP):

**1. Noise Contrastive Estimation-based Matching Framework for Low-Resource Security Attack Pattern Recognition (Nguyen et al., 2024)** introduced a ranking-based Noise Contrastive Estimation (NCE) to address challenges in multi-label classification of Tactics, Techniques, and Procedures (TTPs). While successful, it faced issues with training time and the scope of the MITRE ATT&CK framework.

**2. Automatic Mapping of Unstructured Cyber Threat Intelligence (Orbinato et al., 2022)** explored machine learning and deep learning methods for classifying Cyber Threat Intelligence (CTI) into MITRE ATT&CK categories. Their work showed that deep learning outperformed traditional approaches, but the complexity of natural language made precise classification challenging.

**3. A Pretrained Language Model for Cyber Threat Intelligence (Park & You, 2023)** developed a BERT-based model specifically for cybersecurity, showing it worked better than general models. However, the limited size of datasets made challenges for performance on unseen data.

**4. AnnoCTR: A Dataset for Detecting and Linking Entities, Tactics, and Techniques in Cyber Threat Reports (Lange et al., 2024)** provided a detailed dataset for specific NLP tasks related to MITRE ATT&CK, which will be useful for this research in automating threat classification from cybersecurity news.

**5. Introducing a New Dataset for Event Detection in Cybersecurity Texts (Duc Trong et al., 2020)** focused on improving event detection models, highlighting the limitations of sentence-level context in capturing complex cybersecurity events. This work emphasizes the need for document-level context, which the proposed system aims to use.

**6. Full-Stack Information Extraction System for Cybersecurity Intelligence (Park & Lee, 2022)** proposed a system to extract and organize cybersecurity information. However, it didn’t connect with standardized frameworks like MITRE ATT&CK, which led to disorganized data. This research aims to fix that by automating threat classification within the MITRE ATT&CK framework.


## Methodology

### Old Work Flow

**1.** Manual Search for Threat Information.  <br/>
**2.** Human Judgment & Filtering  <br/>
**3.** Split into Two Main Tasks
 - Task1: Security Awareness
  - summarized readable content
 - Task2: Threat Identification
  - Technique & Tactic Extraction: aligned with frameworks like MITRE ATT&CK
  - Pattern Matching
  - Notify Security Teams
  - Mitigation Planning
    
![Old_Workflow drawio](https://github.com/user-attachments/assets/517005a5-c912-4ebb-ae8f-05c82d26c106)


### New Work Flow

**1.** Select Models for Comparison. Three models selected for comparison are: BERT-Base-Uncased, CTI-BERT and Secure-BERT
**2.** Tokenizer and Word Embeddings
**3.** Training the Models: train 3 models with the same dataset and turn hyperparameters
**4.** Evaluate Performance
**5.** Select the Best Model
**6.** Pre-process the News or Articles
  - Method 1: Sentence-Level Processing: The text will be processed at sentence level, mapping the relation between the sentence and MITRE ATT&CK attack patterns.
  - Method 2: Clause-Level Processing: The text will be processed at clause level, where each clause is mapped to MITRE ATT&CK attack patterns.
**7.** Compare Accuracy of These Two Methods
**8.** Human Evaluation: A cybersecurity analyst or engineer will verify the classification results 


![methodology_nlp drawio-3](https://github.com/user-attachments/assets/2cf8fab1-753b-4225-8c12-f70f5fc3d37d)



## Preliminary Results

This section presents the preliminary results from experiments evaluating the performance of various BERT-based models in classifying cybersecurity-specific dataset into MITRE ATT&CK techniques. The models evaluated were:

- **BERT-base-uncased**
- **CTI-BERT**
- **Secure-BERT**

These models were evaluated using the following classification metrics:

- **Training Loss**
- **Validation Loss**
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

### Model Performance

The table below summarizes the performance of the three models based on the metrics mentioned:

| Metric              | BERT-base-uncased | CTI-BERT | Secure-BERT |
|---------------------|-------------------|----------|-------------|
| **Training Loss**   | 3.58              | 2.77     | 3.58        |
| **Validation Loss** | 3.44              | 2.71     | 3.31        |
| **Accuracy**        | 43.76%            | 54.64%   | 44.45%      |
| **Precision**       | 29.21%            | 43.17%   | 29.98%      |
| **Recall**          | 43.76%            | 54.64%   | 44.45%      |
| **F1-Score**        | 33.35%            | 46.61%   | 33.71%      |

From the table, it's clear that **CTI-BERT** outperforms the other models across all metrics. It achieved the lowest training loss (2.77) and validation loss (2.71), as well as the highest accuracy (54.64%)—about 10% higher than BERT-base-uncased. Additionally, it led in precision (43.17%), recall (54.64%), and F1-score (46.61%).

### Real-world Impact on Threat Intelligence

The expected outcome of this research is that fine-tuned BERT models, especially **CTI-BERT**, will significantly reduce the time needed for threat classification. By automating the process of classifying cybersecurity news into MITRE ATT&CK techniques, cybersecurity teams can quickly identify relevant attack patterns. This will help improve response times to emerging threats and lead to faster decision-making during incidents. 

Automating threat classification also reduces human error and avoids inconsistencies in manual threat mapping, helping security teams to make more accurate, efficient decisions. Ultimately, this will strengthen cybersecurity, optimize resource use, and improve protection against cyberattacks.


## Datasets

**1. [tumeteor/Security-TTP-Mapping](https://huggingface.co/datasets/tumeteor/Security-TTP-Mapping):** A dataset mapping security text to Tactics, Techniques, and Procedures (TTPs) to help identify attack patterns.

**2. [MITRE ATT&CK Dataset (Orbinato et al., 2022)](https://arxiv.org/pdf/2208.12144):** A collection of attack patterns and relationships mapped to MITRE ATT&CK techniques and tactics in JSON format.

**3. [Zainabsa99/mitre_attack](https://huggingface.co/datasets/Zainabsa99/mitre_attack):** A dataset translating MITRE ATT&CK IDs into a consumable format with attack details and detection techniques.


## Summary of Progress
- **Scope the project and system:** From the first proposed solution, there are multiple redundancy and out-of-scope task. This made the scope to be reduced and complying feasible scope within timeline and knowledge.

- **Corporate more related works:** In order to increase more relatable models, the paper were being reviewed more for ingesting the some potential model or methodology in proposed solutions.

- **Model Comparison:** Model with three comparisons will be evaluated in order to pick the best model for this task.

- **Datasets Exploration:** More datasets had been explored which could be potentially insert as a trained data for providing the most relatable results. 

- **Consultant with TA:** For scoping the topic, consultation with TA’s NLP had been done for get a clear picture and staying on the right track.


## Limitations and Challenges
- **Imbalanced and Limited Ground Truth Datasets:**
  - Difficulty in verifying model accuracy due to limited real-world datasets.
  - Data is unbalanced, which makes it harder to train the model properly.

- **Specialized Domain Limitations:**
  - Datasets come from specific areas, so they may not work well for other types of data.

- **Complexity of the MITRE ATT&CK Framework:**
  - The framework’s complexity may affect model's ability to fully capture its context.
  - Potential risk of inaccurate classification for new use cases.


## Next Steps
- **Keyword Extractions:** According to the TA’s comments, the main improve of the model is identifying the potential keywords.

- **Experiment on Pre-processing:** In order to extract the context from sentence, pre-processing methods will be experimented for selecting the best method to processing the sentence in order to improve accuracy.

- **System Evaluation:** After finishing all the coding, the evaluation process will be performed through a human and machine judgment as the final validation. 

- **Deployment:** The model will be deployed on the website allowing user to access for real-word use case.
