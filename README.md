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

1. Manual Search for Threat Information
2. Human Judgment & Filtering
3. Threat Identification
  - Technique & Tactic Extraction: aligned with frameworks like MITRE ATT&CK
  - Pattern Matching
  - Notify Security Teams
  - Mitigation Planning
    
![Old_Workflow drawio](https://github.com/user-attachments/assets/9a024389-6530-4955-9e18-00430f973482)



### New Work Flow

1. Select Models for Comparison. Three models selected for comparison are: BERT-Base-Uncased, CTI-BERT and Secure-BERT
2. Tokenizer and Word Embeddings
3. Training the Models: train 3 models with the same dataset and turn hyperparameters
4. Evaluate Performance
5. Select the Best Model
6. Utlized the QACG-BERT with previous trained model
7. Pre-process the News or Articles
8. Evaluation: A predifined groud truth will be used to evluate the classification results 


![methodology_nlp drawio](https://github.com/user-attachments/assets/2903b294-0f0b-4437-87d3-c3e05f8c5549)



## Results

This section presents the results from experiments evaluating the performance of various BERT-based models in classifying cybersecurity-specific dataset into MITRE ATT&CK techniques. The models evaluated were:

- **BERT-base-uncased**
- **CTI-BERT**
- **Secure-BERT**

After that, the datasets will be modified to add the context inside which according to the QACGBERT model for evluating with groud truth.

### Standard Classification Metrics

The table below summarizes the performance of the four models based on the metrics mentioned:

| Metric              | BERT-base-uncased | CTI-BERT | Secure-BERT |   QACGBERT  |
|---------------------|-------------------|----------|-------------|-------------|
| **Training Loss**   | 3.58              | 2.77     | 3.58        | 1.706076    |
| **Validation Loss** | 3.44              | 2.71     | 3.31        | 4.723936    |
| **Accuracy**        | 43.76%            | 54.64%   | 44.45%      | 0.168077    |
| **Precision**       | 29.21%            | 43.17%   | 29.98%      | 0.163138    |
| **Recall**          | 43.76%            | 54.64%   | 44.45%      | 0.168077    |
| **F1-Score**        | 33.35%            | 46.61%   | 33.71%      | 0.144451    |


### Normalized Discounted Cumulative Gain (NDCG) Metric

The groud truth will be evluated base on ChatGPT classification with 10 samples on news or articles.


| Metric                     | CTI-BERT | QACGBERT |
|----------------------------|--------- |----------|
| **NDCG Score wit k = 20**  | 0.0000   | 0.0163   | 


## Datasets

**1. [tumeteor/Security-TTP-Mapping](https://huggingface.co/datasets/tumeteor/Security-TTP-Mapping):** A dataset mapping security text to Tactics, Techniques, and Procedures (TTPs) to help identify attack patterns.


## Repository Description
- This repository consist of model training, evaluation ,and website deployment. Here are some expliantion about each component:
  - Model Training
  - Evaluation
  - Website Deployment


## Limitations and Challenges
- **Accuracy Issues: Low accuracy due to**:
  - Insufficient or imbalanced dataset.
  - Overfitting and model’s difficulty generalizing to new attack types.

- **Complex Attack Descriptions**: Variability in attack styles and terminologies.

- **Data Quality & Size**: Inadequate coverage of attack techniques, poor learning of certain attack behaviors.

- **Language Limitation: Currently only supports English**, restricting global usability.


## Conclusion & Future Works
- **Conclusion:** The primary goal of this research was to reduce the manual effort required for analyzing cybersecurity news and improve the consistency and accuracy of threat classification. While the system shows potential, the current low accuracy, particularly during the evaluation phase, highlights the need for further development. The model can understand the context and language of cybersecurity news, but it still needs more improvement to ensure reliable and consistent threat detection. Moving forward, improving the model’s accuracy remains a crucial focus, which will enhance its practical application in real-world cybersecurity scenarios..

- **Future Works :**
  - Expand Dataset: Include more types of attacks, descriptions, and tough examples for robustness.
  - Fine-tune Model: Use more specific cybersecurity data for better understanding.
  - Multilingual Support: Add support for different languages to broaden system’s applicability.
  - Integration with Live Threat Feeds: Real-time classification of new threats by adapting to evolving attack methods.


