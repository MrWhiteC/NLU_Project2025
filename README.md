# AI Chatbot for MITRE ATT&CK Threat Classification and Organizational Impact Analysis

## Overview

This project aims to develop an **AI-powered chatbot** that leverages **Natural Language Processing (NLP)** to classify cybersecurity news articles into **MITRE ATT&CK techniques** and assess their potential impact on organizations. The chatbot will utilize **fine-tuned transformer models** (e.g., BERT, GPT) for threat classification and integrate **Retrieval-Augmented Generation (RAG)** to provide organization-specific threat assessments. The goal is to automate threat intelligence, improve incident response times, and enhance decision-making for cybersecurity teams.


## Key Features
- **Automated Threat Classification:** The chatbot will classify cybersecurity news articles into MITRE ATT&CK techniques using fine-tuned NLP models.

- **Organization-Specific Impact Analysis:** By integrating RAG, the chatbot will retrieve relevant information from organizational documents to provide tailored threat assessments.

- **Real-Time Threat Analysis:** A web-based interface will allow users to input cybersecurity news and receive immediate feedback on ATT&CK techniques classification and organizational impact.

- **Comparative Model Analysis:** The project will compare the performance of different NLP models (e.g., BERT, GPT, LSTMs) in classifying cybersecurity threats and use the best one.


## Related Works

**1. Noise Contrastive Estimation-based Matching Framework for Low-Resource Security Attack Pattern Recognition**


## Methodology
**1. Data Preprocessing:** Cybersecurity-specific datasets, such as labeled security incidents and their corresponding MITRE ATT&CK technique labels, will be cleaned, tokenized, and formatted for model training. This ensures the data is compatible with the selected pre-trained models and ready for fine-tuning.

**2. Model Training:** Fine-tuned transformer models (e.g., BERT, GPT) will be trained on labeled cybersecurity datasets to classify news articles into ATT&CK techniques.

**3. RAG Integration:** The chatbot will use Retrieval-Augmented Generation to fetch relevant information from organizational documents, enabling it to provide organization-specific impact assessments.

**4. Web-Based Chatbot Interface:** A user-friendly chatbot web application will be developed to allow security analysts to input cybersecurity news and receive real-time threat analysis.

**5. Evaluation:** The system will be evaluated using metrics such as accuracy, precision, recall, and F1-score. Human evaluation by cybersecurity experts will also be conducted to assess practical effectiveness.


## Expected Results
- **Accurate Threat Classification:** The chatbot will accurately classify cybersecurity news into MITRE ATT&CK techniques.

- **Enhanced Threat Intelligence:** The integration of RAG will provide organization-specific impact assessments, improving decision-making for cybersecurity teams.

- **Model Comparison:** A comparative analysis of different NLP models will identify the most effective approach for ATT&CK classification.

- **Real-Time Web-Based Chatbot:** A functional web-based chatbot will be developed for real-time threat analysis and impact assessment.

- **Improved Incident Response:** The system will reduce manual effort in threat classification and improve incident response times.

- **Human Evaluation and Expert Feedback:** This qualitative feedback will refine the model, ensuring it meets the needs of the organization.


## Datasets

**1. [sarahwei/cyber_MITRE_technique_CTI_dataset_v16](https://huggingface.co/datasets/sarahwei/cyber_MITRE_technique_CTI_dataset_v16):** A dataset mapping security text to MITRE ATT&CK techniques.

**2. [sarahwei/cyber_MITRE_attack_tactics-and-techniques](https://huggingface.co/datasets/sarahwei/cyber_MITRE_attack_tactics-and-techniques):** A dataset with questions and answers related to the MITRE ATT&CK framework.

**3. [mrmoor/cyber-threat-intelligence](https://huggingface.co/datasets/mrmoor/cyber-threat-intelligence):** A dataset containing text on vulnerabilities and corresponding diagnostic solutions.

## Evaluation Metrics
- **Quantitative Metrics:** Accuracy, precision, recall, and F1-score will be used to evaluate the model's classification performance.

- **Human Evaluation:** Cybersecurity experts will assess the chatbot's practical accuracy and relevance in real-world scenarios.


## Challenges and Concerns
- **Data Quality and Availability:** Ensuring the availability of high-quality, labeled datasets for training.

- **Model Fine-Tuning Complexity:** Fine-tuning pre-trained models on domain-specific cybersecurity data.

- **Computational Resources:** Managing the computational resources required for training large transformer models.

- **RAG Integration:** Ensuring seamless integration of RAG for accurate impact assessments.

- **Scalability and Real-Time Performance:** Optimizing the system for real-time threat analysis and scalability.

- **Evaluation and Feedback:** Coordinating comprehensive human evaluation with cybersecurity experts to ensure the system’s practical effectiveness and relevance in real-world scenarios.

- **Ethical and Privacy Concerns:** Handling sensitive organizational data in compliance with privacy regulations.


## Summary of Progress
- **Problem Definition and Scope:** The problem of automating cybersecurity threat analysis has been defined.

- **Literature Review:** A review explored existing cybersecurity systems, highlighting challenges and opportunities in automation using MITRE ATT&CK.

- **Proposed Solution Design:** A solution integrating fine-tuned NLP models and RAG has been conceptualized.

- **Research Questions:** Key research questions have been defined.

- **Methodology Planning:** The methodology for the experiment has been outlined.

- **Expected Results:** The expected outcomes of the research have been defined.


## Next Steps
- **Data Preprocessing and Tokenization:** Prepare the datasets for model training.

- **Model Training and Fine-Tuning:** Fine-tune pre-trained models on cybersecurity datasets.

- **RAG Integration:** Implement the RAG approach for organization-specific impact assessments.

- **Web-Based Interface Development:** Develop a user-friendly web application for real-time threat analysis.

- **Evaluation and Testing:** Conduct thorough testing and evaluation of the system.


## Conclusion

This project aims to develop an **AI-powered chatbot** that automates the classification of **MITRE ATT&CK techniques** and provides organization-specific impact assessments. By leveraging **fine-tuned NLP models** and **RAG**, the system will enhance threat intelligence, improve incident response times, and support informed decision-making in cybersecurity operations.
