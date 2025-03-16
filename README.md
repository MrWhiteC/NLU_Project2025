# AI Chatbot for MITRE ATT&CK Threat Classification and Organizational Impact Analysis

## Overview

This project aims to develop an **AI-powered chatbot** that leverages **Natural Language Processing (NLP)** to classify cybersecurity news articles into **MITRE ATT&CK techniques** and assess their potential impact on organizations. The chatbot will utilize **fine-tuned transformer models** (e.g., BERT, GPT) for threat classification and integrate **Retrieval-Augmented Generation (RAG)** to provide organization-specific threat assessments. The goal is to automate threat intelligence, improve incident response times, and enhance decision-making for cybersecurity teams.


## Key Features
- **Automated Threat Classification:** The chatbot will classify cybersecurity news articles into MITRE ATT&CK techniques using fine-tuned NLP models.

- **Organization-Specific Impact Analysis:** By integrating RAG, the chatbot will retrieve relevant information from organizational documents to provide tailored threat assessments.

- **Real-Time Threat Analysis:** A web-based interface will allow users to input cybersecurity news and receive immediate feedback on ATT&CK techniques classification and organizational impact.

- **Comparative Model Analysis:** The project will compare the performance of different NLP models (e.g., BERT, GPT, LSTMs) in classifying cybersecurity threats and use the best one.


## Related Works
Cyber threat intelligence (CTI) and cybersecurity NLP research have advanced significantly, with various studies addressing different aspects of cyber threat detection, classification, and information extraction. Below are key related works that inform our approach to building an **AI-powered chatbot for MITRE ATT&CK threat classification and organizational impact analysis**.

**1. Noise Contrastive Estimation-based Matching Framework for Low-Resource Security Attack Pattern Recognition**

**Problem Statement:**  
- Classifying long-tailed, multi-label Tactics, Techniques, and Procedures (TTPs) is challenging due to missing labels and the complexity of attack patterns.  

**Key Contributions:**  
- Introduces a **Noise Contrastive Estimation (NCE)-based framework** for TTP classification.  
- Uses a **dual-encoder matching network (Siamese architecture)** to compare cybersecurity reports with predefined TTP profiles.  
- Employs **hierarchical multi-label learning** to classify both attack tactics and techniques.  
- Implements **negative sampling strategies** to enhance learning for cybersecurity threat classification.  

**Relevance to Our Project:**  
- This work demonstrates that **NCE-based learning paradigms** can improve **cybersecurity NLP models**, which aligns with our **fine-tuned classification approach**.

**2. A Pretrained Language Model for Cyber Threat Intelligence**

**Problem Statement:**  
- General-purpose NLP models struggle to understand cybersecurity-specific language and terminology.  

**Key Contributions:**  
- Develops **CTI-BERT**, a **domain-specific BERT model** trained on cybersecurity data.  
- Fine-tunes the model for multiple cybersecurity NLP tasks, including:  
  - **MITRE ATT&CK technique classification**  
  - **Malware sentence detection**  
  - **IoT app security analysis**  
  - **Cybersecurity-specific named entity recognition (NER)**  

**Relevance to Our Project:**  
- Since our chatbot needs to **classify cybersecurity news into MITRE ATT&CK techniques**, CTI-BERT’s **domain-specific pretraining** supports **higher accuracy and relevance in classification tasks**.

**3. AnnoCTR: A Dataset for Detecting and Linking Entities, Tactics, and Techniques in Cyber Threat Reports**

**Problem Statement:**  
- Existing cybersecurity datasets lack **fine-grained annotations** for MITRE ATT&CK techniques and do not support entity linking to structured knowledge bases.  

**Key Contributions:**  
- Introduces **AnnoCTR**, a publicly available dataset with:  
  - **Fine-grained cybersecurity entity annotations** (e.g., hacker groups, malware, ATT&CK techniques).  
  - **Explicit vs. implicit attack technique classification** (i.e., direct vs. inferred mentions).  
  - **Entity linking to MITRE ATT&CK and Wikipedia** for structured knowledge retrieval.  

**Relevance to Our Project:**  
- This dataset supports **more accurate ATT&CK technique classification**, which enhances our **training data for chatbot-based threat intelligence automation**.

**4. Introducing a New Dataset for Event Detection in Cybersecurity Texts**

**Problem Statement:**  
- Existing cybersecurity event detection (ED) datasets, such as **CASIE**, have only **5 event types** and lack **document-level context**, making them insufficient for real-world event detection.  

**Key Contributions:**  
- Introduces **CySecED**, a new dataset with **30 cybersecurity event types** (expanded from CASIE).  
- Compares **sentence-level vs. document-level models**, highlighting the importance of **context-aware event detection**.  
- Benchmarks models such as **BERT-ED, DEEB-RNN**, showing that existing models struggle with event detection.  

**Relevance to Our Project:**  
- Since our chatbot will **classify threats based on cybersecurity news**, insights from **document-aware architectures** will improve our **Retrieval-Augmented Generation (RAG) approach**.

**5. Full-Stack Information Extraction System for Cybersecurity Intelligence**

**Problem Statement:**  
- Extracting actionable intelligence from large volumes of **unstructured cybersecurity data** is difficult.  

**Key Contributions:**  
- Proposes an **end-to-end cybersecurity information extraction system** that:  
  - **Collects, preprocesses, and structures cybersecurity reports**.  
  - **Uses Named Entity Recognition (NER) and relationship extraction** to identify **IP addresses, malware, and attack types**.  
  - **Builds a knowledge graph** to map relationships between cybersecurity entities.  

**Relevance to Our Project:**  
- Our chatbot will **extract threat insights from cybersecurity news**, and this system’s **entity recognition techniques** align with our **automated threat classification goals**.

**6. Bootstrapping a Natural Language Interface to a Cybersecurity Event Collection System**

**Problem Statement:**  
- Many cybersecurity tools require **technical expertise**, making them **inaccessible to non-expert users**.  

**Key Contributions:**  
- Introduces a **hybrid rule-based and machine-learning approach** for **translating natural language queries** into structured cybersecurity event data.  
- Develops a **Natural Language Interface (NLI)** that simplifies querying cybersecurity event systems.  

**Relevance to Our Project:**  
- Since our chatbot will allow **security analysts to query threats using natural language**, this work informs our **conversational interface design**.


## Methodology
**1. Data Preprocessing:** Cybersecurity-specific datasets, such as labeled security incidents and their corresponding MITRE ATT&CK technique labels, will be cleaned, tokenized, and formatted for model training. This ensures the data is compatible with the selected pre-trained models and ready for fine-tuning.

**2. Model Training:** Fine-tuned transformer models (e.g., BERT, GPT) will be trained on labeled cybersecurity datasets to classify news articles into ATT&CK techniques.

**3. RAG Integration:** The chatbot will use Retrieval-Augmented Generation to fetch relevant information from organizational documents, enabling it to provide organization-specific impact assessments.

**4. Web-Based Chatbot Interface:** A user-friendly chatbot web application will be developed to allow security analysts to input cybersecurity news and receive real-time threat analysis.

**5. Evaluation:** The system will be evaluated using metrics such as accuracy, precision, recall, and F1-score. Human evaluation by cybersecurity experts will also be conducted to assess practical effectiveness.

![methodology_nlp drawio](https://github.com/user-attachments/assets/a885a9de-b47d-4fe8-9693-7890f04801b5)


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
