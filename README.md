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
- With the complexity, long-tailedness and huge number of multi-label classification of Tactics, Techniques and Procedures (TTPs), this could hinder the learning ability of the model.

**Key Contributions:**  
- Proposing a new learning paradigm on mapping the TTPs in classification problem.
- Introducing ranking-based Noise Contrastive Estimation (NCE) for handling large labelling together with missing labels problems.
- Curating and publicizing an export-annotation dataset.
- Conducting extensive experiments in their learning methods.

**Methodology:** 
- **Matching Network**: Dual-encoder framework was introduced for matching models through a **Siamese network**. The target text will be compared with the **TTP textual profile**.  
- **Learning to Match and Contrast**: **Dynamic label-informed text matching** was introduced. The ranker will act as the matcher where the higher probability of the match will be marked as **positive** and lower probability as **negative**.  
- **Sampling Strategies**: **Corpus-level negative sampling** was extremely useful for **discriminative mechanisms of the datasets**. Those negative sampling strategies could help to **show noisy samples** for cancellation during learning.  
- **Hierarchical Multi-label Learning**: An **auxiliary task** was introduced to predict the **tactics of the textual input** concurrently with the matching task.  
- **Experiments:** 
  - **Datasets**: Various datasets were tested to evaluate the learning process.  
  - **Metrics and Baselines**: Measured through **micro-average recall, precision, and F1 score** and **Mean Reciprocal Rank (MRR)** for ranking evaluation. Baselines included **Okapi BM25, Binary Relevance, Dynamic Triplet-loss, NAPKINXC**, and others.  
  - **Experimental Setup**: Based on a language model named **SecBERT**, **hyperparameters were tuned** using **grid search** and compared across **all baselines within auxiliary tasks**.  

**Conclusion:**
The proposed **NCE-based models** outperformed **other baselines** through dataset crafting, comparison, and hyperparameter tuning. This new learning paradigm integrates **inductive bias into classification tasks**, significantly improving **TTP classification**.

**Relevance to Our Project:**  
- This work demonstrates that **NCE-based learning paradigms** can improve **cybersecurity NLP models**, which aligns with our **fine-tuned classification approach**.

**2. A Pretrained Language Model for Cyber Threat Intelligence**

**Problem Statement:**  
- To **digest information** from many **Cyber Threat Intelligence (CTI) reports** within a limited time.  

**Key Contributions:**  
- Develop a **pre-trained BERT model** tailored for the **cybersecurity domain**.  
- Perform **extensive experiments** on **various cybersecurity NLP tasks**.  
- Curate a **large-scale cybersecurity dataset** specifically designed for **CTI analysis**.  

**Methodology:** 
- **Data Collection**: Collected from multiple sources including **academic papers, security wikis, threat reports, and vulnerability databases** to form a **diverse cybersecurity corpus**.  
- **Training**:  
  - Used **WordPiece tokenizer** to generate **50,000 tokens**.  
  - Applied **Masked Language Modeling (MLM)** for training.  
- **Evaluation**: The **CTI-BERT model** was evaluated against general-domain models (**BERT-base, RoBERTa**) and cybersecurity-specific models (**SecBERT, SecRoBERTa, SecureBERT**).  
  - **Masked Word Prediction**: Evaluates the **domain knowledge of models**.  
  - **Sentence Classification**: Used **classifier head (hidden layer & output projection layer)**.  
  - **ATT&CK Technique Classification**: Helps **Security Operation Center (SOC) teams** analyze reports.  
  - **IoT App Description Classification**.  
  - **Malware Sentence Detection**: Distinguishes **malware vs. non-malware sentences**.  
  - **Malware Attribute Classification**: Categorizes sentences using **MAEC (Malware Attribute Enumeration and Characterization)**.
- **Token Classification Tasks**: The model will be compared in the tokenization level as well in order to evaluate the effectiveness.
    - **NER1**: **Coarse-grained Security Entities** using **STIX-based labels (8 types)**.  
    - **NER2**: **Fine-grained Security Entities** with **16 entity types**.  
    - **Token Type Classification**: Categorizes tokens into **entity, action, and modifier**.  

**Conclusion:**
The proposed **CTI-BERT model** outperformed existing **general-domain** and **cybersecurity-domain** models.

**Relevance to Our Project:**  
- Since our chatbot needs to **classify cybersecurity news into MITRE ATT&CK techniques**, CTI-BERT’s **domain-specific pretraining** supports **higher accuracy and relevance in classification tasks**.

**3. AnnoCTR: A Dataset for Detecting and Linking Entities, Tactics, and Techniques in Cyber Threat Reports**

**Problem Statement:**  
- CTI relies on **analyzing large volumes of cyber threat reports (CTRs)** to identify attack tactics, techniques, and threat actors. However, most datasets **lack fine-grained annotations** for cybersecurity-specific entities and do not provide **structured links to MITRE ATT&CK**.   

**Key Contributions:**  
- **AnnoCTR Dataset**: **400 Cyber Threat Reports (CTRs)** with **120 reports** containing **detailed annotations**.  
- **Fine-Grained Cybersecurity Annotations**: Entities classified into **general named entities** and **cybersecurity-specific entities** (hacker groups, malware, tools, MITRE ATT&CK tactics & techniques).  
- **Explicit vs. Implicit Attack Technique Detection**: Identifies both **directly stated** and **contextually inferred techniques**.  
- **Entity Linking**: Maps tactics, techniques, and hacker groups to **MITRE ATT&CK** and general entities to **Wikipedia**.  
- **Benchmarking NLP Models**: Evaluated **BERT, RoBERTa, SciBERT, BLINK, and GENRE** for cybersecurity NLP tasks.  

**Methodology:**
- **Data Collection**: Sourced **CTRs from Intel471, Lab52, Proofpoint, QuoIntelligence, and ZScaler (2013-2022)**.  
- **Annotation Process**:  
  - **General Named Entities (GNEs)** labeled by engineers.  
  - **Cybersecurity-Specific Entities (CyNEs)** labeled by domain experts & validated by security professionals.  
- **Entity Linking**:  
  - **Wikipedia** (for general cybersecurity concepts).  
  - **MITRE ATT&CK** (for tactics, techniques, hacker groups).  
- **NLP Model Evaluation**:  
  - **Named Entity Recognition (NER)**: BERT, SciBERT, RoBERTa, BiLSTM-CRF.  
  - **Temporal Expression Tagging**: Rule-based HeidelTime vs. Neural Models with Domain Adaptation.  
  - **Entity Linking**: BLINK & GENRE.  
  - **Tactic & Technique Classification**: Few-shot learning models trained on **MITRE ATT&CK descriptions**.  
- **Performance Metrics**: Measured using **precision, recall, and F1-score**.

**Conclusion:**
**AnnoCTR is the first openly available cybersecurity dataset** with **fine-grained annotations** for **MITRE ATT&CK tactics and techniques**. Evaluation results show that:  
- **RoBERTa performs best** for **Named Entity Recognition (NER)**.  
- **Fine-tuned GENRE** is most effective for **Entity Linking**.  
- **Few-shot learning improves Tactic & Technique Classification**.  

AnnoCTR provides a **strong foundation** for **automated cyber threat intelligence processing**.

**Relevance to Our Project:**  
- This dataset supports **more accurate ATT&CK technique classification**, which enhances our **training data for chatbot-based threat intelligence automation**.

**4. Introducing a New Dataset for Event Detection in Cybersecurity Texts**

**Problem Statement:**  
- Cybersecurity event detection (ED) involves identifying trigger words that indicate cybersecurity events in text. Existing datasets, such as CASIE (Cybersecurity and Attack Strategies for Information Extraction), have limited event types (5 types) and do not consider document-level context, making them insufficient for real-world cybersecurity ED. To address these limitations, the paper introduces CySecED, a new dataset that covers more event types and requires document-level context for accurate detection.

**Key Contributions:**  
- **New Dataset (CySecED):** Introduces a manually annotated dataset with 30 cybersecurity event types, significantly expanding on CASIE.
- **Comprehensive Model Evaluation:** Compares sentence-level and document-level ED models, showing the importance of document context.
- **Challenges & Insights:** Identifies key challenges in cybersecurity event annotation, including trigger ambiguity and domain expertise requirements.
- **Future Research Directions:** Highlights the need for better document-aware architectures, domain-adapted embeddings, and event argument annotation.

**Methodology:** 
- **Dataset Collection:** Extracted cybersecurity news articles from The Hacker News (THN) and selected 30 event types based on the Simmons et al. (2014) cyberattack taxonomy.
- **Annotation Process:** Two annotators with security expertise labeled event trigger words, achieving Cohen’s Kappa = 0.79 after expert review.
- **Model Evaluation:** Tested sentence-level models (CNN, DMCNN, MOGANED, BERT-ED) and document-level models (HBTNGMA, DEEB-RNN) using word2vec and BERT embeddings.

**Conclusion:**
CySecED is a newly introduced dataset for cybersecurity event detection (ED), addressing the limitations of CASIE by expanding event types from 5 to 30 and incorporating document-level context. Evaluation of state-of-the-art models reveals that current ED systems struggle with CySecED, with the best model achieving only 68.4% F1, far below human performance (81.0% F1). The results highlight the need for document-aware architectures and domain-adapted embeddings to improve cybersecurity ED. Future work includes event argument annotation and the exploration of advanced deep learning models. CySecED will be publicly released to support further research in NLP for cybersecurity.

**Relevance to Our Project:**  
- Since our chatbot will **classify threats based on cybersecurity news**, insights from **document-aware architectures** will improve our **Retrieval-Augmented Generation (RAG) approach**.

**5. Full-Stack Information Extraction System for Cybersecurity Intelligence**

**Problem Statement:**  
- The problem addressed in this paper is the difficulty of managing and extracting actionable intelligence from large volumes of unstructured data (such as text reports, news articles, threat intelligence feeds) in the cybersecurity domain. Traditional methods for data extraction are often inefficient and unable to deal with the high volume, variety, and velocity of data in cybersecurity contexts. The need is for a system that can automatically extract relevant cybersecurity information, identify relationships and events, and present it in a structured and actionable form. 

**Key Contributions:**  
- **Full-Stack Information Extraction:** The paper introduces a comprehensive, end-to-end system for information extraction, which spans multiple layers of the process—from raw data collection, preprocessing, and entity recognition to knowledge graph generation and event analysis.
- **Information Extraction:** The system focuses on extracting not just entities (e.g., IP addresses, malware names) but also events (e.g., data breaches, attacks) and their relationships, which is vital for cybersecurity analysis.
- **Integration of Various Tools:** It integrates Natural Language Processing (NLP) and machine learning (ML) with domain-specific cybersecurity intelligence, enabling better context understanding and more accurate extraction.
- **Scalability and Automation:** The system is designed to scale with the growing amount of cybersecurity data, automating the extraction process to save time and human resources.

**Methodology:** 
- **Data Collection and Preprocessing:** The system starts with gathering raw, unstructured cybersecurity data from various sources like news, reports, and threat intelligence platforms. The data is then cleaned and standardized to ensure consistency.
- **Entity Recognition and Event Extraction:** Using machine learning-based techniques and NLP methods (such as Named Entity Recognition (NER)), the system extracts key entities like IP addresses, threat types, and attack methods.
- **Contextual Analysis:** The system applies domain-specific context to better understand and interpret the entities, identifying relationships between them (e.g., which attack type is linked to a specific vulnerability).
- **Knowledge Graph Construction:** Extracted data is used to build a dynamic knowledge graph that maps the relationships between entities and events in cybersecurity, offering an insightful view of ongoing or past incidents.

**Conclusion:**
The proposed system effectively automates the extraction and structuring of valuable cybersecurity intelligence from unstructured data, enhancing scalability and decision-making in cybersecurity operations, with future improvements focused on accuracy and adaptation to emerging threats.

**Relevance to Our Project:**  
- Our chatbot will **extract threat insights from cybersecurity news**, and this system’s **entity recognition techniques** align with our **automated threat classification goals**.

**6. Bootstrapping a Natural Language Interface to a Cybersecurity Event Collection System**

**Problem Statement:**  
- The paper tackles the challenge of enabling non-expert users to interact with complex cybersecurity event collection systems using natural language queries. Traditional query interfaces require technical expertise, which limits accessibility for non-technical stakeholders and hinders broader adoption of cybersecurity tools. This creates a significant barrier for organizations that rely on diverse teams to monitor and respond to cybersecurity events. The lack of a user-friendly interface prevents non-experts from effectively utilizing these systems, reducing overall efficiency and responsiveness in cybersecurity operations.
  
**Key Contributions:**  
- **Hybrid Translation Approach:** The paper introduces a hybrid approach that combines rule-based and machine learning methods for translating natural language queries into system queries or structured data in cybersecurity event collection systems.
- **Natural Language Interface for Cybersecurity:** By developing an NLI, the paper provides a user-friendly interface to interact with complex cybersecurity event systems, reducing the need for specialized knowledge.
- **Bootstrapping the Translation Process:** The paper introduces an innovative method for bootstrapping the translation process, which involves using an initial set of rules and examples to "train" the system to handle increasingly complex user queries.
- **Improved Usability for Security Analysts:** The interface simplifies the process of querying cybersecurity event data, making it more accessible to analysts without deep technical knowledge.

**Methodology:** 
- **Data Collection:** The researchers gather cybersecurity event data and related queries from existing systems. This forms the basis for training the hybrid model and understanding the requirements of the cybersecurity domain.
- **Rule-Based Translation:** Initially, a rule-based approach is used to translate predefined, simple queries into system-specific queries that can be processed by the cybersecurity event collection system.
- **Machine Learning Integration:** As more queries are processed, machine learning techniques are applied to refine and expand the system's understanding, enabling it to handle more complex, nuanced natural language queries.
- **Hybrid System Development:** A hybrid system is developed by combining the simplicity and interpretability of rule-based approaches with the flexibility and scalability of machine learning. This enables better accuracy and adaptability to various types of user queries.
- **User Interface:** Implements an intuitive NLI for users to input queries and receive results.

**Conclusion:**
The hybrid translation approach effectively enables non-expert users to query complex cybersecurity event systems using natural language, enhancing accessibility and adaptability. Future work could focus on expanding natural language capabilities, improving the system’s handling of ambiguous queries, and integrating with a wider range of cybersecurity tools and data sources.

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
