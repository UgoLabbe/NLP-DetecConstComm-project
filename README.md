## Project: Detection of Constructive Comments

### Overview
This project focuses on detecting constructive comments using various NLP and machine learning approaches. The codebase is organized with the main functionality in main.py and supporting utilities in utils.py.

**Dataset Overview**: 
- The data for this project is a subset of comments from the SFU Opinion and Comments Corpus. This subset, the Constructive Comments Corpus (C3) consists of 12,000 comments annotated by crowdworkers for constructiveness and its characteristics.
- Total records: 12,000

### Main Components

The `main.py` file serves as the primary entry point for the project, loading and handling:

- Data preprocessing and feature extraction
- Model training
- Model evaluation 

when executing the python file, you have the choice to select different labels: 
- the original labels (done by crowdworkers), this will also be more in depth, with more models and a SHAP value analysis.
- the rule based labels
- the rule based labels weighted by specific keywords

In order to have all the required modules to run the code, you can run : `pip install -r requirements.txt`

You can find the model performances on the original labels in the `model_perf.md`

### Archive and previous steps

In the archive folder you can find: 

The `Milestone1.ipynb` where you can find: 

**Exploratory data analysis**:
- Plots on some of the dataset statistics
- Distribution of certain words

**Preprocessing Steps**:
- Using a stanza english pipeline: converted text to lowercase, removed punctuation, tokenized, lemmatized  and added Part of Speech (POS) tagging.

**CoNLL-U Format Export**:
- Stored data in CoNLL-U format, including `sent_id` and `text` annotations for each sentence.
- Custom code was implemented to meet format requirements.

**Potential Issues identification**:
- Managed multiple sentences per record, used Stanza pipeline for consistent NLP preprocessing.
- Uses of slang or abreviations
- Uses of link in comments

**Cration of labels**:
- Using a basic set of rules
- Using the same set of rules, weighted by specific keywords

The `Milestone2.ipynb` where you can find: 

**Model Training and Evaluation**:
- Naive Bayesian Classifier
- Feature Based models (Logistic Regression, Random Forest, k-NN)
- Neural Network
- BERT

**Qualitative Analysis**:
- SHAP Values analysis
- Manual analysis of misclassified comments

The `analysis_labels.ipynb` where you can find: 

- The difference between the different labels
- How classification differs from a label to another

### Presentation and Management summary

- The final presentations slides can be found on the file `NLP_Constructive_Detection.pdf`
- The management summary can be found on the file `NLP_Management_Summary.pdf`


