## Project: Detection of Constructive Comments
### Milestone 1: Dataset Preparation and Preprocessing

## Overleaf for management report: 
https://www.overleaf.com/9393782685xkkwjwgqdrqs#28659f

**Dataset Overview**: 
- The data for this project is a subset of comments from the SFU Opinion and Comments Corpus. This subset, the Constructive Comments Corpus (C3) consists of 12,000 comments annotated by crowdworkers for constructiveness and its characteristics.
- Total records: 12,000

**Preprocessing Steps**:
- Using a stanza english pipeline: converted text to lowercase, removed punctuation, tokenized,lemmatized  and added Part of Speech (POS) tagging.

**CoNLL-U Format Export**:
- Stored data in CoNLL-U format, including `sent_id` and `text` annotations for each sentence.
- Custom code was implemented to meet format requirements.

**Potential Issues**:
- Managed multiple sentences per record, used Stanza pipeline for consistent NLP preprocessing.
- Uses of slang or abreviations
- Uses of link in comments

### Milestone 2: Implement NLP Solutions

**Implement Non-Deep Learning solutions**:
- Naive Bayesian classifier on both original and preprocessed text
- Feature based Machine Learning models:
    - K Nearest Neighbor
    - Random Forest (with SHAP values for explainability and qualitative analysis)
    - Logisitic Regression

**Implement Deep Learning solutions**:
- Neural Network

**Implement Transform-based solutions**:
- BERT

You can find the model performancs in the `model_perf.md`