## Project: Detection of Constructive Comments
### Milestone 1: Dataset Preparation and Preprocessing

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

**Implement Deep Learning solutions**:
