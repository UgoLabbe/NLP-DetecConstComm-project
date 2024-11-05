
## Project: Detection of Constructive Comments
### Milestone 1: Dataset Preparation and Preprocessing

**Dataset Overview**: 
- Description of dataset source and purpose.
- Total records: 11,999

**Preprocessing Steps**:
- Converted text to lowercase, removed punctuation, tokenized, lemmatized, and applied POS tagging.

**CoNLL-U Format Export**:
- Stored data in CoNLL-U format, including `sent_id` and `text` annotations for each sentence.
- Custom code was implemented to meet format requirements.

**Potential Issues**:
- Managed multiple sentences per record, used Stanza pipeline for consistent NLP preprocessing.
