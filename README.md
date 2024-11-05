For Milestone 1 of the Project exercise your team should gather the dataset(s) they are planning to use, perform standard preprocessing steps and INSPECT THE RESULTS to uncover potential issues that need to be handled. 
Finally, datasets should be stored in CoNLL-U format and pushed to the repository together with a short documentation of how the data was created.

Data: https://www.kaggle.com/datasets/mtaboada/c3-constructive-comments-corpus/data?select=C3_anonymized.csv


## Project: Detection of Constructive Comments
## Milestone 1: Dataset Preparation and Preprocessing

**Dataset Overview:** 
- Description of dataset source and purpose.
- Total records: 11,999

**Preprocessing Steps:**
- Converted text to lowercase, removed punctuation, tokenized, lemmatized, and applied POS tagging.

**CoNLL-U Format Export:**
- Stored data in CoNLL-U format, including `sent_id` and `text` annotations for each sentence.
- Custom code was implemented to meet format requirements.

**Potential Issues:**
- Managed multiple sentences per record, used Stanza pipeline for consistent NLP preprocessing.
