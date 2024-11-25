import pandas as pd
import stanza

class CoNLLUPreprocessing:
    def __init__(self, input_path='./input/C3_anonymized.csv', output_path='./preprocessed_dataset.conllu'):
        """
        Initialize the conll_u_preprocessing class with input and output paths.

        Parameters:
        input_path (str): Path to the input dataset.
        output_path (str): Path to the output conll-u file. Default is './output'.
        """
        # Path to the input dataset
        self.input_path = input_path

        # Path to the output conll-u file
        self.output_path = output_path

        # Read the input dataset
        self.input_data = self.read_input_dataset()

        # Create a Stanza pipeline for the English language
        self.nlp = stanza.Pipeline(lang='en', processors='tokenize,lemma,pos')

    
    def read_input_dataset(self):
        """
        Read the input dataset from the given path, keeping only the comment_text and constructive_binary columns.

        Returns:
        pd.DataFrame: The input dataset as a pandas DataFrame.
        """
        # Read the input dataset as a pandas DataFrame
        df = pd.read_csv(self.input_path)

        #Create a new DataFrame with only the comment_text and constructive_binary columns
        df_anno = df[['comment_text','constructive_binary']].copy()

        #Change the constructive binary column to int (1 or 0)
        df_anno['constructive_binary'] = df_anno['constructive_binary'].astype(int)

        return df_anno
    

    def clean_data(self):
        """
        Remove unusable rows from the `Constructive Comments Corpus (C3)` dataset,
        based on the insights found during analysis.

        Parameters:
        input_data (pd.DataFrame): The input dataset as a pandas DataFrame.

        Returns:
        pd.DataFrame: The cleaned dataset.
        """

        # Remove all rows with less than 10 characters
        data = self.input_data[self.input_data['comment_text'].str.len() > 10]

        # Remove all rows with Japanese characters
        data = data[~data['comment_text'].str.contains('[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff66-\uff9f]', regex=True)]

        return data


    def save_output_dataset(self, data):
        """
        Save the output dataset to the output path.
        """
        with open(self.output_path, 'w', encoding='utf-8') as f:
            # Iterate over each comment in the DataFrame
            for doc_id, comment in enumerate(data["comment_text"]):
                # Process the comment with the Stanza NLP pipeline
                doc = self.nlp(comment)
                
                # Iterate over each sentence in the processed document
                for sent_id, sentence in enumerate(doc.sentences):
                    # Add sentence-level metadata
                    f.write(f"# sent_id = {doc_id}_{sent_id}\n")
                    f.write(f"# text = {sentence.text}\n")
                    
                    # Write each token in the sentence in CoNLL-U format
                    for word in sentence.words:
                        f.write(f"{word.id}\t{word.text}\t{word.lemma}\t{word.upos}\t{word.xpos}\t{word.feats or '_'}\t"
                                f"{word.head}\t{word.deprel}\t_\t_\n")
                    
                    # Add a blank line after each sentence
                    f.write("\n")
