import pandas as pd
import stanza
from concurrent.futures import ThreadPoolExecutor, as_completed

class CoNLLUPreprocessing:
    def __init__(
            self, 
            input_path='./input/C3_anonymized.csv',
            output_path='./output/preprocessed_dataset.conllu',
            preprocessing_opt={}
        ):
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

        # Depine aditional preprocessing options
        self.preprocessing_opt = preprocessing_opt

    
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
    

    def remove_rows(self, data):
        """
        Remove unusable rows from the `Constructive Comments Corpus (C3)` dataset,
        based on the insights found during analysis.

        Parameters:
        data (pd.DataFrame): The input dataset as a pandas DataFrame.

        Returns:
        pd.DataFrame: The dataset without unusable rows.
        """

        # Remove all rows with less than 10 characters
        cleaned_data = data[data['comment_text'].str.len() > 10]

        # Remove all rows with Japanese characters
        cleaned_data = cleaned_data[~cleaned_data['comment_text'].str.contains('[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff66-\uff9f]', regex=True)]

        return cleaned_data


    def prior_preprocessing(self, data):
        """
        Perform any prior preprocessing steps before sending the data to the NLP pipeline.

        Parameters:
        data (pd.DataFrame): The input dataset as a pandas DataFrame.

        Returns:
        pd.DataFrame: The preprocessed dataset.
        """

        remove = []

        # Remove stopwords from the comments
        if 'stop_words' in self.preprocessing_opt and self.preprocessing_opt['stop_words']:
            remove.extend(self.preprocessing_opt['stop_words'])
        
        # Remove punctuation from the comments
        if 'punctuation' in self.preprocessing_opt and self.preprocessing_opt['punctuation']:
            remove.extend(self.preprocessing_opt['punctuation'])
    
        # Remove specific words from the comments
        if len(remove) > 0:
            data['comment_text'] = data['comment_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (remove)]))
        
        return data


    def process_comment(self, doc_id, comment, pipeline):
        """
        Process a single comment with the Stanza NLP pipeline and return the formatted lines.

        Parameters:
        doc_id (int): The document ID.
        comment (str): The comment to process.

        Returns:
        tuple: A tuple containing the document ID and the formatted lines.
        """
        doc = pipeline(comment)
        lines = []

        # Use list comprehension to accumulate lines
        lines.extend(
            [f"# sent_id = {doc_id}_{sent_id}\n"
             f"# text = {sentence.text}\n" +
             ''.join(f"{word.id}\t{word.text}\t{word.lemma}\t{word.upos}\t{word.xpos}\t{word.feats or '_'}\t"
                     f"{word.head}\t{word.deprel}\t_\t_\n" for word in sentence.words) +
             "\n"
             for sent_id, sentence in enumerate(doc.sentences)]
        )

        return doc_id, lines

    def save_classes(self, classes, class_output_path='./output/classes.txt'):
        """
        Save the constructive_binary classes to a separate file.

        Parameters:
        classes (list): List of tuples containing doc_id and constructive_binary.
        class_output_path (str): Path to the output classes file. Default is './output/classes.txt'.

        Returns:
        None (saves the output to the class_output_path)
        """
        classes.sort(key=lambda x: x[0])

        with open(class_output_path, 'w', encoding='utf-8') as f:
            f.write("doc_id\tconstructive\n")
            for doc_id, constructive_binary in classes:
                f.write(f"{doc_id}\t{constructive_binary}\n")

    def save_output_dataset(self, data):
        """
        Apply preprocessing to the data, and format it as a CoNLL-U file.

        Parameters:
        data (pd.DataFrame): The input dataset as a pandas DataFrame.

        Returns:
        None (saves the output to the output_path)
        """

        pipeline = stanza.Pipeline(lang='en', processors='tokenize,lemma,pos')

        output_lines = []
        batch_size = 1000

        with ThreadPoolExecutor(6) as executor:
            futures = {executor.submit(self.process_comment, doc_id, comment, pipeline): doc_id for doc_id, comment in enumerate(data["comment_text"])}

            results = []
            classes = []
            for future in as_completed(futures):
                doc_id = futures[future]
                try:
                    doc_id, lines = future.result()
                    results.append((doc_id, lines))
                    classes.append((doc_id, data["constructive_binary"].iloc[doc_id]))
                    print(f"Processed document {doc_id}...")
                except Exception as exc:
                    print(f"Document {doc_id} generated an exception: {exc}")

            # Sort results by doc_id
            results.sort(key=lambda x: x[0])

            with open(self.output_path, 'w', encoding='utf-8') as f:
                for doc_id, lines in results:
                    output_lines.extend(lines)

                    # Write to file if batch size is reached and reset the list
                    if len(output_lines) >= batch_size:
                        f.writelines(output_lines)
                        output_lines = []

                # Write any remaining lines to the file
                if output_lines:
                    f.writelines(output_lines)

            # Save the classes to a separate file
            self.save_classes(classes)
