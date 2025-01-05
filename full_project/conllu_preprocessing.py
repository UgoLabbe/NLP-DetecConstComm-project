import stanza
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import load_conllu_data, extract_preprocessed_text

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


    def save_classes(self, classes, class_output_path='./input/old_annotations.txt'):
        """
        Save the constructive_binary classes to a separate file.

        Parameters:
        classes (list): List of tuples containing doc_id and constructive_binary.
        class_output_path (str): Path to the output classes file. Default is './input/old_annotations.txt'.

        Returns:
        None (saves the output to the class_output_path)
        """
        classes.sort(key=lambda x: x[0])

        with open(class_output_path, 'w', encoding='utf-8') as f:
            # f.write("doc_id\tconstructive\n")
            for doc_id, constructive_binary in classes:
                # f.write(f"{doc_id}\t{constructive_binary}\n")
                f.write(f"{constructive_binary}\n")


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


class RuleBasedAnnotations:

    def __init__ (self, 
                  preprocessed_conllu_path='./input/preprocessed_dataset.conllu',
                  output_path='./input/new_annotations.txt',
                  vocabulary_constructive='./input/vocabulary_constructive.txt',
                  vocabulary_non_constructive='./input/vocabulary_non_constructive.txt'
        ):
        """
        Initialize the rule_based_annotations class with input and output paths.
        """

        # Path to the preprocessed conll-u file
        self.preprocessed_conllu_path = preprocessed_conllu_path

        # Path to the output classes file
        self.output_path = output_path

        # Load count of verbs, adverbs, etc from conllu data
        self.conllu_data = load_conllu_data(self.preprocessed_conllu_path)

        # Load texts from conllu data
        self.conllu_texts = extract_preprocessed_text(self.preprocessed_conllu_path)

    
    def generate_feature_based_annotation(self, data):
        """
        Perform rule-based annotation on the preprocessed dataset based on the features of the conllu data (#tokens, #verbs, and #adjectives)

        Parameters:
        data (list): The preprocessed conllu dataset.

        Returns:
        list: The annotated dataset.
        """

        data['constructive'] = data.apply(constructive_based_on_features, axis=1)

        # Keep only comment_id and constructive
        data = data[['constructive']]

        # Rename comment_id to doc_id
        data.rename(columns={'comment_id': 'doc_id'}, inplace=True)

        # Save the annotations to a file
        data.to_csv(self.output_path, sep='\t', index=False, header=False)

        return data


    def generate_keywords_based_annotation(self, data):
        """
        Perform rule-based annotation on the preprocessed dataset based on the presence of specific keywords for each class, plush features of the conllu data (#tokens, #verbs, and #adjectives).

        Parameters:
        data (list): The preprocessed conllu dataset.

        Returns:
        list: The annotated dataset.
        """

        data['constructive'] = data.apply(construcive_base_on_keywords_and_features, axis=1, args=(self.vocabulary_constructive, self.vocabulary_non_constructive))

        # Keep only comment_id and constructive
        data = data[['constructive']]

        # Rename comment_id to doc_id
        data.rename(columns={'comment_id': 'doc_id'}, inplace=True)

        # Save the annotations to a file
        data.to_csv(self.output_path, sep='\t', index=False, header=False)

        return data


def constructive_based_on_features(row):
    if row['num_verbs'] >= 10 and row['num_adjectives'] >= 8 and row['num_tokens'] >= 300:
        return 1  # Constructive
    else:
        return 0  # Not constructive


def constructive_based_on_keywords(row, vocabulary_constructive, vocabulary_non_constructive):
    # Count the number of constructive and non-constructive words in the comment
    constructive_count = sum(row['comment_text'].count(word) for word in vocabulary_constructive)
    non_constructive_count = sum(row['comment_text'].count(word) for word in vocabulary_non_constructive)

    # Calculate the ratio of constructive to non-constructive words
    if non_constructive_count > 0:
        return constructive_count / non_constructive_count # if > 1, constructive else non-constructive
    else:
        return 1 if constructive_count > 0 else 0  # Constructive if count > 0, else Not constructive


def construcive_base_on_keywords_and_features(row):
    features_weight = constructive_based_on_features(row)
    keywords_weight = constructive_based_on_keywords(row)

    features_weight = features_weight * .70 # 70% weight
    keywords_weight = keywords_weight * .30 # 30% weight

    if features_weight + keywords_weight >= 0.5:
        return 1 # Constructive
    else:
        return 0 # Not constructive