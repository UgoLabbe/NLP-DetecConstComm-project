import argparse
import string
import warnings
import logging
import pandas as pd
from conllu_preprocessing import CoNLLUPreprocessing, RuleBasedAnnotations

import nltk
from nltk.corpus import stopwords

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.WARNING)

def download_stopwords():
    nltk.download('stopwords')

def run_pipeline(input_path, output_path, preprocessing_opt):
    stop_words = stopwords.words('english')
    punctuation = list(string.punctuation)

    try:
        # Initialize the CoNLLUPreprocessing class
        pipeline = CoNLLUPreprocessing(
            input_path=input_path,
            output_path=output_path,
            preprocessing_opt=preprocessing_opt
        )

        # Get the input data, and remove unusable rows
        cleaned_data = pipeline.remove_rows(pipeline.input_data)

        # Preprocess the data (remove stopwords, punctuation, etc.)
        preprocessed_data = pipeline.prior_preprocessing(cleaned_data)

        # Apply the Stanza pipeline to the preprocessed data and save it
        pipeline.save_output_dataset(preprocessed_data)

        print(f"Pipeline executed successfully. File saved on: {output_path}")

    except pd.errors.ParserError as e:
        print(f"Error creating CoNLL-U file: {e}")

def run_rule_based_annotations(preprocessed_file):
    try:
        rule_based_annotations = RuleBasedAnnotations(preprocessed_file)

        # Generate the rule-based annotations based on tokens, verbs, and adjetives
        rule_based_annotations.generate_feature_based_annotation(rule_based_annotations.conllu_data)
        # rule_based_annotations.generate_keywords_based_annotation(rule_based_annotations.conllu_data)

        print(f"Rule-based annotations generated successfully for: {preprocessed_file}")

    except Exception as e:
        print(f"Error generating rule-based annotations: {e}")

if __name__ == "__main__":
    # Example of usage:
    # python main.py --download_stopwords --preprocessing --annotations --input_path ./input/C3_anonymized.csv --output_path ./input/preprocessed_dataset.conllu --preprocessing_opt '{"stop_words":stop_words, "punctuation":punctuation}'
    # python main.py --preprocessing --annotations

    parser = argparse.ArgumentParser(description="Run the pipeline to preprocess C3_anonymized.csv into CoNLL-U format and provide our rule-based annotations.")
    parser.add_argument('--download_stopwords', action='store_true', help="Download NLTK stopwords.")
    parser.add_argument('--preprocessing', action='store_true', help="Run the NLP pipeline.")
    parser.add_argument('--preprocessing_opt', type=dict, default={}, help='Format: { "stop_words":stop_words, "punctuation":punctuation }. If `stop_words` are provided they will be removed from the text, stop_words should be a list of strings or the variable `stop_words` which is a list of stopwords from NLTK. If `punctuation` is provided it will be removed from the text, punctuation should be a list of strings or the variable `punctuation` which is a list of punctuation characters.')
    parser.add_argument('--annotations', action='store_true', help="Run rule-based annotations.")
    parser.add_argument('--input_path', type=str, default='./input/C3_anonymized.csv', help="Path of the original C3_anonymized.csv file.")
    parser.add_argument('--output_path', type=str, default='./input/preprocessed_dataset.conllu', help="Path to save the preprocessed CoNLL-U file.")

    args = parser.parse_args()

    if args.download_stopwords:
        download_stopwords()

    if args.preprocessing:
        run_pipeline(args.input_path, args.output_path, args.preprocessing_opt)

    if args.annotations:
        run_rule_based_annotations(args.output_path)
