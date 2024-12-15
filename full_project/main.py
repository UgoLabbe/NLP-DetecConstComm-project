import string
import warnings
import logging
import pandas as pd
from conllu_preprocessing import CoNLLUPreprocessing

import nltk
from nltk.corpus import stopwords

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.WARNING)

def main():
    # Make sure to download the stopwords from nltk
    nltk.download('stopwords')

    input_path = './input/C3_anonymized.csv'
    output_path = './output/preprocessed_dataset.conllu'

    stop_words = stopwords.words('english')
    punctuation = list(string.punctuation)

    # Remove the punctuation and stopwords
    # preprocessing_opt={ "stop_words":stop_words, "punctuation":punctuation }

    # No additional preprocessing
    preprocessing_opt={}

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

        print(f"File saved successfully on: {output_path}")

    except pd.errors.ParserError as e:
        print(f"Error creating CoNLL-U file: {e}")


if __name__ == "__main__":
    main()
