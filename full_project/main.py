import warnings
import logging
from conllu_preprocessing import CoNLLUPreprocessing

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.WARNING)

def main():
    input_path = './input/C3_anonymized.csv'
    output_path = './output/preprocessed_dataset.conllu'

    processor = CoNLLUPreprocessing(input_path=input_path, output_path=output_path)

    cleaned_data = processor.clean_data()

    processor.save_output_dataset(cleaned_data)

    print(f"File saved successfully on: {output_path}")

if __name__ == "__main__":
    main()
