Management summary Your submission must be accompanied by a 2-page PDF document that presents a summary of your solution — this is a management summary, so it should be written in a way that is easy to understand by top management, not NLP colleagues. The summary should contain an overview of the task, the challenges you faced, the external resources you used, the solution you implemented and its limitations, and possible next steps.




Final Presentation Each group will present the main results of their work to all other groups working on the same topic. The format is 20 minutes of presentation and 10 minutes of discussion — we will be very strict with the timing, and stop the presentation at the 20 minute mark. Each team member must present their own contributions to their project, so that they can be evaluated individually. The presentation should be aimed at NLP colleagues, so highlight which approaches and techniques you used, which datasets you used, and the insights obtained. Presentation slides must be pushed to your project repository the day before the presentations. The schedule of presentations will be announced via TUWEL, please attend all presentations in your



Topic: Detection of Constructive Comments (dataset: https://www.kaggle.com/datasets/mtaboada/c3-constructive-comments-corpus/data?select=C3_anonymized.csv)

We thought it could be interesting to explore the semantic starting from sentences including certain words (like “suggest,” “improve,” “consider”) to a more complex system of detection (POS, DL models etc...). 



For Milestone 1 of the Project exercise your team should gather the dataset(s) they are planning to use, perform standard preprocessing steps and INSPECT THE RESULTS to uncover potential issues that need to be handled. Finally, datasets should be stored in CoNLL-U format and pushed to the repository together with a short documentation of how the data was created.

For Milestone 2 of the Project exercise you should perform error analysis on your baseline models and discuss your findings, including implications for how your solution could be improved

import string
import warnings
import logging
import pandas as pd
from utils import load_conllu_data, extract_original_text, extract_preprocessed_text
import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import shap
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.WARNING)

INPUT_PATH = './input/preprocessed_dataset.conllu'

OLD_ANNOTATIONS_PATH = './input/old_annotations.txt'
NEW_ANNOTATIONS_PATH = './input/new_annotations.txt'
NEW_ANNOTATIONS_PATH_ONLY_FEATURES = './input/new_annotations_features_only.txt'

OLD_RANDOM_FOREST_MODEL_PATH = './models/rf_model.pkl'
OLD_NAIVE_BAYES_MODEL_PATH = './models/original_naive_bayes_model.pkl'

NEW_RANDOM_FOREST_MODEL_PATH = './models/rf_model_new_labels_rb.pkl'
NEW_BAYESIAN_MODEL_PATH = './models/naive_bayes_model_new_labels_rb.pkl'

NEW_RANDOM_FOREST_MODEL_ONLY_FEATURES_PATH = './models/rf_model_new_labels_features_only.pkl'
NEW_BAYESIAN_MODEL_ONLY_FEATURES_PATH = './models/naive_bayes_model_new_labels_features_only.pkl'

def main():

    print("\n" + "="*60)
    print(" Welcome to the TU Wien Winter Term NLP Project ".center(60, " "))
    print(" By Bernal Nicolas, Labbé Ugo, and Karbeutz Gerhard ".center(60, " "))
    print("="*60 + "\n")

    print("This project aims to classify constructive comments from a subset of the SFU Opinion and Comments Corpus.\n")
    print("As constructiveness is highly subjective and the annotations were done by crowdworkers, you have the option to:")
    print("  1. Use the old annotations (done by crowdworkers).\n")
    print("  2. Use the new annotations (generated using the following set of rules: at least 300 characters, 10 verbs and 8 adjectives).\n")
    print("  3. Use the new annotations (generated using the following set of rules: at least 300 characters, 10 verbs, 8 adjectives, and the ratio of constructive/non-constructive keywords.) ")

    # Load the different input for our models
    original_texts = extract_original_text(INPUT_PATH)
    print("\nOriginal comments loaded")
    df_features = load_conllu_data(INPUT_PATH)
    print("Comments features loaded\n")

    # User selection for annotations
    print("Choose your annotation option:")
    print("1. Use old annotations (crowdworkers)")
    print("2. Use new annotations (rule based - features only)")
    print("3. Use new annotations (rule based - features and keywords)")

    user_choice = input("Enter 1, 2, or 3: ").strip()

    if user_choice == '1':
        # Load annotations based on user choice
        annotations = pd.read_table(OLD_ANNOTATIONS_PATH, header=None)
        print("\nAnnotations loaded")

        # Load the different models 
        original_NB = joblib.load(OLD_NAIVE_BAYES_MODEL_PATH)
        RF = joblib.load(OLD_RANDOM_FOREST_MODEL_PATH)
        print("Models to predict crowdworkers annotations loaded")

        # Test the models with old annotations
        test_models(annotations, original_texts, df_features, original_NB, RF, user_choice)

    elif user_choice == '2':
        # Load annotations based on user choice
        annotations = pd.read_table(NEW_ANNOTATIONS_PATH_ONLY_FEATURES, header=None)
        print("Annotations loaded")

        # Load the different models
        new_NB = joblib.load(NEW_BAYESIAN_MODEL_ONLY_FEATURES_PATH)
        new_RF = joblib.load(NEW_RANDOM_FOREST_MODEL_ONLY_FEATURES_PATH)
        print("Models to predict rule-based annotations loaded (just features)")

        # Test the models with new annotations
        test_models(annotations, original_texts, df_features, new_NB, new_RF, user_choice)

    elif user_choice == '3':
        # Load annotations based on user choice
        annotations = pd.read_table(NEW_ANNOTATIONS_PATH, header=None)
        print("Annotations loaded")

        # Load the different models
        new_NB = joblib.load(NEW_BAYESIAN_MODEL_PATH)
        new_RF = joblib.load(NEW_RANDOM_FOREST_MODEL_PATH)
        print("Models to predict rule-based annotations loaded (keywords and features)")

        # Test the models with new annotations
        test_models(annotations, original_texts, df_features, new_NB, new_RF, user_choice)
    else:
        print("Invalid choice. Please restart and choose 1, 2 or 3.")
        return


def test_models(annotations, original_texts, df_features, NB, RF, user_choice):
    # Preparation of data
    y = annotations

    # Vectorization
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(original_texts)

    # Splitting into training and testing data
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Evaluate the model
    y_pred = NB.predict(X_test)
    print("\nBest Naive Bayesian Classification Report:\n", classification_report(y_test, y_pred))        

    ## Feature-based model
    X = df_features.drop(['comment_id'], axis=1)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Evaluate the model
    y_pred = RF.predict(X_test)
    print("\nBest Feature Based Model (Random Forest) Classification Report:\n", classification_report(y_test, y_pred))

    if user_choice == '1':
        #Print the Shap values of the feature based model
        # Create a SHAP Tree Explainer
        print("The feature based classifier has really good result, it would be really intersting to look into how it takes decision, for this, SHAP values are a really intersting tool")
        print("=" * 40)
        print("\n1. The number of tokens, nouns, and verbs are the features that help the most in our Random Forest model.")
        print("2. Features with large values (e.g., many nouns or verbs) increase the likelihood of predicting class 1: Constructive comments.")
        print("   - Exception: More sentences in a comment increase the likelihood of being predicted as Not-Constructive, but the SHAP value for this is still very low.")
        print("3. Conjunctions, pronouns, and adverbs don't seem to play a significant role in our model's decision-making process.")
        print("\n" + "=" * 40)
        explainer = shap.TreeExplainer(RF)
        shap_values = explainer.shap_values(X_test)
        shap.plots.violin(shap_values[:, :, 1], X_test, plot_type="layered_violin")
        print("Following the SHAP Values Analysis, we decided to create our own definiton of constructive comments, with the defined set of rules for our new annotations")
    else: 
        print("Rule based annotations improve the overall performance of the bayesian classifier, however the performance to predict the class 1 (constructive) is quite bad.")
        print("It also important to point out that it is not necessary to build feature based model, as the rules are already clearly defined.")
        print("We can aswell see that with the selected rules, we have more comments falling under the non-constructive class.")
        print("It doesn't seem to surprising as we could expect online comments to be mostly short and less detailed than a real conversation or debate for example")


if __name__ == "__main__":
    main()
