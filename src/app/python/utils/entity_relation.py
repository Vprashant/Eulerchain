from src.app.python.constant.project_constant import Constant as constant
from src.app.python.constant.global_data import GlobalData
from typing import Any, Optional, List
import pickle
import spacy

nlp = spacy.load("en_core_web_sm")

class EntityRelation(object):
    def __init__(self) -> None:
        pass
    def extract_relationships(self, text, keywords):
        relationships = {}
        doc = nlp(text)
        for token in doc:
            if token.text in keywords:
                relationships[token.text] = [child.text for child in token.children if child.text in keywords]
        print(relationships)
        return relationships

    def save_relationships_to_pickle(self, relationships, pickle_file):
        with open(pickle_file, 'wb') as f:
            pickle.dump(relationships, f)
        print(f"[INFO] {pickle_file} Pickle Saved.")

    def load_relationships_from_pickle(self, pickle_file):
        with open(pickle_file, 'rb') as f:
            relationships = pickle.load(f)
        return relationships

    def check_relationships(self, text, relationships):
        doc = nlp(text)
        for token in doc:
            if token.text in relationships:
                print(f"Relationships for {token.text}: {relationships[token.text]}")

text = "sabic has a connection with psa and linde"
keywords = ["sabic", "psa", "linde", "dow"]

er = EntityRelation()
relationships = er.extract_relationships(text, keywords)
er.save_relationships_to_pickle(relationships, "relationships.pickle")

loaded_relationships = er.load_relationships_from_pickle("relationships.pickle")

another_text = "sabic is delicious. I like red apples."
er.check_relationships(another_text, loaded_relationships)

