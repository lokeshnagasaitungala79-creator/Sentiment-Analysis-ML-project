import os
import pickle
import sys

from envs.Machinelearning.Lib import re
from src.exception import CustomException
from src.logger import logger


def clean_text(text):
    try:
        text = str(text).lower()
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    except Exception as e:
        raise CustomException(e, sys)
    
def save_object(file_path , obj):
    try:
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory , exist_ok = True)
            with open(file_path , "wb") as file_obj:
                pickle.dump(obj , file_obj)
    except Exception as e:
        raise CustomException(e , sys)
    

def load_object(file_path):
    try:
        with open(file_path , "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e , sys)
        
