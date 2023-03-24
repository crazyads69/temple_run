import polars as pl
from os import listdir
from pyvi import ViTokenizer
from os.path import isfile, join
import string
import emoji
import csv
import re

"""
    Create a function that remove non-Vietnamese word in sentences of dataset
"""


def remove_non_vietnamese_word(text):
    """
        Split sentence to word (aka token)
    """
    tokens = []
    tokens = ViTokenizer.tokenize(text)
    # Define a list of Vietnamese characters
    vietnamese_chars = 'aăâbcdđeêghiklmnoôơpqrstuưvxy'

    # Remove non-Vietnamese words from the list of tokens
    vietnamese_tokens = [token for token in tokens if all(
        char in vietnamese_chars for char in token)]

    # Join the Vietnamese tokens back into a string
    vietnamese_text = ' '.join(vietnamese_tokens)

    return vietnamese_text


"""
    Read dataset from csv file and drop topic column
"""
path = "dataset/csv/"
translator = str.maketrans("", "", string.punctuation)
csv_list = [f for f in listdir(path) if isfile(join(path, f))]
file_list = []
for x in csv_list:
    file_list.append(path+x)
print(file_list)

"""
    Process CSV file to remove topic column and neutral feedback
"""


def clean_csv():
    for x in file_list:
        df = pl.read_csv(x)
        df = df.drop("topic")
        df_1 = df.filter(pl.col("sentiment") != 1)
        """
        Convert positive feedback with indicate value 2 to 1
        """
        df_2 = df_1.with_columns([pl.when(pl.col("sentiment") == 2).then(
            1).otherwise(pl.col("sentiment")).alias("sentiment")])
        df_2.write_csv(x)
        print("Finish clear csv file: "+x)


"""
    Read "sentence" column and convert to raw list then remove invalid character and punctuation
"""


def prepare_train_label():
    df_1 = pl.read_csv(path+"vsf_train.csv", columns=["sentiment"])
    train_label = df_1["sentiment"].to_list()
    print("Finish prepair train label")
    return train_label


def prepare_train_set():
    df = pl.read_csv(path+"vsf_train.csv", columns=["sentence"])
    raw_train_data = df["sentence"].to_list()
    train_data = []
    """
    Create a translate table to remove punctuation and invalid characteres (convert to empty string)
    """
    for x in raw_train_data:
        x = re.sub(r'\bcolon\w+\b', '', x)
        x = re.sub(r'\s+', ' ', x)
        x = x.lower()
        x = remove_non_vietnamese_word(x)
        train_data.append(x.translate(translator))
    print("Finish prepair train set")
    return train_data


"""
    Read "sentence" column and convert to raw list then remove invalid character and punctuation
"""


def prepare_val_label():
    df_1 = pl.read_csv(path+"vsf_val.csv", columns=["sentiment"])
    val_label = df_1["sentiment"].to_list()
    print("Finish prepair val label")
    return val_label


def prepare_val_set():
    df = pl.read_csv(path+"vsf_val.csv", columns=["sentence"])
    raw_val_data = df["sentence"].to_list()
    val_data = []
    """
        Create a translate table to remove punctuation and invalid characteres (convert to empty string)
    """
    for x in raw_val_data:
        x = re.sub(r'\bcolon\w+\b', '', x)
        x = re.sub(r'\s+', ' ', x)
        x = x.lower()
        x = remove_non_vietnamese_word(x)
        val_data.append(x.translate(translator))
    print("Finish prepair val set")
    return val_data


"""
    Read "sentence" column and convert to raw list then remove invalid character and punctuation
"""


def prepare_test_label():
    df_1 = pl.read_csv(path+"vsf_test.csv", columns=["sentiment"])
    test_label = df_1["sentiment"].to_list()
    print("Finish prepair test label")
    return test_label


def prepare_test_set():
    df = pl.read_csv(path+"vsf_test.csv", columns=["sentence"])
    raw_test_data = df["sentence"].to_list()
    test_data = []
    """
        Create a translate table to remove punctuation and invalid characteres (convert to empty string)
    """
    for x in raw_test_data:
        x = re.sub(r'\bcolon\w+\b', '', x)
        x = re.sub(r'\s+', ' ', x)
        x = x.lower()
        x = remove_non_vietnamese_word(x)
        test_data.append(x.translate(translator))
    print(test_data)
    print("Finish prepair test set")
    return test_data
