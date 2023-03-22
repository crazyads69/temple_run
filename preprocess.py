import polars as pl
from os import listdir
from os.path import isfile, join
import glob as gl
import csv
import string
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
df = pl.read_csv(path+"vsf_train.csv", columns=["sentence"])
df_1 = pl.read_csv(path+"vsf_train.csv", columns=["sentiment"])
raw_train_data = df["sentence"].to_list()
train_label = df_1["sentiment"].to_list()
train_data = []
"""
    Create a translate table to remove punctuation and invalid characteres (convert to empty string)
"""
for x in raw_train_data:
    x.lower()
    train_data.append(x.translate(translator))
print("Finish prepair train set")

"""
    Read "sentence" column and convert to raw list then remove invalid character and punctuation
"""
df = pl.read_csv(path+"vsf_val.csv", columns=["sentence"])
df_1 = pl.read_csv(path+"vsf_val.csv", columns=["sentiment"])
raw_val_data = df["sentence"].to_list()
val_label = df_1["sentiment"].to_list()
val_data = []
"""
    Create a translate table to remove punctuation and invalid characteres (convert to empty string)
"""
for x in raw_val_data:
    x.lower()
    val_data.append(x.translate(translator))
print("Finish prepair val set")


"""
    Read "sentence" column and convert to raw list then remove invalid character and punctuation
"""
df = pl.read_csv(path+"vsf_test.csv", columns=["sentence"])
df_1 = pl.read_csv(path+"vsf_test.csv", columns=["sentiment"])
raw_test_data = df["sentence"].to_list()
test_label = df_1["sentiment"].to_list()
test_data = []
"""
    Create a translate table to remove punctuation and invalid characteres (convert to empty string)
"""
for x in raw_test_data:
    x.lower()
    test_data.append(x.translate(translator))
print("Finish prepair test set")
