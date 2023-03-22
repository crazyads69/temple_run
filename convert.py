from os import listdir
from os.path import isfile, join
import polars as pl
"""
    Read all file in dataset folder and return a list of file names
"""
path = "dataset/parquet/"
dataset = [f for f in listdir(path) if isfile(join(path, f))]
print(dataset)

"""
    Read file content in dataset and convert from parquet to csv
"""
for x in dataset:
    print(path+x)
    df = pl.read_parquet(path+x)
    df.write_csv("dataset/csv/"+x[:-8]+".csv")
    print("Finish convert: "+"dataset/csv/"+x[:-8]+".csv")
