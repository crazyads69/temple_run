import polars as pl
from os import listdir
from os.path import isfile, join
import string
import re
import demoji

demoji.download_codes()
stop_words = ['là', 'của', 'làm', 'và', 'có', 'trong', 'được', 'ở', 'như', 'cho', 'này', 'để', 'không', 'được', 'với', 'cũng', 'vì', 'lên', 'nhiều', 'nhưng', 'còn', 'nữa', 'hay', 'đang', 'thì', 'đã', 'sẽ', 'vẫn', 'lại', 'hoặc', 'vậy', 'tại', 'khi', 'nào', 'cùng', 'đến', 'đều', 'thường', 'nên', 'mà', 'ở', 'ra', 'trên', 'theo', 'thấy', 'từ', 'nên', 'phải', 'đấy', 'thế', 'quá', 'thì', 'đó', 'mới', 'chỉ', 'được', 'chưa', 'đầu', 'chỉ', 'thôi', 'hơn', 'thế', 'những', 'nhất', 'đặc', 'biệt', 'thì', 'đúng', 'qua',
              'rất', 'nên', 'thêm', 'vào', 'khi', 'các', 'mỗi', 'về', 'sau', 'sẽ', 'nếu', 'bị', 'là', 'giữa', 'cả', 'rồi', 'trước', 'muốn', 'cảm', 'ơn', 'thôi', 'nào', 'tới', 'từng', 'rồi', 'ngay', 'người', 'người', 'ta', 'trên', 'trên', 'dưới', 'dưới', 'đến', 'tất', 'cả', 'thực', 'sự', 'tương', 'tự', 'điều', 'gì', 'còn', 'gì', 'lúc', 'nào', 'khác', 'nhau', 'thấp', 'cao', 'trong', 'ngoài', 'nhằm', 'mỗi', 'tùy', 'từng', 'mọi', 'cách', 'từ', 'ngày', 'ngày', 'một', 'tháng', 'tháng', 'năm', 'năm', 'trong', 'ngoài', 'doubledot', 'dot']


def remove_stopwords(sentence, stop_words):
    words = sentence.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    text = ' '.join(filtered_words)
    return text


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
        df_2 = df.drop("topic")
        df_3 = df_2.filter(pl.col("sentiment") != 1)
        """
        Convert positive feedback with indicate value 2 to 1
        """
        df_4 = df_3.with_columns([pl.when(pl.col("sentiment") == 2).then(
            1).otherwise(pl.col("sentiment")).alias("sentiment")])
        df_4.write_csv(x)
        print("Finish clear csv file: "+x)


"""
    Read "sentence" column and convert to raw list then remove invalid character and punctuation
"""


def prepare_train_label():
    df_5 = pl.read_csv(path+"vsf_train.csv", columns=["sentiment"])
    train_label = df_5["sentiment"].to_list()
    print("Finish prepair train label")
    return train_label


def prepare_train_set():
    df_6 = pl.read_csv(path+"vsf_train.csv", columns=["sentence"])
    raw_train_data = df_6["sentence"].to_list()
    train_data = []
    """
    Create a translate table to remove punctuation and invalid characteres (convert to empty string)
    """
    for x in raw_train_data:
        x = re.sub(r'\bcolon\w+\b', '', x)
        x = re.sub(r'\s+', ' ', x)
        x = x.lower()
        x = re.sub(r'\bwzjwz\w+\b', '', x)
        x = remove_stopwords(x, stop_words)
        x = demoji.replace(x, '')
        train_data.append(x.translate(translator))
    print("Finish prepair train set")
    return train_data


"""
    Read "sentence" column and convert to raw list then remove invalid character and punctuation
"""


def prepare_val_label():
    df_7 = pl.read_csv(path+"vsf_val.csv", columns=["sentiment"])
    val_label = df_7["sentiment"].to_list()
    print("Finish prepair val label")
    return val_label


def prepare_val_set():
    df_8 = pl.read_csv(path+"vsf_val.csv", columns=["sentence"])
    raw_val_data = df_8["sentence"].to_list()
    val_data = []
    """
        Create a translate table to remove punctuation and invalid characteres (convert to empty string)
    """
    for x in raw_val_data:
        x = re.sub(r'\bcolon\w+\b', '', x)
        x = re.sub(r'\s+', ' ', x)
        x = x.lower()
        x = re.sub(r'\bwzjwz\w+\b', '', x)
        x = remove_stopwords(x, stop_words)
        x = demoji.replace(x, '')
        val_data.append(x.translate(translator))
    print("Finish prepair val set")
    return val_data


"""
    Read "sentence" column and convert to raw list then remove invalid character and punctuation
"""


def prepare_test_label():
    df_9 = pl.read_csv(path+"vsf_test.csv", columns=["sentiment"])
    test_label = df_9["sentiment"].to_list()
    print("Finish prepair test label")
    return test_label


def prepare_test_set():
    df_10 = pl.read_csv(path+"vsf_test.csv", columns=["sentence"])
    raw_test_data = df_10["sentence"].to_list()
    test_data = []
    """
        Create a translate table to remove punctuation and invalid characteres (convert to empty string)
    """
    for x in raw_test_data:
        x = re.sub(r'\bcolon\w+\b', '', x)
        x = re.sub(r'\s+', ' ', x)
        x = x.lower()
        x = re.sub(r'\bwzjwz\w+\b', '', x)
        x = remove_stopwords(x, stop_words)
        x = demoji.replace(x, '')
        test_data.append(x.translate(translator))
    print("Finish prepair test set")
    return test_data
