from preprocess import *

clean_csv()
train_data = prepare_train_set()
train_label = prepare_train_label()
val_data = prepare_val_set()
val_label = prepare_val_label()
test_data = prepare_test_set()
test_label = prepare_test_label()
