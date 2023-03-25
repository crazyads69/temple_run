from sklearn.metrics import accuracy_score
import torch.nn as nn
from preprocess import *
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from transformers import AutoTokenizer
import torch.nn.functional as F
"""
    Prepare train, validation, test dataset
"""
clean_csv()
train_data = prepare_train_set()
train_label = prepare_train_label()
val_data = prepare_val_set()
val_label = prepare_val_label()
test_data = prepare_test_set()
test_label = prepare_test_label()
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-large")


class SentenceDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.float),
        }


train_dataset = SentenceDataset(train_data, train_label, tokenizer)
val_dataset = SentenceDataset(val_data, val_label, tokenizer)
test_dataset = SentenceDataset(test_data, test_label, tokenizer)

train_dataloader = DataLoader(
    train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(
    val_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(
    test_dataset, batch_size=32, shuffle=False)


class BiLSTMModel(pl.LightningModule):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout_prob):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim,
                              num_layers=num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)

        embedded = self.dropout(embedded)
        outputs, _ = self.bilstm(embedded)
        outputs = self.dropout(outputs)
        attention_weights = F.softmax(self.attention(outputs), dim=1)
        weighted_outputs = torch.sum(outputs * attention_weights, dim=1)
        dense_outputs = F.relu(self.fc1(weighted_outputs))
        logits = self.fc2(dense_outputs)
        return logits.squeeze()

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        logits = self(input_ids, attention_mask)
        loss = self.loss_fn(logits, labels)
        self.log('train_loss', loss, prog_bar=True,
                 on_step=True, on_epoch=True)
        self.training_step_outputs.append(
            {'loss': loss, 'label': labels, 'logits': logits})
        return loss

    def on_train_epoch_end(self):
        train_loss = torch.stack([x['loss']
                                 for x in self.training_step_outputs]).mean()
        self.log('train_loss_epoch', train_loss, prog_bar=True, on_epoch=True)
        y_true = []
        y_pred = []
        for output in self.training_step_outputs:
            if 'label' in output:
                y_true.append(output['label'].item())
            y_pred.append(torch.sigmoid(output['logits']).round().item())
        acc = accuracy_score(y_true, y_pred)
        self.log('train_acc', acc, prog_bar=True, on_epoch=True)
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        logits = self(input_ids, attention_mask)
        loss = self.loss_fn(
            logits, labels)
        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.validation_step_outputs.append(
            {'loss': loss, 'label': labels, 'logits': logits})
        return loss

    def on_validation_epoch_end(self):
        val_loss = torch.stack([x['loss']
                                for x in self.validation_step_outputs]).mean()
        self.log('val_loss_epoch', val_loss, prog_bar=True, on_epoch=True)
        y_true = []
        y_pred = []
        for output in self.validation_step_outputs:
            if 'label' in output:
                y_true.append(output['label'].item())
            y_pred.append(torch.sigmoid(output['logits']).round().item())
        acc = accuracy_score(y_true, y_pred)
        self.log('val_acc', acc, prog_bar=True, on_epoch=True)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        logits = self(input_ids, attention_mask)
        loss = self.loss_fn(
            logits, labels)
        self.log('test_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.test_step_outputs.append(
            {'loss': loss, 'label': labels, 'logits': logits})
        return loss

    def on_test_epoch_end(self):
        test_loss = torch.stack([x['loss']
                                for x in self.test_step_outputs]).mean()
        self.log('test_loss_epoch', test_loss, prog_bar=True, on_epoch=True)
        y_true = []
        y_pred = []
        for output in self.test_step_outputs:
            if 'label' in output:
                y_true.append(output['label'].item())
            y_pred.append(torch.sigmoid(output['logits']).round().item())
        acc = accuracy_score(y_true, y_pred)
        self.log('test_acc', acc, prog_bar=True, on_epoch=True)
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=1)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def predict(self, sentence):
        # Tokenize the input sentence
        sentence = clean_text(sentence)
        input_ids = tokenizer.encode(
            sentence, truncation=True, padding=True, return_tensors='pt')

        # Pass the input sequence through the model to get the predicted logits
        logits = self(input_ids)

        # Apply a sigmoid function to the logits to get the predicted probabilities
        probs = torch.sigmoid(logits)

        # Round the probabilities to get the predicted labels
        labels = torch.round(probs)

        # Convert the predicted labels to a Python list or NumPy array
        labels = labels.tolist()

        # Return the predicted label (0 for negative, 1 for positive)
        return labels[0]


vocab_size = tokenizer.vocab_size
embedding_dim = 128
hidden_dim = 64
num_layers = 6
dropout_prob = 0.2
model = BiLSTMModel(vocab_size, embedding_dim,
                    hidden_dim, num_layers, dropout_prob)

trainer = pl.Trainer(max_epochs=20, reload_dataloaders_every_n_epochs=1,
                     enable_checkpointing=1, enable_progress_bar=1, detect_anomaly=True)
trainer.fit(model, train_dataset, val_dataset)
trainer.save_checkpoint('checkpoint.ckpt')
trainer.test(model, 'checkpoint.ckpt', test_dataset)


model.load_from_checkpoint('checkpoint.ckpt')
sentence = "Cô dạy chán, không tạo hứng thú cho sinh viên."
label = model.predict(sentence)
print(label)  # Output: 1
