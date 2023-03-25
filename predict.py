from sklearn.metrics import accuracy_score
import torch.nn as nn
import torch
import pytorch_lightning as pl
from transformers import AutoTokenizer
import torch.nn.functional as F
from transformers import AutoTokenizer
import re
import string

translator = str.maketrans("", "", string.punctuation)


def clean_text(text):
    text = re.sub(r'\bcolon\w+\b', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    text = re.sub(r'\bwzjwz\w+\b', '', text)
    text = text.translate(translator)
    return text


tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-large")
vocab_size = tokenizer.vocab_size
embedding_dim = 128
hidden_dim = 64
num_layers = 6
dropout_prob = 0.2


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
        print(sentence)
        input_ids = tokenizer.encode(
            sentence, add_special_tokens=True, max_length=128, truncation=True, padding='max_length', return_tensors='pt')
        attention_mask = tokenizer.encode_plus(sentence,
                                               add_special_tokens=True,
                                               max_length=128,
                                               padding='max_length',
                                               truncation=True,
                                               return_attention_mask=True,
                                               return_tensors='pt')
        # Pass the input sequence through the model to get the predicted logits
        logits = self(input_ids, attention_mask)
        # Apply a sigmoid function to the logits to get the predicted probabilities
        probs = torch.sigmoid(logits)
        # Round the probabilities to get the predicted labels
        print(probs.squeeze().item())
        probs = (probs >= 0.5).long().squeeze()
        # Return the predicted label (0 for negative, 1 for positive)
        return probs.item()


"""
model = BiLSTMModel(vocab_size=vocab_size,
                    embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout_prob=dropout_prob)
"""
model = BiLSTMModel.load_from_checkpoint('checkpoint.ckpt', vocab_size=tokenizer.vocab_size,
                                         embedding_dim=128, hidden_dim=64, num_layers=4, dropout_prob=0.2)
sentence = "nhiệt tình giảng dạy , gần gũi với sinh viên ."
label = model.predict(sentence)
print(label)
