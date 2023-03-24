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
        preds = torch.round(torch.sigmoid(logits))
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        acc = (preds == labels).float().mean()
        self.log('train_loss', loss, prog_bar=True,
                 on_step=True, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True, on_step=True, on_epoch=True)
        return {'loss': loss, 'acc': acc}

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        logits = self(input_ids, attention_mask)
        preds = torch.round(torch.sigmoid(logits))
        acc = (preds == labels).float().mean()
        loss = F.binary_cross_entropy_with_logits(
            logits, labels)
        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_step=True, on_epoch=True)
        return {'loss': loss, 'acc': acc}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=1)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}


vocab_size = tokenizer.vocab_size
embedding_dim = 128
hidden_dim = 64
num_layers = 2
dropout_prob = 0.2
model = BiLSTMModel(vocab_size, embedding_dim,
                    hidden_dim, num_layers, dropout_prob)

trainer = pl.Trainer(max_epochs=20, reload_dataloaders_every_n_epochs=1,
                     enable_checkpointing=1, enable_progress_bar=1, detect_anomaly=True)
trainer.fit(model, train_dataset, val_dataset)
trainer.save_checkpoint('checkpoint.ckpt')
trainer.test(model, test_dataset)
