from torch import nn
from transformers import BertModel
from transformers import BertModel, AdamW, get_linear_schedule_with_warmup

import torch
import numpy as np


class NewsClassifier(nn.Module):

    def __init__(self, n_classes):
        super(NewsClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids,
                        attention_mask=attention_mask
                        )

        output = self.drop(out.pooler_output)

        return self.out(output)


class CustomTrainer:
    def __init__(self,
                 epochs,
                 train_data_loader,
                 validation_data_loader,
                 device,
                 loss_fn,
                 n_examples,  # len(df_train)
                 n_labels
                 ):
        # Total number of training steps is [number of batches] x [number of epochs].
        # (Note that this is not the same as the number of training samples).
        total_steps = len(train_data_loader) * epochs
        self.model = NewsClassifier(n_labels)
        self.optimizer = AdamW(self.model.parameters(), lr=2e-5, correct_bias=False)
        # Create the learning rate scheduler.
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=0,  # Default value in run_glue.py
                                                         num_training_steps=total_steps)
        self.epochs = epochs
        self.train_data_loader = train_data_loader
        self.validation_data_loader = validation_data_loader
        self.device = device
        self.loss_fn = loss_fn
        self.n_examples = n_examples

    def train_epoch(self):

        model = self.model.train()
        losses = []
        correct_predictions = 0

        for d in self.train_data_loader:
            input_ids = d["input_ids"].to(self.device)
            attention_mask = d["attention_mask"].to(self.device)
            targets = d["targets"].to(self.device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask
                            )

            # m = torch.nn.Softmax(dim=1)
            # outputs = m(outputs_model)

            _, preds = torch.max(outputs, dim=1)

            print('preds:   ', preds)
            print('targets: ', targets)

            loss = self.loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            print(correct_predictions)

            losses.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            print('losses:', losses)

        return float(correct_predictions) / self.n_examples, np.mean(losses)

    def eval_model(self):

        model = self.model.eval()
        losses = []
        correct_predictions = 0

        with torch.no_grad():
            for d in self.validation_data_loader:
                input_ids = d["input_ids"].to(self.device)
                attention_mask = d["attention_mask"].to(self.device)
                targets = d["targets"].to(self.device)

                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask
                                )

                _, preds = torch.max(outputs, dim=1)

                loss = self.loss_fn(outputs, targets)
                correct_predictions += torch.sum(preds == targets)
                losses.append(loss.item())

        return float(correct_predictions) / self.n_examples, np.mean(losses)

    def train(self):
        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            print('-' * 10)
            train_acc, train_loss = self.train_epoch()
            print(f'Train loss {train_loss} accuracy {train_acc}')

        print('training done')
