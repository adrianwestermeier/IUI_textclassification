import time
import torch

from functions.utils import format_time, flat_accuracy
from transformers import BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup


class SequenceTrainer:
    def __init__(self,
                 epochs,
                 train_data_loader,
                 validation_data_loader,
                 device,
                 number_of_labels):
        # Total number of training steps is [number of batches] x [number of epochs].
        # (Note that this is not the same as the number of training samples).
        total_steps = len(train_data_loader) * epochs
        # Load BertForSequenceClassification, pretrained BERT model with single linear classification layer on top
        self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                                   num_labels=number_of_labels,
                                                                   output_attentions=False,
                                                                   output_hidden_states=False
                                                                   )
        # Tell pytorch to run this model on the GPU.
        # self.model.cuda()
        # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
        self.optimizer = AdamW(self.model.parameters(),
                               lr=2e-5,  # args.learning_rate - default is 5e-5
                               eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                               )
        # Create the learning rate scheduler.
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=0,  # Default value in run_glue.py
                                                         num_training_steps=total_steps)
        self.epochs = epochs
        self.train_data_loader = train_data_loader
        self.validation_data_loader = validation_data_loader
        self.device = device

    def train(self):
        training_stats = []

        # Measure the total training time for the whole run.
        total_t0 = time.time()

        for epoch_i in range(0, self.epochs):

            # ========================================
            #               Training
            # ========================================

            # Perform one full pass over the training set.
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.epochs))
            print('Training...')

            # Measure how long the training epoch takes.
            t0 = time.time()

            # Reset the total loss for this epoch.
            total_train_loss = 0

            # Put the model into training mode.
            # (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
            self.model.train()

            # For each batch of training data...
            for step, batch in enumerate(self.train_data_loader):

                # Progress update every 40 batches.
                if step % 40 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = format_time(time.time() - t0)
                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'
                          .format(step, len(self.train_data_loader), elapsed))

                # Unpack this training batch from our dataloader.
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU using the
                # `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids
                #   [1]: attention masks
                #   [2]: labels
                b_input_ids = batch['input_ids'].to(self.device)
                b_input_mask = batch['attention_mask'].to(self.device)
                b_labels = batch['targets'].to(self.device)

                # Always clear any previously calculated gradients before performing a backward pass.
                # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
                self.model.zero_grad()

                # Perform a forward pass (evaluate the model on this training batch).
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # "logits": the model outputs prior to activation.
                out = self.model(b_input_ids,
                                 token_type_ids=None,
                                 attention_mask=b_input_mask,
                                 labels=b_labels)

                loss = out.loss
                # logits = out.logits
                print('loss: ', loss.item())

                # Accumulate the training loss over all of the batches so that we can
                # calculate the average loss at the end.
                total_train_loss += loss.item()

                # Perform a backward pass to calculate the gradients.
                loss.backward()

                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Update parameters and take a step using the computed gradient.
                self.optimizer.step()

                # Update the learning rate.
                self.scheduler.step()

            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(self.train_data_loader)

            # Measure how long this epoch took.
            training_time = format_time(time.time() - t0)

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(training_time))

            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.

            print("")
            print("Running Validation...")

            t0 = time.time()

            # Put the model in evaluation mode--the dropout layers behave differently
            # during evaluation.
            self.model.eval()

            # Tracking variables
            total_eval_accuracy = 0
            total_eval_loss = 0
            nb_eval_steps = 0

            # Evaluate data for one epoch
            for batch in self.validation_data_loader:
                # Unpack this training batch from our dataloader.
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU using
                # the `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids
                #   [1]: attention masks
                #   [2]: labels
                # b_input_ids = batch[0].to(self.device)
                # b_input_mask = batch[1].to(self.device)
                # b_labels = batch[2].to(self.device)
                b_input_ids = batch['input_ids'].to(self.device)
                b_input_mask = batch['attention_mask'].to(self.device)
                b_labels = batch['targets'].to(self.device)

                # Tell pytorch not to bother with constructing the compute graph during
                # the forward pass, since this is only needed for backprop (training).
                with torch.no_grad():
                    # Forward pass, calculate logit predictions.
                    # token_type_ids is the same as the "segment ids", which
                    # differentiates sentence 1 and 2 in 2-sentence tasks.
                    # The documentation for this `model` function is here:
                    # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                    # Get the "logits" output by the model. The "logits" are the output
                    # values prior to applying an activation function like the softmax.
                    out = self.model(b_input_ids,
                                     token_type_ids=None,
                                     attention_mask=b_input_mask,
                                     labels=b_labels)
                    loss = out.loss
                    logits = out.logits
                # Accumulate the validation loss.
                total_eval_loss += loss.item()

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                # Calculate the accuracy for this batch of test sentences, and
                # accumulate it over all batches.
                total_eval_accuracy += flat_accuracy(logits, label_ids)

            # Report the final accuracy for this validation run.
            avg_val_accuracy = total_eval_accuracy / len(self.validation_data_loader)
            print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

            # Calculate the average loss over all of the batches.
            avg_val_loss = total_eval_loss / len(self.validation_data_loader)

            # Measure how long the validation run took.
            validation_time = format_time(time.time() - t0)

            print("  Validation Loss: {0:.2f}".format(avg_val_loss))
            print("  Validation took: {:}".format(validation_time))

            # Record all statistics from this epoch.
            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Valid. Accur.': avg_val_accuracy,
                    'Training Time': training_time,
                    'Validation Time': validation_time
                }
            )

        print("")
        print("Training complete!")

        print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))
