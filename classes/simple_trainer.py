import logging
import pandas as pd

from simpletransformers.classification import ClassificationModel, ClassificationArgs
from sklearn.metrics import confusion_matrix, accuracy_score


class SimpleTrainer:
    def __init__(self,
                 model_name,
                 epochs,
                 batch_size,
                 device,
                 number_of_labels,
                 max_seq_length,
                 train_df,
                 eval_df,
                 output_dir):
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.num_labels = number_of_labels
        self.max_seq_length = max_seq_length
        self.train_df = train_df
        self.eval_df = eval_df
        self.output_dir = output_dir

    def run_trainer(self):
        logging.basicConfig(level=logging.INFO)
        transformers_logger = logging.getLogger("transformers")
        transformers_logger.setLevel(logging.WARNING)

        # Preparing train data
        # train_data = [
        #     ["Aragorn was the heir of Isildur", 1],
        #     ["Frodo was the heir of Isildur", 0],
        # ]
        # train_df = pd.DataFrame(train_data)
        # train_df.columns = ["text", "labels"]
        #
        # # Preparing eval data
        # eval_data = [
        #     ["Theoden was the king of Rohan", 1],
        #     ["Merry was the king of Rohan", 0],
        # ]
        # eval_df = pd.DataFrame(eval_data)
        # eval_df.columns = ["text", "labels"]

        # # Optional model configuration
        # model_args = ClassificationArgs(num_train_epochs=1)
        #
        # # Create a ClassificationModel
        # model = ClassificationModel(
        #     "distilbert", "distilbert-base-uncased", args=model_args, use_cuda=(self.device == 'cuda')
        # )

        model_args = {'max_seq_length': self.max_seq_length,
                      'learning_rate': 4e-5,
                      'num_train_epochs': self.epochs,
                      'reprocess_input_data': True,
                      'overwrite_output_dir': True,
                      'evaluate_during_training': True,
                      'evaluate_during_training_steps': 200,
                      'logging_steps': 50,
                      'do_lower_case': True,
                      'train_batch_size': self.batch_size,
                      'use_batch_norm': False,
                      'tensorboard_dir': '{}/runs'.format(self.output_dir),
                      'patience': 5,
                      'save_only_best': True,
                      'overwrite_last_saved': True,
                      'save_steps': 0}
        # Create a ClassificationModel
        model = ClassificationModel(
            self.model_name,
            self.model_name + "-base-uncased",
            num_labels=self.num_labels,
            args=model_args,
            use_cuda=(not (self.device.type == 'cpu'))
        )

        # Train the model
        # model.train_model(self.train_df)
        # model.train_model(self.train_df, output_dir=output_dir, eval_df=test_x, acc=accuracy_score)
        model.train_model(self.train_df, eval_df=self.eval_df, acc=accuracy_score)

        # Evaluate the model
        # eval_df, multi_label=False, output_dir=None, verbose=True, silent=False, wandb_log=True, **kwargs
        result, model_outputs, wrong_predictions = model.eval_model(eval_df=self.eval_df,
                                                                    multi_label=False,
                                                                    output_dir=self.output_dir,
                                                                    verbose=True,
                                                                    silent=False,
                                                                    wandb_log=True)

        print("result: ", result)
        return model
