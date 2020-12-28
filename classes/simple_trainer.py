import logging
import pandas as pd

from simpletransformers.classification import ClassificationModel, ClassificationArgs


class SimpleTrainer:
    def __init__(self,
                 epochs,
                 device,
                 number_of_labels,
                 train_df,
                 eval_df):
        self.epochs = epochs
        self.device = device
        self.num_labels = number_of_labels
        self.train_df = train_df
        self.eval_df = eval_df

    def run_trainer(self):
        logging.basicConfig(level=logging.INFO)
        transformers_logger = logging.getLogger("transformers")
        transformers_logger.setLevel(logging.WARNING)

        # Preparing train data
        train_data = [
            ["Aragorn was the heir of Isildur", 1],
            ["Frodo was the heir of Isildur", 0],
        ]
        train_df = pd.DataFrame(train_data)
        train_df.columns = ["text", "labels"]

        # Preparing eval data
        eval_data = [
            ["Theoden was the king of Rohan", 1],
            ["Merry was the king of Rohan", 0],
        ]
        eval_df = pd.DataFrame(eval_data)
        eval_df.columns = ["text", "labels"]

        # # Optional model configuration
        # model_args = ClassificationArgs(num_train_epochs=1)
        #
        # # Create a ClassificationModel
        # model = ClassificationModel(
        #     "distilbert", "distilbert-base-uncased", args=model_args, use_cuda=(self.device == 'cuda')
        # )

        model_args = {"reprocess_input_data": True, "overwrite_output_dir": True}
        # Create a ClassificationModel
        model = ClassificationModel(
            "distilbert",
            "distilbert-base-uncased",
            num_labels=self.num_labels,
            args=model_args,
            use_cuda=(not (self.device.type == 'cpu'))
        )

        # Train the model
        model.train_model(train_df)

        # Evaluate the model
        result, model_outputs, wrong_predictions = model.eval_model(eval_df)

        # Make predictions with the model
        predictions, raw_outputs = model.predict(["Sam was a Wizard"])

        print("predictions: ", predictions)
        print("result: ", result)
