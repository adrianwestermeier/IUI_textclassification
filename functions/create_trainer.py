from classes.sequence_trainer import SequenceTrainer
from classes.custom_trainer import CustomTrainer


def create_trainer(type_of_trainer,
                   epochs,
                   train_data_loader,
                   validation_data_loader,
                   device,
                   loss_fn,
                   n_examples,
                   number_of_labels):
    if type_of_trainer == 'custom':
        custom_trainer = CustomTrainer(epochs=epochs,
                                       train_data_loader=train_data_loader,
                                       validation_data_loader=validation_data_loader,
                                       device=device,
                                       loss_fn=loss_fn,
                                       n_examples=n_examples,
                                       n_labels=number_of_labels)
        return custom_trainer
    else:
        sequence_trainer = SequenceTrainer(epochs=epochs,
                                           train_data_loader=train_data_loader,
                                           validation_data_loader=validation_data_loader,
                                           device=device,
                                           number_of_labels=number_of_labels)
        return sequence_trainer
