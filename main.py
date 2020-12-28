import torch  # If there's a GPU available...
import transformers
from transformers import BertForSequenceClassification, AdamW, BertConfig
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from functions.create_dataloader import create_data_loader
from functions.utils import load_dataset
from functions.create_trainer import create_trainer
from classes.sequence_trainer import SequenceTrainer
from classes.custom_trainer import NewsClassifier, CustomTrainer

from torch import nn, optim
from transformers import AutoTokenizer, AutoModel, BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib import rc
from collections import defaultdict
from pylab import rcParams
# from textwrap import wrap


def test_model(data_sample, label_number):
    model = BertModel.from_pretrained("bert-base-cased")

    input_ids = data_sample['input_ids'].to(device)
    attention_mask = data_sample['attention_mask'].to(device)
    print(input_ids.shape)  # seq length
    print(attention_mask.shape)  # seq length

    out = model(input_ids,
                attention_mask
                )
    print('last hidden state shape: ', out.last_hidden_state.shape)

    model = NewsClassifier(label_number)
    model = model.to(device)

    m = torch.nn.Softmax(dim=1)
    # print(m(model(encoded_text['input_ids'], encoded_text['attention_mask'])))
    output = m(model(input_ids, attention_mask))
    print(output)


if __name__ == '__main__':
    #############################################
    # some settings
    pd.set_option('mode.chained_assignment', None)
    sns.set(style='whitegrid', palette='muted', font_scale=1.2)
    HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
    sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
    rcParams['figure.figsize'] = 12, 8
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    #############################################

    device = torch.device("cpu")

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    sample_txt = 'Vatican allows gay marriage in council.'
    encoded_text = tokenizer.encode_plus(sample_txt,
                                         max_length=32,
                                         truncation=True,
                                         add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                                         return_token_type_ids=False,
                                         pad_to_max_length=True,
                                         return_attention_mask=True,
                                         return_tensors='pt',  # Return PyTorch tensors
                                         )

    print('encoded_text keys: ', encoded_text.keys())
    print('tokens of encoded text: ', tokenizer.convert_ids_to_tokens(encoded_text['input_ids'][0]))

    df_sample, max_token_length = load_dataset(tokenizer=tokenizer, random_seed=RANDOM_SEED)
    # df_sample, max_token_length = load_dataset(tokenizer=tokenizer, random_seed=RANDOM_SEED, path='sample_dataset.csv')

    print('df sample value counts: ')
    print(df_sample.category.value_counts())
    print('prepared dataset: ')
    print(df_sample.head())

    NUMBER_OF_LABELS = df_sample.category.nunique()
    MAX_LEN = min(512, max_token_length)

    # label encoded_text the categories. After this each category would be mapped to an integer.
    encoder = LabelEncoder()
    df_sample['category_encoded'] = encoder.fit_transform(df_sample['category'])
    print('encoded categories: ', df_sample.category_encoded.unique())
    print('decoded 0 ', encoder.inverse_transform([0]))
    print('decoded 1 ', encoder.inverse_transform([1]))
    print('decoded 2 ', encoder.inverse_transform([2]))
    print('decoded 3 ', encoder.inverse_transform([3]))
    print('decoded 4 ', encoder.inverse_transform([4]))
    print('encoded BUSINESS: ', encoder.transform(['BUSINESS']))
    print('encoded ENTERTAINMENT: ', encoder.transform(['ENTERTAINMENT']))
    print('encoded HEALTH: ', encoder.transform(['HEALTH']))
    print('encoded SCIENCE: ', encoder.transform(['SCIENCE']))
    print('encoded POLITICS: ', encoder.transform(['POLITICS']))

    df_train, df_test = train_test_split(df_sample, test_size=0.3, random_state=RANDOM_SEED)
    df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)

    print('train set shape: ', df_train.shape)
    print('val set shape: ', df_val.shape)
    print('test set shape: ', df_test.shape)

    BATCH_SIZE = 16

    # define data-loaders to sample batches for training and validation
    train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
    val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
    test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

    # to play around with data
    # data = train_data_loader.dataset.__getitem__(1)
    # data = next(iter(train_data_loader))
    # test_model(data)

    # Number of training epochs. The BERT authors recommend between 2 and 4.
    # We chose to run for 4, but we'll see later that this may be over-fitting the training data.
    EPOCHS = 1

    print('number of labels: ', NUMBER_OF_LABELS)
    print('max token length: ', MAX_LEN)
    print('batch size: ', BATCH_SIZE)
    print('epochs: ', EPOCHS)

    loss_fn = nn.CrossEntropyLoss().to(device)
    # trainer = create_trainer('custom',
    #                          EPOCHS,
    #                          train_data_loader,
    #                          val_data_loader,
    #                          device,
    #                          loss_fn,
    #                          len(df_train),
    #                          NUMBER_OF_LABELS)

    trainer = create_trainer('sequence',
                             EPOCHS,
                             train_data_loader,
                             val_data_loader,
                             device,
                             0,
                             0,
                             NUMBER_OF_LABELS)
    trainer.train()

    print('main close')
