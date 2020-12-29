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
from classes.simple_trainer import SimpleTrainer

from torch import nn, optim
from transformers import AutoTokenizer, AutoModel, BertModel, BertTokenizer, DistilBertTokenizer, AdamW, \
    get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib import rc
from collections import defaultdict
from pylab import rcParams
# from textwrap import wrap
from simpletransformers.classification import ClassificationModel, ClassificationArgs


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
    mode = 'simple_transformers'
    # mode = 'custom'
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

    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"

    device = torch.device(dev)
    print('device: ', device)

    if mode == 'custom':
        tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

        # sample_txt = 'Vatican allows gay marriage in council.'
        # encoded_text = tokenizer.encode_plus(sample_txt,
        #                                      max_length=32,
        #                                      truncation=True,
        #                                      add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        #                                      return_token_type_ids=False,
        #                                      pad_to_max_length=True,
        #                                      return_attention_mask=True,
        #                                      return_tensors='pt',  # Return PyTorch tensors
        #                                      )
        #
        # print('encoded_text keys: ', encoded_text.keys())
        # print('tokens of encoded text: ', tokenizer.convert_ids_to_tokens(encoded_text['input_ids'][0]))

        df_sample, max_token_length = load_dataset(tokenizer=tokenizer, random_seed=RANDOM_SEED)
        # df_sample, max_token_length = load_dataset(tokenizer=tokenizer, random_seed=RANDOM_SEED, path='sample_dataset.csv')

        print('df sample value counts: ')
        print(df_sample.label.value_counts())
        print('prepared dataset: ')
        print(df_sample.head())

        NUMBER_OF_LABELS = df_sample.label.nunique()
        MAX_LEN = min(512, max_token_length)
        BATCH_SIZE = 16
        # Number of training epochs. The BERT authors recommend between 2 and 4.
        # We chose to run for 4, but we'll see later that this may be over-fitting the training data.
        EPOCHS = 1

        print('number of labels: ', NUMBER_OF_LABELS)
        print('max token length: ', MAX_LEN)
        print('batch size: ', BATCH_SIZE)
        print('epochs: ', EPOCHS)

        # label encoded_text the categories. After this each category would be mapped to an integer.
        encoder = LabelEncoder()
        df_sample['category_encoded'] = encoder.fit_transform(df_sample['label'])
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

        # define data-loaders to sample batches for training and validation
        train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
        val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
        test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

        # to play around with data
        # data = train_data_loader.dataset.__getitem__(1)
        # data = next(iter(train_data_loader))
        # test_model(data)

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

    else:
        test_set = [
            ["A 7-year-old boy named Eli saved his baby sister's life when "
             "he jumped into her room through a window to rescue her from "
             "a fire that destroyed his family's home."],
            [" Muslim and Jewish paramedics pause to pray together.One of many inspiring moments "
             "in the coronavirus crisis "],
            ["Adidas developing plant-based leather material that will be used to make shoes...material made "
             "from mycelium, which is part of fungus. Company produced 15 million pairs of shoes "
             "in 2020 made from recycled plastic waste collected from beaches and coastal regions."],
            ["A new cancer therapy simultaneously zaps tumors with imaging-guided laser radiation and stimulates "
             "the anti-cancer immune response. This technology, developed by South Korean scientists, combines "
             "photodynamic therapy with immunotherapy for the treatment of cancer and is the first of its kind."]
        ]
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

        # load data
        df_sample, max_token_length = load_dataset(tokenizer=tokenizer, random_seed=RANDOM_SEED)
        # df_sample, max_token_length = load_dataset(tokenizer=tokenizer, random_seed=RANDOM_SEED, path='sample_dataset.csv')
        print('df sample value counts: ')
        print(df_sample.label.value_counts())
        print('prepared dataset: ')
        print(df_sample.head())

        # hyperparams
        NUMBER_OF_LABELS = df_sample.label.nunique()
        MAX_LEN = min(512, max_token_length)
        BATCH_SIZE = 16
        EPOCHS = 1

        # label encoded_text the categories. After this each category would be mapped to an integer.
        encoder = LabelEncoder()
        df_sample['category_encoded'] = encoder.fit_transform(df_sample['label'])
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

        # take encoded labels as labels
        df_sample.drop(['label'], axis=1, inplace=True)
        df_sample.rename(columns={"category_encoded": "labels"}, inplace=True)

        df_train, df_test = train_test_split(df_sample, test_size=0.3, random_state=RANDOM_SEED)
        df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)

        trainer = SimpleTrainer(epochs=EPOCHS,
                                device=device,
                                batch_size=BATCH_SIZE,
                                number_of_labels=NUMBER_OF_LABELS,
                                max_seq_length=MAX_LEN,
                                train_df=df_train,
                                eval_df=df_val)
        model = trainer.run_trainer()

        for el in test_set:
            # Make predictions with the model
            predictions, raw_outputs = model.predict(el)
            print("predictions: ", predictions)
            print("decoded: ", encoder.inverse_transform(predictions))

    print('main close')
