import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
import os


def load_dataset_json(name):
    directory = os.path.dirname(os.path.abspath(__file__))
    directory = directory + '/datasets/' + name
    df = pd.read_json(directory, lines=True)
    return df


def load_dataset_csv(name):
    directory = os.path.dirname(os.path.abspath(__file__))
    directory = directory + '/datasets/' + name
    df = pd.read_csv(directory)
    return df


def plot_token_distribution(token_lengths):
    sns.histplot(token_lengths)

    plt.xlim([0, 256])

    plt.xlabel('Token count')
    plt.show()


def plot_number_of_occurrences(df, column_name, df_name):
    sns.countplot(df[column_name])
    plt.xlabel(column_name)
    plt.xticks(rotation=70)
    # plt.show()
    plt.tight_layout()
    plt.savefig(df_name + '.png')
    plt.close()


def calculate_max_token_length(df, tokenizer):
    print('calculating max token length')
    token_lens = []

    for description in df.short_description:
        tokens = tokenizer.encode(description, max_length=512, truncation=True)

        token_lens.append(len(tokens))
    print('calculating max token length done')

    # plot_token_distribution(token_lens)
    return max(token_lens)


def investigate_dataset(df, column_name, df_name, tokenizer=None):
    print('df name: ', df_name)
    print('columns: ', df.columns.tolist())
    print('number of categories: ', df[column_name].nunique())
    print('categories: ', df[column_name].unique())
    print('df value counts: ')
    print(df[column_name].value_counts())
    # plot_number_of_occurrences(df, column_name, df_name)

    # to determine the proper max_len let uncomment the line below, it's 512 and 352
    # max_token_length = calculate_max_token_length(df, tokenizer)
    # print('max token length: ', max_token_length)
    return 512


def is_in_category_list(category, category_list):
    if category_list:
        if category in category_list:
            return category
        else:
            return 'OTHER'
    else:
        return category


# the datasets which are expected in the further process should have only columns category and short_description with
# the text which is categorized
def prepare_dataset(name, tokenizer, category_list=None):
    if name == 'News_Category_Dataset_v2.json':
        df = load_dataset_json(name)
        df.category = df.category.map(lambda x: "WORLDPOST" if x == "THE WORLDPOST" else x)
        df_truncated = df[['category', 'headline', 'short_description']]
        # max_token_length = investigate_dataset(df_truncated, 'category', 'news_category', tokenizer)
        df_truncated['headline'] = df_truncated['headline'].apply(lambda headline: str(headline).lower())
        df_truncated['short_description'] = df_truncated['short_description'].apply(lambda descr: str(descr).lower())
        df_truncated['short_description'] = df_truncated['headline'] + df_truncated['short_description']
        df_truncated['category'] = df_truncated['category'].apply(lambda category: is_in_category_list(category, category_list))
        # mappings
        df_truncated.category = df_truncated.category.map(lambda x: "SCIENCE" if x == "TECH" else x)
        df_truncated.category = df_truncated.category.map(lambda x: "HEALTH" if x == "WELLNESS" else x)
        if category_list:
            df_truncated = df_truncated[df_truncated['category'].isin(category_list)]
        df_truncated.rename(columns={"category": "label", "short_description": "text"}, inplace=True)
        df_truncated = df_truncated[['text', 'label']]
        max_token_length = investigate_dataset(df_truncated, 'label', 'news_category', tokenizer)
        return df_truncated[['text', 'label']], max_token_length
    else:
        df = load_dataset_csv(name)
        di = {'b': 'BUSINESS', 't': 'SCIENCE', 'e': 'ENTERTAINMENT', 'm': 'HEALTH'}
        df['CATEGORY'].replace(di, inplace=True)
        df.rename(columns={"CATEGORY": "label", "TITLE": "text"}, inplace=True)
        df = df[['text', 'label']]
        df['text'] = df['text'].apply(lambda text: str(text).lower())
        max_token_length = investigate_dataset(df, 'label', 'aggregator', tokenizer)
        return df[['text', 'label']], max_token_length


def load_dataset(tokenizer, number_samples, random_seed, path=None):
    if path:
        print("loading existing sample dataset")
        df_sample = load_dataset_csv(path)
        print('df sample value counts: ')
        print(df_sample.category.value_counts())
        return df_sample, 512
    else:
        # category_list = ['BUSINESS',
        #                  'SCIENCE',
        #                  'ENTERTAINMENT',
        #                  'HEALTH',
        #                  'POLITICS',
        #                  'SPORTS',
        #                  'RELIGION',
        #                  'WORLD NEWS',
        #                  'TECH',  # gets mapped to science
        #                  'WELLNESS',  # gets mapped to health
        #                  'OTHER']
        category_list = ['BUSINESS',
                         'SCIENCE',
                         'ENTERTAINMENT',
                         'HEALTH',
                         'POLITICS',
                         'TECH',  # gets mapped to science
                         'WELLNESS',  # gets mapped to health
                         'OTHER']

        df_news_category, max_token_length_category = prepare_dataset(name='News_Category_Dataset_v2.json',
                                                                      tokenizer=tokenizer,
                                                                      category_list=category_list)
        #  'WELLNESS', 'TRAVEL', 'STYLE & BEAUTY'
        df_news_aggregator, max_token_length_aggregator = prepare_dataset(name='uci-news-aggregator.csv',
                                                                          tokenizer=tokenizer)

        print('df category value counts: ')
        print(df_news_category.label.value_counts())
        print('df aggregator value counts: ')
        print(df_news_aggregator.label.value_counts())

        # append dicts and sample uniformly
        df_news = df_news_aggregator.append(df_news_category, ignore_index=True)
        investigate_dataset(df_news, 'label', 'all_categories')

        print('df value counts: ')
        print(df_news.label.value_counts())

        df_sample = df_news.groupby("label").sample(n=number_samples, random_state=random_seed)

        # directory = os.path.dirname(os.path.abspath(__file__))
        # directory = directory + '/datasets/sample_dataset_subset_categories.csv'
        # df_sample.to_csv(directory, index=False)

        max_token_length = max(max_token_length_aggregator, max_token_length_category)
        return df_sample, max_token_length


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
