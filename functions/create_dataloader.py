from classes.news_dataset import NewsDataset

from torch.utils.data import DataLoader


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = NewsDataset(texts=df.text.to_numpy(),
                     targets=df.category_encoded.to_numpy(),
                     tokenizer=tokenizer,
                     max_len=max_len
                     )

    return DataLoader(ds,
                      batch_size=batch_size,
                      num_workers=1  # original 4
                      )
