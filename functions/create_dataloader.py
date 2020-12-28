from classes.news_dataset import NewsDataset

from torch.utils.data import DataLoader


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = NewsDataset(texts=df.short_description.to_numpy(),
                     targets=df.category_encoded.to_numpy(),
                     tokenizer=tokenizer,
                     max_len=max_len
                     )

    return DataLoader(ds,
                      batch_size=batch_size,
                      num_workers=4  # original 4
                      )
