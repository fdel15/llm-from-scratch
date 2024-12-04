import re
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


class SimpleTokenizerV1:

    # This regex matches:
    # 1. Any single character that is one of: , . ? _ ! " ( ) '
    # 2. OR two consecutive hyphens (--)
    # 3. OR any single whitespace character (\s)
    #
    # The r prefix makes it a raw string, treating backslashes literally.
    # The parentheses ( ) create a capturing group for the entire pattern.
    # The square brackets [ ] define a character set to match any single character within.
    # The pipe | acts as an OR operator between different options.
    ENCODING_REGEX = r'([,.:;?_!"()\']|--|\s)'

    # Removes spaces before specified punctuation
    # This regex matches:
    #   1. One or more whitespace characters (\s+)
    #   2. Followed by a single character that is one of:
    #      comma, period, colon, semi-colon, question mark, exclamation mark,
    #      double quote, opening parenthesis, closing parenthesis,
    #      or single quote ([,.?!"()'])
    DECODING_REGEX = r'\s+([,.:;?!"()\'])'

    def __init__(self, vocab: dict[str, int]) -> None:
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> list[int]:
        """
        Takes string as input and transforms it into tokens using vocab dict
        """
        # split text into tokens
        preprocessed = re.split(self.ENCODING_REGEX, text)

        # remove white space characters
        preprocessed = [item.strip() for item in preprocessed if item.strip()]

        # map tokens to ids in vocab
        ids = [self.str_to_int[s] for s in preprocessed]

        return ids

    def decode(self, ids: list[int]) -> str:
        words = [self.int_to_str[id] for id in ids]
        words = " ".join(words)

        # Replace spaces before punctuations specified in regex
        words = re.sub(self.DECODING_REGEX, r"\1", words)
        return words


class SimpleTokenizerV2:
    # This regex matches:
    # 1. Any single character that is one of: , . ? _ ! " ( ) '
    # 2. OR two consecutive hyphens (--)
    # 3. OR any single whitespace character (\s)
    #
    # The r prefix makes it a raw string, treating backslashes literally.
    # The parentheses ( ) create a capturing group for the entire pattern.
    # The square brackets [ ] define a character set to match any single character within.
    # The pipe | acts as an OR operator between different options.
    ENCODING_REGEX = r'([,.:;?_!"()\']|--|\s)'

    # Removes spaces before specified punctuation
    # This regex matches:
    #   1. One or more whitespace characters (\s+)
    #   2. Followed by a single character that is one of:
    #      comma, period, colon, semi-colon, question mark, exclamation mark,
    #      double quote, opening parenthesis, closing parenthesis,
    #      or single quote ([,.?!"()'])
    DECODING_REGEX = r'\s+([,.:;?!"()\'])'

    UNKNOWN = "<|unk|>"

    def __init__(self, vocab: dict[str, int]) -> None:
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> list[int]:
        """
        Takes string as input and transforms it into tokens using vocab dict
        """
        # split text into tokens
        preprocessed = re.split(self.ENCODING_REGEX, text)

        # remove white space characters
        preprocessed = [item.strip() for item in preprocessed if item.strip()]

        # replace unknown words
        preprocessed = [
            item if item in self.str_to_int else self.UNKNOWN for item in preprocessed
        ]

        # map tokens to ids in vocab
        ids = [self.str_to_int[s] for s in preprocessed]

        return ids

    def decode(self, ids: list[int]) -> str:
        words = [self.int_to_str[id] for id in ids]
        words = " ".join(words)

        # Replace spaces before punctuations specified in regex
        words = re.sub(self.DECODING_REGEX, r"\1", words)
        return words


def get_raw_text():
    the_verdict_file_path = "../ch02/01_main-chapter-code/the-verdict.txt"
    with open(the_verdict_file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    return raw_text


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_data_loader_v1(
    txt: str,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,  # drops last batch if it is shorter than max_length to prevent loss spikes during training
    num_workers: int = 0,
):
    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader


raw_text = get_raw_text()
dataloader = create_data_loader_v1(
    raw_text, batch_size=8, max_length=4, stride=4, shuffle=False
)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("TOKEN_IDS: ", inputs)
print("Inputs Shape: ", inputs.shape)

vocab_size = 50257
output_dim = 256

embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

token_embeddings = embedding_layer(inputs)
print(token_embeddings.shape)

context_length = 4
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))

print(pos_embeddings.shape)

input_embeddings = token_embeddings + pos_embeddings

print(input_embeddings.shape)
