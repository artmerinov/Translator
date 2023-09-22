import torch
from torch.utils.data import Dataset

from datasets import load_dataset
from tokenizer import create_tokenizer
from torch.utils.data import DataLoader, random_split


class BilingualDataset(Dataset):
    """
    Prepare dataset using SOS, EOS, PAD tokens 
    in a correct format for transformer.
    """
    def __init__(self, dataset, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, max_len):
        super().__init__()
        self.max_len = max_len
        self.dataset = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        # ids of special tokens
        self.pad_token_id = torch.tensor([tokenizer_src.token_to_id("[PAD]")], dtype=torch.int64)
        self.sos_token_id = torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token_id = torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # get pair by index
        pair = self.dataset[idx]
        src_text = pair["translation"][self.src_lang]
        tgt_text = pair["translation"][self.tgt_lang]

        # transform the text into tokens ids
        src_tokens_ids = self.tokenizer_src.encode(src_text).ids
        tgt_tokens_ids = self.tokenizer_tgt.encode(tgt_text).ids

        # count number of padding tokens
        src_num_padding_tokens = self.max_len - len(src_tokens_ids) - 2
        tgt_num_padding_tokens = self.max_len - len(tgt_tokens_ids) - 1

        # number of padding tokens must not be negative
        if src_num_padding_tokens < 0 or tgt_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # <SOS> <sequence> <EOS> <PAD>
        # (self.max_len)
        encoder_input = torch.cat([
            self.sos_token_id,
            torch.tensor(src_tokens_ids, dtype=torch.int64),
            self.eos_token_id,
            torch.tensor([self.pad_token_id] * src_num_padding_tokens, dtype=torch.int64)
        ], dim=0)

        # <SOS> <sequence> <PAD>
        # (self.max_len)
        decoder_input = torch.cat([
            self.sos_token_id,
            torch.tensor(tgt_tokens_ids, dtype=torch.int64),
            torch.tensor([self.pad_token_id] * tgt_num_padding_tokens, dtype=torch.int64)
        ], dim=0)

        # <sequence> <EOS> <PAD>
        # (self.max_len)
        label = torch.cat([
            torch.tensor(tgt_tokens_ids, dtype=torch.int64),
            self.eos_token_id,
            torch.tensor([self.pad_token_id] * tgt_num_padding_tokens, dtype=torch.int64)
        ], dim=0)

        # the size of the tensors must be equal to max_len
        assert encoder_input.size(0) == self.max_len
        assert decoder_input.size(0) == self.max_len
        assert label.size(0) == self.max_len

        # create encoder masks:
        # padding tokens should not participate in self-attention
        # unsqueeze to add sequence and batch dimentions
        encoder_mask = (encoder_input != self.pad_token_id).int().unsqueeze(0).unsqueeze(0)
        # create decoder mask:
        # padding tokens should not participate in self-attention
        # unsqueeze to add sequence and batch dimentions
        # also each token can only look at previous non-padding tokens
        decoder_mask = (decoder_input != self.pad_token_id).int().unsqueeze(0).unsqueeze(0)
        decoder_mask = decoder_mask & causal_mask(decoder_input.size(0))

        return {
            "encoder_input": encoder_input,  # (max_len)
            "decoder_input": decoder_input,  # (max_len)
            "encoder_mask": encoder_mask,  # (1, 1, max_len)
            "decoder_mask": decoder_mask,  # (1, max_len, max_len)
            "label": label,  # (max_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0


def get_dataset(config):
    """
    Prepare train and validation dataloader.
    """
    # load raw data
    # note, name of dataset is "it-ru" and not "ru-it"
    data_raw = load_dataset(
        path=config.DATASOURCE, 
        name=f"{config.LANG_TGT}-{config.LANG_SRC}",
        split="train"
    )

    # split on train and validation
    tr_size = int(config.TR_FRAC * len(data_raw))
    va_size = len(data_raw) - tr_size
    tr_data_raw, va_data_raw = random_split(data_raw, [tr_size, va_size])

    # create or load tokenizers
    tokenizer_src = create_tokenizer(dataset=data_raw, config=config, lang=config.LANG_SRC)
    tokenizer_tgt = create_tokenizer(dataset=data_raw, config=config, lang=config.LANG_TGT)

    # find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0
    for item in data_raw:
        src_ids = tokenizer_src.encode(item["translation"][config.LANG_SRC]).ids
        tgt_ids = tokenizer_tgt.encode(item["translation"][config.LANG_TGT]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length of source ({config.LANG_SRC}) sentence: {max_len_src}")
    print(f"Max length of target ({config.LANG_TGT}) sentence: {max_len_tgt}")

    # prepare data for transformer
    tr_data = BilingualDataset(
        dataset=tr_data_raw,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
        src_lang=config.LANG_SRC,
        tgt_lang=config.LANG_TGT,
        max_len=config.MAX_LEN,
    )
    va_data = BilingualDataset(
        dataset=va_data_raw,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
        src_lang=config.LANG_SRC,
        tgt_lang=config.LANG_TGT,
        max_len=config.MAX_LEN,
    )

    tr_dataloader = DataLoader(dataset=tr_data, batch_size=config.BATCH_SIZE, shuffle=True)
    va_dataloader = DataLoader(dataset=va_data, batch_size=1, shuffle=True)

    return tr_dataloader, va_dataloader