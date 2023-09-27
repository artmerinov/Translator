import os

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace


def create_tokenizer(dataset, config, lang: str):
    """
    Create simple word tokenizer.
    """
    # path to the tokenizer folder
    # make sure the tokenizer folder exists
    if not os.path.exists(config.PATH_TO_TOKENIZERS):
        os.mkdir(config.PATH_TO_TOKENIZERS)

    # path to the tokenizer file
    tokenizer_path = os.path.join(config.PATH_TO_TOKENIZERS, f'tokenizer_{lang}.json')
    if not os.path.exists(tokenizer_path):
        # if tokenizer file doesn't exist, we create tokenizer from scratch
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=5)
        tokenizer.train_from_iterator(get_all_sentences(dataset, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        # if tokenizer file exists, we load tokenizer file
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def craete_bpe_tokenzer():
    raise NotImplementedError("This function has not been implemented yet.")


def load_tokenizer(config, lang):
    """
    Load tokenizer from file.
    """
    tokenizer_path = os.path.join(config.PATH_TO_TOKENIZERS, f'tokenizer_{lang}.json')
    if os.path.exists(tokenizer_path):
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    else:
        raise FileNotFoundError(f"Tokenizer file {tokenizer_path} is not found.")
    return tokenizer


def get_all_sentences(dataset, lang):
    """
    Iterator of sentences.
    """
    for item in dataset:
        yield item["translation"][lang]