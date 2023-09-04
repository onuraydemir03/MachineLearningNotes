from pathlib import Path

import torch
from torch import nn
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import random_split, DataLoader

from Transformer.dataset import BilingualDataset
from Transformer.model import build_transformer


def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(config, ds, lang):
    # It is like: tokenizer_{en}.json, tokenizer_{it}.json
    tokenizer_path = Path(config['tokenizer_file'].format(lang))

    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=['[UNK]', '[PAD]', '[SOS]', '[EOS]'], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        trainer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def get_ds(config):
    ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split="train")

    # build tokenizer
    tokenizer_src = get_or_build_tokenizer(config=config, ds=ds_raw, lang=config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config=config, ds=ds_raw, lang=config['lang_tgt'])

    # Keep .9 trainin .1 validation

    train_ds_size = int(.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(ds=train_ds_raw, tokenizer_src=tokenizer_src,
                                tokenizer_tgt=tokenizer_tgt, src_lang=config['lang_src'],
                                tgt_lang=config['lang_tgt'], seq_length=config['seq_len'])

    val_ds = BilingualDataset(ds=val_ds_raw, tokenizer_src=tokenizer_src,
                              tokenizer_tgt=tokenizer_tgt, src_lang=config['lang_src'],
                              tgt_lang=config['lang_tgt'], seq_length=config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_src.encode(item['translation'][config['lang_tgt']]).ids

        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length of source: {max_len_src} | target: {max_len_tgt}")

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_length, vocab_tgt_length):
    model = build_transformer(src_vocab_size=vocab_src_length,
                              tgt_vocab_size=vocab_tgt_length,
                              src_seq_length=config['seq_len'],
                              tgt_seq_length=config['seq_len'])
    return model

