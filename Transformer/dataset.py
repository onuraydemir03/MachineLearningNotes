import torch
from torch import nn
from torch.utils.data import Dataset


class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_length):
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_length = seq_length

        self.sos_token = torch.Tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.Tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.Tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, item):
        src_target_pair = self.ds[item]
        src_text = src_target_pair["translation"][self.src_lang]
        tgt_text = src_target_pair["translation"][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_pad_tokens = self.seq_length - len(enc_input_tokens) - 2  # EOS - SOS
        dec_num_pad_tokens = self.seq_length - len(dec_input_tokens) - 1  # Only EOS

        if enc_num_pad_tokens < 0 or dec_num_pad_tokens < 0:
            raise ValueError("Sentence is too long")

        # Add SOS & EOS to the source text
        encoder_input = torch.cat([
            self.sos_token, torch.tensor(enc_input_tokens, dtype=torch.int64), self.eos_token,
            torch.tensor([self.pad_token] * enc_num_pad_tokens, dtype=torch.int64)
        ])

        # Add SOS to the decoder input
        decode_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token] * dec_num_pad_tokens, dtype=torch.int64)
        ])

        # Add EOS to the label (what we expect as output from decoder)
        label = torch.cat([
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * dec_num_pad_tokens, dtype=torch.int64)
        ])

        assert encoder_input.size(0) == self.seq_length
        assert decode_input.size(0) == self.seq_length
        assert label.size(0) == self.seq_length

        return {
            "encoder_input": encoder_input, # seq_length
            "decoder_input": decode_input, # seq_length
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),  # (1, 1, seq_length)
            "decoder_mask": (decode_input != self.pad_token).unsqueeze(0).unsqueeze(1).int() & causal_mask(decode_input.size(0)),  # (1, seq_length) & (1, seq_length, seq_length)
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text
        }

def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0


