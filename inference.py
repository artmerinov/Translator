import torch

from dataset import causal_mask


def translate_text(text, model, config, device, src_tokenizer, tgt_tokenizer):

    model.eval()
    with torch.no_grad():

        sos_tokens_ids = torch.tensor([src_tokenizer.token_to_id("[SOS]")], dtype=torch.int64)
        eos_tokens_ids = torch.tensor([src_tokenizer.token_to_id("[EOS]")], dtype=torch.int64)
        pad_tokens_ids = torch.tensor([src_tokenizer.token_to_id("[PAD]")], dtype=torch.int64)

        # transform sentence into token ids
        # <SOS> <sequence> <EOS> <PAD>
        src_tokens_ids = src_tokenizer.encode(text).ids
        src_tokens_ids = torch.cat([
            sos_tokens_ids,
            torch.tensor(src_tokens_ids, dtype=torch.int64),
            eos_tokens_ids,
            torch.tensor([pad_tokens_ids] * (config.MAX_LEN - len(src_tokens_ids) - 2), dtype=torch.int64),
        ], dim=0).to(device)

        # precompute the encoder output 
        # to reuse it for every generation step
        src_mask = (src_tokens_ids != pad_tokens_ids).int().unsqueeze(0).unsqueeze(0).to(device)
        encoder_output = model.encode(src_tokens_ids, src_mask)

        # initialize the decoder input with the <SOS> token
        decoder_input = sos_tokens_ids.unsqueeze(0).to(device)

        # Generate the translation word by word
        while decoder_input.size(1) < config.MAX_LEN:
            
            # build mask for target and calculate output
            decoder_mask = causal_mask(decoder_input.size(1)).type_as(src_mask).to(device)
            out = model.decode(
                tgt=decoder_input, 
                encoder_output=encoder_output, 
                src_mask=src_mask, 
                tgt_mask=decoder_mask
            )

            # add next token
            _, next_word = torch.max(out[:, -1], dim=1)
            decoder_input = torch.cat([decoder_input, next_word.view(1, 1).to(device)], dim=1)

            # break if we predict the end of sentence token
            if next_word == tgt_tokenizer.token_to_id('[EOS]'):
                break

    translation = tgt_tokenizer.decode(decoder_input.squeeze(0).detach().cpu().numpy())
    
    return translation