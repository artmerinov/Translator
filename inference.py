import torch

from dataset import causal_mask


def translate_text(text, model, config, device, tokenizer_src, tokenizer_tgt):

    model.eval()
    with torch.no_grad():

        sos_token_id = torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64)
        eos_token_id = torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64)
        pad_token_id = torch.tensor([tokenizer_src.token_to_id("[PAD]")], dtype=torch.int64)

        # transform sentence into token ids
        # <SOS> <sequence> <EOS> <PAD>
        src_tokens_ids = tokenizer_src.encode(text).ids
        src_tokens_ids = torch.cat([
            sos_token_id,
            torch.tensor(src_tokens_ids, dtype=torch.int64),
            eos_token_id,
            torch.tensor([pad_token_id] * (config.MAX_LEN - len(src_tokens_ids) - 2), dtype=torch.int64),
        ], dim=0).to(device)

        # precompute the encoder output and reuse it for every generation step
        src_mask = (src_tokens_ids != pad_token_id).int().unsqueeze(0).unsqueeze(0).to(device)
        encoder_output = model.encode(src_tokens_ids, src_mask)

        # initialize the decoder input with the sos token
        # <SOS>
        decoder_input = sos_token_id.unsqueeze(0).to(device)

        # Generate the translation word by word
        while decoder_input.size(1) < config.MAX_LEN:
            # build mask for target and calculate output
            decoder_mask = causal_mask(decoder_input.size(1)).type_as(src_mask).to(device)
            out = model.decode(encoder_output, src_mask, decoder_input, decoder_mask)

            # add next token
            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat([
                decoder_input, 
                torch.empty(1, 1).type_as(src_tokens_ids).fill_(next_word.item())
            ], dim=1).to(device)

            # break if we predict the end of sentence token
            if next_word == tokenizer_tgt.token_to_id('[EOS]'):
                break

    translation = tokenizer_tgt.decode(decoder_input.squeeze(0).detach().cpu().numpy())
    
    return translation