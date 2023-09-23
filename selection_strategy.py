import torch 

from dataset import causal_mask


def greedy(model, source, source_mask, tokenizer_src, max_len, device):
    """
    Greedy strategy to choose next token.
    """
    sos_idx = tokenizer_src.token_to_id('[SOS]')
    eos_idx = tokenizer_src.token_to_id('[EOS]')

    # precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)

    # initialize the decoder input 
    # <SOS>
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        
        out = model.decode(
            tgt=decoder_input, 
            encoder_output=encoder_output, 
            src_mask=source_mask, 
            tgt_mask=decoder_mask
        )

        # get next token
        _, next_word = torch.max(out[:, -1], dim=1)
        decoder_input = torch.cat([
            decoder_input, 
            torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)
        ], dim=1)

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def beam_search():
    raise NotImplementedError("This function has not been implemented yet.")
