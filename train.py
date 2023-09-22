import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.text import WordErrorRate, CharErrorRate, BLEUScore

from dataset import get_dataset
from utils import set_random_seed, Config, get_weights_file_path
from model import build_transformer
from tokenizer import load_tokenizer
from selection_strategy import greedy


def train_model(config):
    """
    Training process.
    """
    # set up device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # make sure the weights folder exists
    weights_dir = config.MODEL_FOLDER_NAME
    if not os.path.exists(weights_dir):
        os.mkdir(weights_dir)

    # load tokenizers
    tokenizer_src = load_tokenizer(config=config, lang=config.LANG_SRC)
    tokenizer_tgt = load_tokenizer(config=config, lang=config.LANG_TGT)

    tr_dataloader, va_dataloader = get_dataset(config)

    model = build_transformer(
        src_vocab_size=tokenizer_src.get_vocab_size(),
        tgt_vocab_size=tokenizer_tgt.get_vocab_size(),
        src_seq_len=config.MAX_LEN,
        tgt_seq_len=config.MAX_LEN,
        embed_size=config.D_MODEL,
    ).to(device)

    writer = SummaryWriter(config.EXPERIMENT_FOLDER_NAME)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.LR, eps=1e-9)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1).to(device)

    batch_processed = 0
    for epoch in range(config.NUM_EPOCHS):

        # TRAIN
        # =====
        torch.cuda.empty_cache()
        model.train()
        
        batch_iterator = tqdm(tr_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for tr_batch in batch_iterator:

            batch_processed += 1

            encoder_input = tr_batch["encoder_input"].to(device)  # (batch_size, max_len)
            decoder_input = tr_batch["decoder_input"].to(device)  # (batch_size, max_len)
            encoder_mask = tr_batch["encoder_mask"].to(device)  # (batch_size, 1, 1, max_len)
            decoder_mask = tr_batch["decoder_mask"].to(device)  # (batch_size, 1, max_len, max_len)
            label = tr_batch['label'].to(device) # (batch_size, max_len)

            # run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(src=encoder_input, src_mask=encoder_mask)  # (batch_size, max_len, embed_size)
            decoder_output = model.decode(
                encoder_output=encoder_output, 
                src_mask=encoder_mask, 
                tgt=decoder_input, 
                tgt_mask=decoder_mask
            )  # (batch_size, max_len, embed_size)
            proj_output = model.project(x=decoder_output)  # (batch_size, max_len, vocab_size)

            # compute the loss using a simple cross entropy
            loss = loss_fn(
                input=proj_output.view(-1, tokenizer_tgt.get_vocab_size()), # (batch_size * max_len, vocab_size)
                target=label.view(-1) # (batch_size * max_len)
            )
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # log the loss
            writer.add_scalar("train loss", loss.item(), batch_processed)

            # backpropagate the loss
            loss.backward()

            # update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            
        # VALIDATION
        # ==========
        model.eval()

        # accumulated metrics
        acc_cer = 0
        acc_wer = 0
        acc_bleu = 0

        bleu = BLEUScore()
        wer = WordErrorRate()
        cer = CharErrorRate()

        acc = 0
        with torch.no_grad():
            for va_batch in va_dataloader:
                acc += 1

                encoder_input = va_batch["encoder_input"].to(device) # (b, seq_len)
                encoder_mask = va_batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

                # decide on the next token
                model_output = greedy(
                    model=model,
                    source=encoder_input,
                    source_mask=encoder_mask,
                    tokenizer_src=tokenizer_src,
                    max_len=config.MAX_LEN,
                    device=device
                )

                src_text = va_batch["src_text"][0]
                tgt_text = va_batch["tgt_text"][0]
                prd_text = tokenizer_tgt.decode(model_output.detach().cpu().numpy())

                acc_bleu += bleu([prd_text], [[tgt_text]])
                acc_cer += cer(prd_text, tgt_text)
                acc_wer += wer(prd_text, tgt_text)

                if acc < 20:
                    # print translation examples from validation set
                    print(f"{f'SRC: ':>12}{src_text}")
                    print(f"{f'TGT: ':>12}{tgt_text}")
                    print(f"{f'PRD: ':>12}{prd_text}")
                    print('-'*80)
                else:
                    # stop validation black to save time
                    break

        writer.add_scalar('validation CER', (acc_cer / acc).item(), epoch)
        writer.add_scalar('validation WER', (acc_wer / acc).item(), epoch)
        writer.add_scalar('validation BLEU', (acc_bleu / acc).item(), epoch)
        writer.flush()
        writer.close()

        print(f'CER: {(acc_cer / acc).item():.4f}')
        print(f'WER: {(acc_wer / acc).item():.4f}')
        print(f'BLEU: {(acc_bleu / acc).item():.4f}')

        # save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "batch_processed": batch_processed
            }, model_filename
        )


if __name__ == "__main__":
    config = Config('config.yaml')
    set_random_seed(seed=0)
    train_model(config)
