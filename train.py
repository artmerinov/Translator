import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.text import WordErrorRate, CharErrorRate, BLEUScore

from dataset import get_dataset
from utils import set_random_seed, Config
from model import Transformer
from tokenizer import load_tokenizer
from selection_strategy import greedy


def train_model(config):
    """
    Training process.
    """
    # make sure the weights folder exists
    if not os.path.exists(config.PATH_TO_MODEL_WEIGHTS):
        os.mkdir(config.PATH_TO_MODEL_WEIGHTS)

    # load tokenizers
    src_tokenizer = load_tokenizer(config=config, lang=config.LANG_SRC)
    tgt_tokenizer = load_tokenizer(config=config, lang=config.LANG_TGT)

    tr_dataloader, va_dataloader = get_dataset(config)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = Transformer(
        src_vocab_size = src_tokenizer.get_vocab_size(),
        tgt_vocab_size = tgt_tokenizer.get_vocab_size(),
        embed_size     = config.EMBED_SIZE,
        hidden_size    = config.HIDDEN_SIZE,
        max_len        = config.MAX_LEN,
        dropout        = config.DROPOUT,
        heads          = config.HEADS,
        N              = config.LAYERS,
    ).to(device)

    writer = SummaryWriter(config.PATH_TO_EXPERIMENTS)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-5, eps=1e-9)
    loss_fn = nn.CrossEntropyLoss(ignore_index=src_tokenizer.token_to_id("[PAD]"), label_smoothing=0.1).to(device)

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

            model_output = model(
                src=encoder_input, 
                tgt=decoder_input,
                src_mask=encoder_mask, 
                tgt_mask=decoder_mask
            )

            # compute the loss using a simple cross entropy
            loss = loss_fn(
                input=model_output.view(-1, tgt_tokenizer.get_vocab_size()), # (batch_size * max_len, vocab_size)
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

        batches_processed = 0
        with torch.no_grad():
            for va_batch in va_dataloader:
                acbatches_processedc += 1

                encoder_input = va_batch["encoder_input"].to(device) # (b, seq_len)
                encoder_mask = va_batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

                # decide on the next token
                model_output = greedy(
                    model=model,
                    src=encoder_input,
                    src_mask=encoder_mask,
                    src_tokenizer=src_tokenizer,
                    max_len=config.MAX_LEN,
                    device=device
                )

                src_text = va_batch["src_text"][0]
                tgt_text = va_batch["tgt_text"][0]
                prd_text = tgt_tokenizer.decode(model_output.detach().cpu().numpy())

                acc_bleu += bleu([prd_text], [[tgt_text]])
                acc_cer += cer(prd_text, tgt_text)
                acc_wer += wer(prd_text, tgt_text)

                if batches_processed % 10 == 0:
                    print(f"{f'SRC: ':>12}{src_text}")
                    print(f"{f'TGT: ':>12}{tgt_text}")
                    print(f"{f'PRD: ':>12}{prd_text}")
                    print('-'*80)

                if batches_processed == 100:
                    # will calculate va scores based on first 100 batches 
                    # (in our case 1 batch = 1 sentence)
                    break

        writer.add_scalar('validation CER', (acc_cer / batches_processed).item(), epoch)
        writer.add_scalar('validation WER', (acc_wer / batches_processed).item(), epoch)
        writer.add_scalar('validation BLEU', (acc_bleu / batches_processed).item(), epoch)
        writer.flush()
        writer.close()

        print(f'CER: {(acc_cer / batches_processed).item():.4f}')
        print(f'WER: {(acc_wer / batches_processed).item():.4f}')
        print(f'BLEU: {(acc_bleu / batches_processed).item():.4f}')

        # save the model at the end of every epoch
        model_filename = os.path.join(config.PATH_TO_MODEL_WEIGHTS, f'epoch_{epoch:02d}.pt')
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
