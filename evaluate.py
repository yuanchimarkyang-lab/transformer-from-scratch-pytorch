from pathlib import Path
import pandas as pd
import shutil
from functools import partial

from datasets import load_dataset
import sentencepiece as spm

import torch
from torch.utils.data import DataLoader
from torchmetrics.text import SacreBLEUScore

from transformer.config import load_config
from transformer.constants import PAD, BOS, EOS, UNK
from transformer.model import transformer
from transformer.data import collate_fn


sacre_bleu = SacreBLEUScore()

def BLEU_score_generated(dataloader, model, BLEU_func = sacre_bleu):

    model.eval()
    preds = []
    target = []
    with torch.no_grad():
        for encoder_input, decoder_input, decoder_output in dataloader:
            encoder_input, decoder_input, decoder_output = encoder_input.to(device), decoder_input.to(device), decoder_output.to(device)
            pred = greedy_decode(model,encoder_input,max_len=config["max_len"])
            preds+=en_sp.decode(pred.cpu().numpy())
            target+=en_sp.decode(decoder_output.cpu().numpy())

    bleu_score = 100*BLEU_func(preds = preds, target= [[x] for x in target]).item() 

    return bleu_score


def greedy_decode(model,encoder_input,max_len=100):
    model.eval()
    # The beginning of decoder_input starts with BOS
    decoder_input = torch.full((encoder_input.shape[0], 1), BOS, dtype = torch.long, device = encoder_input.device) 
    # This array as indicator if all the decoding has finished (True)
    decoder_end = torch.full((encoder_input.shape[0], 1), False, dtype = torch.bool, device = encoder_input.device)


    i = 0
    with torch.no_grad():
        while (i < max_len) & (not torch.all(decoder_end)):

            pred_score = model(encoder_input, decoder_input)
            decoder_output = pred_score[:,:,i].argmax(1).unsqueeze(1)

            append = decoder_output.masked_fill(decoder_end, value=PAD)


            decoder_input = torch.cat((decoder_input, append), axis = 1)

            decoder_end = decoder_end | (decoder_output == EOS)

            assert decoder_input.shape[1] == i+2, f"{decoder_input.shape}"

            i = i+1
    
    generated = decoder_input[:, 1:]

    return generated


if __name__ == "__main__":
    
    output_path = Path("results/baseline")
    config_path = output_path / "config.yaml"
    config = load_config(config_path)
    checkpoints = ["000010", "000020","000030","000040","000050","000060", "000070","000080","000090","000100"]
    

    dataset = load_dataset("data/multi30k")
    print(dataset)
    
    dataset_val = dataset["validation"]
    dataset_test = dataset["test"]


    # load the Tokenizer
    vocab_size = config["vocab_size"]
    de_sp = spm.SentencePieceProcessor(model_file=f"data/tokenizer/de_{vocab_size}.model")
    en_sp = spm.SentencePieceProcessor(model_file=f"data/tokenizer/en_{vocab_size}.model")

    batch_size = 64
    collate_fn_filled = partial(collate_fn, de_sp=de_sp, en_sp=en_sp)
    val_dataloader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_filled)
    test_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_filled)


    device = config["device"]

    model = transformer(D_model=config["D_model"],
                        h=config["h"], # n_heads 
                        Dict_size = config["vocab_size"], # vocab_size, 
                        N_encoder = config["N_encoder"], 
                        N_decoder = config["N_decoder"], 
                        D_FFN=config["D_FFN"], 
                        PAD=PAD,
                        max_len=config["max_len"], 
                    ).to(device)
    
    info_list = []
    for checkpoint in checkpoints:
        print(f"checkpoint {checkpoint}\n----------------------------------------")
        checkpoint_path = output_path / f"checkpoints/{checkpoint}.pth"

        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        print(f"The following checkpoint is loaded {checkpoint_path}")

        
        val_bleu_score = BLEU_score_generated(val_dataloader, model)
        print(f"Validation BLEU score: {val_bleu_score:>5.3f}")
        test_bleu_score = BLEU_score_generated(test_dataloader, model)
        print(f"Test BLEU score: {test_bleu_score:>5.3f}")

        info = {"checkpoint": checkpoint,
                "val_bleu_score": val_bleu_score, 
                "test_bleu_score": test_bleu_score,
            }
        info_list.append(info)

        df_info = pd.DataFrame(info_list)
        df_info.to_csv(output_path / "df_bleu.csv", index=False)
        print(f"---------------------------------------------------------------")    

    print("Done!")
    print(f"Evaluation stored at {output_path}")



    