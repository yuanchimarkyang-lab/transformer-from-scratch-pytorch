"""
This script implements the training of an encoder-decoder transformer as a German to English translation model
"""

from pathlib import Path
import pandas as pd
import shutil
from functools import partial

from datasets import load_dataset
import sentencepiece as spm

import torch
from torch.utils.data import DataLoader


from transformer.config import load_config
from transformer.constants import PAD, BOS, EOS, UNK
from transformer.model import transformer
from transformer.data import collate_fn

def train(dataloader, model, loss_fn, optimizer):
    """
    This function train the model for one epoch
    Parameters:
        dataloader: the dataloader for training data.
        model: the pytorch model to be trained
        loss_fn: the loss function
        optimizer: the optimizer
    """
    size = len(dataloader.dataset)
    model.train()
    for batch, (encoder_input, decoder_input, decoder_output) in enumerate(dataloader):
        encoder_input, decoder_input, decoder_output = encoder_input.to(device), decoder_input.to(device), decoder_output.to(device) 

        # forward-propagation
        pred_score = model(encoder_input, decoder_input)
        loss = loss_fn(pred_score, decoder_output)

        # Back-propagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()



def evaluate(dataloader, model, loss_fn):
    """
    This function evaluate the loss and accuracy
    Parameters:
        dataloader: the dataloader for data to be evaluated.
        model: the pytorch model to be trained
        loss_fn: the loss function
    Return:
        loss: the loss
        accuracy: the percentage correct predictions
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    eval_loss, correct, num = 0, 0, 0
    with torch.no_grad():
        for encoder_input, decoder_input, decoder_output in dataloader:
            encoder_input, decoder_input, decoder_output = encoder_input.to(device), decoder_input.to(device), decoder_output.to(device)

            pred_score = model(encoder_input, decoder_input)
            
            eval_loss += loss_fn(pred_score, decoder_output).item()
            pred = pred_score.argmax(1)
            for idx in range(len(decoder_output)):
                seq_len = decoder_output[idx].nonzero(as_tuple=True)[0][-1]
                correct+=(pred[idx,:seq_len] == decoder_output[idx,:seq_len]).type(torch.float).sum().item()
                num+=seq_len

    eval_loss /=num_batches
    correct /=num

    return correct, eval_loss



if __name__ == "__main__":
    # load the configuration
    config_path = Path("configs/config.yaml")
    config = load_config(config_path)
    # create the output directory 
    output_path = Path(config["output_dir"])
    output_path.mkdir(exist_ok=True)
    # copy the configuration file
    shutil.copy(config_path, output_path)

    # create the 'checkpoints' folder under the output directory 
    print(f"The results will be stored at {output_path}")
    checkpoint_path = output_path / "checkpoints"
    checkpoint_path.mkdir(exist_ok=True)

    log_freq = config["log_freq"]
    save_freq = config["save_freq"]

    # load the data
    dataset = load_dataset("data/multi30k")
    print(dataset)

    dataset_train = dataset["train"]
    dataset_val = dataset["validation"]
    
    # load the Tokenizer
    vocab_size = config["vocab_size"]
    de_sp = spm.SentencePieceProcessor(model_file=f"data/tokenizer/de_{vocab_size}.model")
    en_sp = spm.SentencePieceProcessor(model_file=f"data/tokenizer/en_{vocab_size}.model")


    batch_size = 64
    collate_fn_filled = partial(collate_fn, de_sp=de_sp, en_sp=en_sp)
    train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_filled)
    val_dataloader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_filled)


    device = config["device"]
    
    model = transformer(D_model=config["D_model"],
                        h=config["h"], # n_heads 
                        Vocab_size = config["vocab_size"], # vocab_size, 
                        N_encoder = config["N_encoder"], 
                        N_decoder = config["N_decoder"], 
                        D_FFN=config["D_FFN"], 
                        PAD=PAD,
                        max_len=config["max_len"], 
                    ).to(device)


    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD) 
    optimizer = torch.optim.SGD(model.parameters(),lr=config["learning_rate"])
    # Using Scheduler to reduce learnign rate on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                            mode="min", 
                                                            factor=config["lr_factor"], 
                                                            patience=config["lr_patience"], 
                                                            min_lr=config["lr_min"])

    epochs = config["epochs"]
    
    info_list = []
    for t in range (epochs):
        # training
        train(train_dataloader, model, loss_fn, optimizer)
        train_accuracy, train_loss = evaluate(train_dataloader, model, loss_fn) 
        val_accuracy, val_loss = evaluate(val_dataloader, model, loss_fn)
        # use val_loss to determine the learning rate
        scheduler.step(val_loss)
        

        if (t+1) % save_freq == 0:
            # store the checkpoint every save_freq
            model_path = checkpoint_path / f"{t+1:06d}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"{t+1} checkpoint saved at {model_path}")

        if t % log_freq == 0:
            # output the evaluation results every log_freq
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {t+1}\n----------------------------------------")
            print(f"Learning Rate: {lr}")
            print(f"Train:\tAccuracy: {(100*train_accuracy):>0.1f}%, Avg loss: {train_loss:>8f}")
            print(f"Validation:\tAccuracy: {(100*val_accuracy):>0.1f}%, Avg loss: {val_loss:>8f}")

            
            info = {"epoch": t, 
                    "train_accuracy": train_accuracy,
                    "train_loss": train_loss,
                    "val_accuracy": val_accuracy,
                    "val_loss": val_loss,
                    "learning_rate": lr
            }
            info_list.append(info)

            df_info = pd.DataFrame(info_list)
            # save the evaluation results every log_freq
            df_info.to_csv(output_path / "df_info.csv", index=False)
            print(f"---------------------------------------------------------------")    
        
        
    
    df_info = pd.DataFrame(info_list)
    df_info.to_csv(output_path / "df_info.csv", index=False)
    print("Done!")
    print(f"Evaluation stored at {output_path}")



    