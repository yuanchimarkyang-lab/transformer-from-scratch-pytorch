from pathlib import Path
from datasets import load_dataset
import sentencepiece as spm

from transformer.config import load_config

if __name__ == "__main__":

    config_path = Path("configs/config.yaml")
    config = load_config(config_path)
    
    dataset = load_dataset("bentrevett/multi30k")
    dataset.save_to_disk("data/multi30k")

    # prepare the training data for the tokenizer
    with open("data/tokenizer/train.de", "w", encoding="utf-8") as f_de, \
        open("data/tokenizer/train.en", "w", encoding="utf-8") as f_en:

        for example in dataset["train"]:
            f_de.write(example["de"].strip() + "\n")
            f_en.write(example["en"].strip() + "\n")

    vocab_size = config["vocab_size"]
    # train tokenizer
    spm.SentencePieceTrainer.train(
        input="data/tokenizer/train.de",
        model_prefix=f"data/tokenizer/de_{vocab_size}",
        vocab_size=vocab_size,
        model_type="bpe",
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
    )

    spm.SentencePieceTrainer.train(
        input="data/tokenizer/train.en",
        model_prefix=f"data/tokenizer/en_{vocab_size}",
        vocab_size=vocab_size,
        model_type="bpe",
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
    )