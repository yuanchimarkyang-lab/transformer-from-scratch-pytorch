"""
This script is to generate figures to evaluate the training, including Loss vs Epoch and BLEU vs Epoch
"""
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == "__main__":
    folder = "results/baseline"

    df_info = pd.read_csv(f"{folder}/df_info.csv")
    df_bleu = pd.read_csv(f"{folder}/df_bleu.csv")

    # Plot the Loss Curve
    plt.plot(df_info["epoch"], df_info["train_loss"], color = 'k', marker = 'o', linestyle = '-', label = "Train")
    plt.plot(df_info["epoch"], df_info["val_loss"], color = 'b', marker = 'x', linestyle = '--', label = "Validation")
    plt.grid()
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Train/Val Loss", fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{folder}/loss_curve.png",dpi=600)

    # Plot the Validation/Test BLEU score versus checkpoint (epoch)
    plt.plot(df_bleu["checkpoint"], df_bleu["val_bleu_score"], color = 'b', marker = 'x', linestyle = '--', label = "Validation")
    plt.plot(df_bleu["checkpoint"], df_bleu["test_bleu_score"], color = 'r', marker = 's', linestyle = '-.', label = "Test")
    plt.grid()
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("BLEU score", fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{folder}/BLEU_curve.png",dpi=600)

    print(f"Done! Figures are generated in {folder}")


