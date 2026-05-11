from datasets import load_dataset
from fire import Fire


def bouquet_to_files(split: str):
    """
    Downloads the facebook/bouquet dataset and saves it to source and target files.

    Args:
        split: Split to use
    """
    bouquet_split = load_dataset("facebook/bouquet", split, split="test")

    src_filename = f"bouquet-{split}.txt"
    tgt_filename = f"bouquet-eng_Latn.txt"

    print(f"Saving to {tgt_filename} and {src_filename}")
    with open(tgt_filename, "w") as f_tgt:
        with open(src_filename, "w") as f_src:
            for i in bouquet_split:
                f_tgt.write(i["tgt_text"] + "\n")
                f_src.write(i["src_text"] + "\n")


def main():
    Fire(bouquet_to_files)


if __name__ == "__main__":
    main()
