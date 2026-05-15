from datasets import load_dataset
from fire import Fire


def bouquet_to_files(split: str):
    """
    Downloads the facebook/bouquet dataset and saves it to source and target files.

    Args:
        split: Split to use
    """
    for dataset in ['dev', 'test']:
        bouquet_split = load_dataset("facebook/bouquet", split, split=dataset)

        src_filename = f"bouquet-{dataset}-{split}.txt"
        tgt_filename = f"bouquet-{dataset}-eng_Latn.txt"

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
