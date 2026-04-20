from datasets import load_dataset
from fire import Fire


def flores_plus_to_files(src_lang: str, tgt_lang: str = "eng_Latn"):
    """
    Downloads the Flores+ dataset and saves it to source and target files.

    Args:
        src_lang: Source language code
        tgt_lang: Target language code
    """
    try:
        tgt_flores = load_dataset(
            "openlanguagedata/flores_plus", tgt_lang, split="devtest"
        )
        src_flores = load_dataset(
            "openlanguagedata/flores_plus", src_lang, split="devtest"
        )
    except Exception:
        print(f"ERROR: Language {tgt_lang} or {src_lang} not found")
        raise

    tgt_filename = f"flores_plus_{tgt_lang}.txt"
    src_filename = f"flores_plus_{src_lang}.txt"
    print(f"Saving to {tgt_filename} and {src_filename}")

    with open(tgt_filename, "w") as f:
        for i in tgt_flores:
            f.write(i["text"] + "\n")
    with open(src_filename, "w") as f:
        for i in src_flores:
            f.write(i["text"] + "\n")


def main():
    Fire(flores_plus_to_files)


if __name__ == "__main__":
    main()
