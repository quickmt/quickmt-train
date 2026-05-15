import datasets
import os
import fire
from typing import Optional
from .data import smart_open


def convert(
    dataset_name: str,
    src_lang: str,
    tgt_lang: str,
    src_filename: Optional[str] = None,
    tgt_filename: Optional[str] = None,
    limit: int = 500_000_000,
    num_proc: int = os.cpu_count(),
):
    """
    Downloads a dataset from Hugging Face and saves it to source and target files.

    Args:
        dataset_name: Hugging Face dataset name
        src_lang: Source language code
        tgt_lang: Target language code
        src_filename: Output filename for source language
        tgt_filename: Output filename for target language
        limit: Maximum number of samples to save
        num_proc: Number of processes to use for mapping
    """
    if not src_filename:
        src_filename = f"{dataset_name.split('/')[-1]}.{src_lang}.zst"
    if not tgt_filename:
        tgt_filename = f"{dataset_name.split('/')[-1]}.{tgt_lang}.zst"

    # 1. Load dataset (streaming=False is fine if it fits in RAM/cache)
    print("Loading dataset")
    ds = datasets.load_dataset(dataset_name, split="train", streaming=False)

    # Clear files if they exist
    for f in [src_filename, tgt_filename]:
        if os.path.exists(f):
            os.remove(f)

    # 3. Write data
    print("Creating output files: ", src_filename, tgt_filename)
    with smart_open(src_filename, "wt") as f_src, smart_open(tgt_filename, "wt") as f_tgt:
        batch_size = 10_000
        for i in range(0, len(ds), batch_size):
            batch = ds[i : i + batch_size]
            f_src.write("\n".join(batch[src_lang]) + "\n")
            f_tgt.write("\n".join(batch[tgt_lang]) + "\n")
            if i % (batch_size * 10) == 0:
                print(f"Processed {i} / {len(ds)} lines...")

    print("Done!")


def main():
    fire.Fire(convert)


if __name__ == "__main__":
    main()
