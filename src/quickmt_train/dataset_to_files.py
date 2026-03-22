import datasets
import os
import fire
from typing import Optional


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
        src_filename = f"{dataset_name.split('/')[-1]}.{src_lang}"
    if not tgt_filename:
        tgt_filename = f"{dataset_name.split('/')[-1]}.{tgt_lang}"

    # 1. Load dataset (streaming=False is fine if it fits in RAM/cache)
    print("Loading dataset")
    ds = datasets.load_dataset(dataset_name, split="train", streaming=False)

    # Clear files if they exist
    for f in [src_filename, tgt_filename]:
        if os.path.exists(f):
            os.remove(f)

    # 3. Use a batched writing function
    def write_batches(batch):
        with open(src_filename, "a", encoding="utf-8") as f_src, open(
            tgt_filename, "a", encoding="utf-8"
        ) as f_tgt:
            if len(batch) > 0:
                f_src.write("\n".join(batch[src_lang]) + "\n")
                f_tgt.write("\n".join(batch[tgt_lang]) + "\n")
            else:
                # Map expects a return, even if empty
                return {}

    # 4. Execute with multiple processes (num_proc) and batching
    print("Creating output files: ", src_filename, tgt_filename)
    ds.map(
        write_batches,
        batched=True,
        batch_size=10_000,
        num_proc=num_proc,  # Use all available cores
        desc="Downloading data",
    )

    print("Done!")


def main():
    fire.Fire(convert)


if __name__ == "__main__":
    main()
