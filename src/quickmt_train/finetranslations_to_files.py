import fire

from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path


def convert(
    src_lang: str,
    output_folder: str = "./",
    length_ratio: int = 3,
    max_char_len: int = 1000,
    min_char_len: int = 4,
    stream_dataset: bool = True,
    line_limit: int = 100_000_000,
):
    """Download and Filter Huggingface FineTranslations Data

    Args:
        src_lang (str, optional): Source languages in flores format.
        output_folder (str, optional): Output folder. Defaults to "./".
        length_ratio (int, optional): Char length ratio for filtering. Defaults to 3.
    """
    tgt_lang: str = "eng_Latn"

    print("Loading dataset HuggingFaceFW/finetranslations")

    ds = load_dataset(
        "HuggingFaceFW/finetranslations",
        name=src_lang,
        split="train",
        streaming=stream_dataset,
    )

    # Raise exception if output files exist
    src_output_file = (
        Path(output_folder) / f"finetranslations.{src_lang}-{tgt_lang}.{src_lang}"
    )
    tgt_output_file = (
        Path(output_folder) / f"finetranslations.{src_lang}-{tgt_lang}.{tgt_lang}"
    )

    if src_output_file.exists():
        raise FileExistsError(f"Source file {src_output_file} exists - will not remove")

    if tgt_output_file.exists():
        raise FileExistsError(f"Target file {tgt_output_file} exists - will not remove")

    line_counter = 0

    with open(src_output_file, "wt") as srcfile:
        with open(
            tgt_output_file,
            "wt",
        ) as tgtfile:
            for x in tqdm(ds):
                mt = x["translated_chunks"][0].splitlines()
                src = x["og_chunks"][0].splitlines()
                # Ensure same number of lines in src and tgt
                if len(src) == len(mt):
                    for i, j in zip(src, mt):
                        slen = len(i)
                        tlen = len(j)
                        # Ensure src and tgt segments are within length limits
                        if (
                            slen > min_char_len
                            and tlen > min_char_len
                            and slen < max_char_len
                            and tlen < max_char_len
                        ):
                            # Ensure src and tgt segments are within length ratio limits
                            if (
                                slen / tlen > 1.0 / length_ratio
                                and slen / tlen < length_ratio
                            ):
                                line_counter += 1
                                srcfile.write(i + "\n")
                                tgtfile.write(j + "\n")
                                if line_counter >= line_limit:
                                    print("Line limit reached, exiting...")
                                    return

    print("Line limit not reached, done!")


def main():
    fire.Fire(convert)


if __name__ == "__main__":
    main()
