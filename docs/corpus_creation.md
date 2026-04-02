# Downloading and Cleaning Machine Translation Data Tutorial

**WORK IN PROGRESS**

## Overview

Suppose you want to train a NMT model for Ukranian->English translation. The first step will be identifying which data to use to train your model. 

## 1 - Downloading Data

[`mt-data`](https://github.com/thammegowda/mtdata) is an excellent tool for managing and curating parallel corpora for NMT training, and we'll show how to use it to download NMT training data. First use the `mtdata list` to show the available datasets. Here we write the list of available datasets to a file named `corpora.txt`:

```bash
# We will create a working directory "is-en-1"
mkdir uk-en-1
cd uk-en-1

# If you don't have `mtdata` installed, pip install it
pip install mtdata
mtdata list --langs ukr-eng | cut -f1 > corpora.txt
```

Choosing which corpora to use for training your model is crucial and will affect performance.  Here is some guidance to help you in your choices:

* **Download Failures** - Some corpora are may not be available to download or have parsing errors. Try to download them a couple of times and if that doesn't work just remove them from your list.
* **Corpora Quality** - Some corpora contain high-quality parallel sentences for training your NMT model. Others contain web-scraped automatically generated parallel sentences. These are often lower quality due to the nature of the data collection process. Lower-quality corpora may contain errors, inconsistencies, or noise that can negatively impact model performance. 
* **Corpora Versions** - Different versions of the same corpus may have different quality levels or preprocessing steps. You might choose all available versions if you have enough storage and compute (we will deduplicate later), or if it is a very high resource language perhaps only include the most recent.
* **Datasets for Validation** - It is important to have a small, high-quality validation set to monitor the training process and prevent overfitting. This set should be representative of the data you expect to see in production.
* **Don't Train on Test Sets** - You want to be able to accurately compare your model's performance on the test set to other models or baselines. Don't include the flores "devtest", the "ntrex" dataset or Google-wmt24pp.
* **Localization** - Many datasets have localized versions (eng_GB, eng_CA, ...), it is probably best to stick with a single locale
* **Maybe Skip Opus-100** - All of the segments in Opus-100 are likely contained in the other corpora, so you may want to skip it.
* **Licenses** - Ensure that the corpora you choose are licensed for use in your project. Some datasets may have restrictive licenses that prevent you from using them for commercial purposes or in certain regions. Always check the license terms before using a dataset.


For this example we will use the following corpora:

```
Statmt-ccaligned-1-eng-ukr_UA
Tilde-worldbank-1-eng-ukr
Facebook-wikimatrix-1-eng-ukr
OPUS-ccaligned-v1-eng-ukr
OPUS-ccmatrix-v1-eng-ukr
OPUS-elrc_3043_wikipedia_health-v1-eng-ukr
OPUS-elrc_5174_french_polish_ukrain-v1-eng-ukr
OPUS-elrc_5179_acts_ukrainian-v1-eng-ukr
OPUS-elrc_5180_official_parliament_-v1-eng-ukr
OPUS-elrc_5181_official_parliament_-v1-eng-ukr
OPUS-elrc_5182_official_parliament_-v1-eng-ukr
OPUS-elrc_5183_scipar_ukraine-v1-eng-ukr
OPUS-elrc_5214_a_lexicon_named-v1-eng-ukr
OPUS-elrc_5217_ukrainian_legal_mt-v1-eng-ukr
OPUS-elrc_wikipedia_health-v1-eng-ukr
OPUS-elrc_2922-v1-eng-ukr
OPUS-eubookshop-v2-eng-ukr
OPUS-gnome-v1-eng-ukr
OPUS-hplt-v2-eng-ukr
OPUS-kde4-v2-eng-ukr
OPUS-kdedoc-v1-eng_GB-ukr
OPUS-macocu-v2-eng-ukr
OPUS-multimacocu-v2-eng-ukr
OPUS-nllb-v1-eng-ukr
OPUS-neulab_tedtalks-v1-eng-ukr
OPUS-opensubtitles-v2016-eng-ukr
OPUS-opensubtitles-v2018-eng-ukr
OPUS-opensubtitles-v2024-eng-ukr
OPUS-paracrawl-v9-eng-ukr
OPUS-paracrawl_bonus-v9-eng-ukr
OPUS-qed-v2.0a-eng-ukr
OPUS-summa-v1-eng-ukr
OPUS-ted2020-v1-eng-ukr
OPUS-tatoeba-v2-eng-ukr
OPUS-tatoeba-v20190709-eng-ukr
OPUS-tatoeba-v20200531-eng-ukr
OPUS-tatoeba-v20201109-eng-ukr
OPUS-tatoeba-v20210310-eng-ukr
OPUS-tatoeba-v20210722-eng-ukr
OPUS-tatoeba-v20220303-eng-ukr
OPUS-tatoeba-v20230412-eng-ukr
OPUS-tildemodel-v2018-eng-ukr
OPUS-ubuntu-v14.10-eng-ukr
OPUS-wikimatrix-v1-eng-ukr
OPUS-xlent-v1-eng-ukr
OPUS-xlent-v1.1-eng-ukr
OPUS-xlent-v1.2-eng-ukr
OPUS-bible_uedin-v1-eng-ukr
OPUS-tldr_pages-v20230829-eng-ukr
OPUS-wikimedia-v20210402-eng-ukr
OPUS-wikimedia-v20230407-eng-ukr
```

and following for validation:

```
Flores-flores200_dev-1-eng-ukr
Statmt-generaltest-2022_refA-ukr-eng 
Statmt-generaltest-2023_refA-ukr-eng 
```

We use the `mtdata` tool again to download this data using the following syntax:

```bash
mtdata get -l ukr-eng --merge --out ./  -j 4  \
--dev  Flores-flores200_dev-1-eng-ukr Statmt-generaltest-2022_refA-ukr-eng Statmt-generaltest-2023_refA-ukr-eng \
--train Statmt-ccaligned-1-eng-ukr_UA Tilde-worldbank-1-eng-ukr Facebook-wikimatrix-1-eng-ukr OPUS-ccmatrix-v1-eng-ukr OPUS-elrc_3043_wikipedia_health-v1-eng-ukr OPUS-elrc_5174_french_polish_ukrain-v1-eng-ukr OPUS-elrc_5179_acts_ukrainian-v1-eng-ukr OPUS-elrc_5180_official_parliament_-v1-eng-ukr OPUS-elrc_5181_official_parliament_-v1-eng-ukr OPUS-elrc_5182_official_parliament_-v1-eng-ukr OPUS-elrc_5183_scipar_ukraine-v1-eng-ukr OPUS-elrc_5214_a_lexicon_named-v1-eng-ukr OPUS-elrc_5217_ukrainian_legal_mt-v1-eng-ukr OPUS-elrc_wikipedia_health-v1-eng-ukr OPUS-elrc_2922-v1-eng-ukr OPUS-eubookshop-v2-eng-ukr OPUS-gnome-v1-eng-ukr OPUS-hplt-v2-eng-ukr OPUS-kde4-v2-eng-ukr OPUS-kdedoc-v1-eng_GB-ukr OPUS-macocu-v2-eng-ukr OPUS-multimacocu-v2-eng-ukr OPUS-nllb-v1-eng-ukr OPUS-neulab_tedtalks-v1-eng-ukr OPUS-opensubtitles-v2016-eng-ukr OPUS-opensubtitles-v2018-eng-ukr OPUS-opensubtitles-v2024-eng-ukr OPUS-paracrawl-v9-eng-ukr OPUS-paracrawl_bonus-v9-eng-ukr OPUS-qed-v2.0a-eng-ukr OPUS-summa-v1-eng-ukr OPUS-ted2020-v1-eng-ukr OPUS-tatoeba-v2-eng-ukr OPUS-tatoeba-v20190709-eng-ukr OPUS-tatoeba-v20200531-eng-ukr OPUS-tatoeba-v20201109-eng-ukr OPUS-tatoeba-v20210310-eng-ukr OPUS-tatoeba-v20210722-eng-ukr OPUS-tatoeba-v20220303-eng-ukr OPUS-tatoeba-v20230412-eng-ukr OPUS-tildemodel-v2018-eng-ukr OPUS-ubuntu-v14.10-eng-ukr OPUS-wikimatrix-v1-eng-ukr OPUS-xlent-v1-eng-ukr OPUS-xlent-v1.1-eng-ukr OPUS-xlent-v1.2-eng-ukr OPUS-bible_uedin-v1-eng-ukr OPUS-tldr_pages-v20230829-eng-ukr OPUS-wikimedia-v20210402-eng-ukr OPUS-wikimedia-v20230407-eng-ukr
```

After quite a while (largely a function of your download speed) this will create four key files:

* dev.eng - English validation data
* dev.ukr - Ukrainian validation data
* train.eng - English training data
* train.ukr - Ukrainian training data

As well as a folder "tests" containing the test set files, and a folder "train-parts" containing the individual training corpora. **Note:** `mtdata` will also cache corpus downloads in a seperate directory, "~/.mtdata" by default. Keep an eye on your disk space and periodically clean this directory to free up space.

## 1.1 - Look For Other Datasets

The more clean data you can find the better, so take a [look around](https://huggingface.co/datasets?task_categories=task_categories:translation&language=language:uk,language:en&sort=trending) for other sources of translation training data.

Some possibilities:

* https://huggingface.co/datasets/ayymen/Weblate-Translations
* https://huggingface.co/datasets/HuggingFaceFW/finetranslations


## 2 - Basic Filter

Next we will use a basic filter to clean up the data. 

This will do several things reasonably efficiently for large corpora using multiple cores:

* Deduplication
* Length ratio filtering
* Language ID filtering
* Sentence length filtering
* Custom rules-based filtering
* Shuffle the data

Note that it assumes `gnu parallel` is installed.

```bash
# Download language identification model
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

# Setting src_min_langid_score to 0 in case Language ID confuses different Cyrillic languages
paste -d '\t' train.ukr train.eng \
    | sort | uniq  \
    | parallel --block 70M -j 4 --pipe -k -l 200000 quickmt-clean-primary --src_lang uk --tgt_lang en --length_ratio 3 --src_min_langid_score 0 --tgt_min_langid_score 0.5 --ft_model_path="lid.176.bin" \
    | awk 'BEGIN{srand()}{print rand(), $0}' | sort -n -k 1 | awk 'sub(/\S* /,"\t")' \
    | awk -v FS="\t" '{ print $2 > "train.cleaned.ukr" ; print $3 > "train.cleaned.eng" }'
```

## 3 - Semantic Filter

Next we will use sentence transformer static embedding model to filter out semantically dissimilar sentences. 

```bash
quickmt-clean-embeddings \
    --src_input train.cleaned.ukr \
    --src_output train.cleaned.filtered.ukr \
    --tgt_input train.cleaned.eng \
    --tgt_output train.cleaned.filtered.eng \
    --src_dev dev.ukr \
    --tgt_dev dev.eng \
    --src_bad_output filtered.bad.ukr \
    --tgt_bad_output filtered.bad.eng \
    --sim_cutoff_quantile 0.01
```

## 5 - Upload Cleaned Data to HuggingFace Hub

Once you have downloaded and cleaned up your data, upload it to the Huggingface Hub to make it available for others to use and share.

```bash
# Make sure you are logged in
hf auth login

quickmt-dataset-upload \
    quickmt/quickmt-train.uk-en \
    --src_in train.cleaned.filtered.ukr  \
    --tgt_in train.cleaned.filtered.eng  \
    --src_lang uk \
    --tgt_lang en
```

