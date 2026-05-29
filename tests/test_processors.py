import pytest
import os
import random
from typing import Optional, Tuple
from quickmt_train.processors import TextProcessor, load_processor, ProcessingPipeline, CharacterNoiseProcessor
from quickmt_train.config import DataConfig, CorpusConfig


class MockUppercaseProcessor(TextProcessor):
    def __init__(self, target_only: bool = False):
        self.target_only = target_only

    def __call__(self, src: str, tgt: str, global_step: int) -> Optional[Tuple[str, str]]:
        if self.target_only:
            return src, tgt.upper()
        return src.upper(), tgt.upper()


class MockFilteringProcessor(TextProcessor):
    def __init__(self, filter_word: str):
        self.filter_word = filter_word

    def __call__(self, src: str, tgt: str, global_step: int) -> Optional[Tuple[str, str]]:
        if self.filter_word in src or self.filter_word in tgt:
            return None
        return src, tgt


def test_text_processor_abc_and_pipeline():
    # Setup test processors in pipeline configuration format
    pipeline_config = [
        {
            "path": "test_processors.MockUppercaseProcessor",
            "start_step": 0,
            "stop_step": 10,
            "kwargs": {"target_only": True}
        },
        {
            "path": "test_processors.MockFilteringProcessor",
            "start_step": 5,
            "stop_step": 20,
            "kwargs": {"filter_word": "BAD"}
        }
    ]

    pipeline = ProcessingPipeline(pipeline_config)

    # Step 0: Only MockUppercaseProcessor is active. Target should be capitalized.
    res = pipeline("hello", "world", global_step=0)
    assert res == ("hello", "WORLD")

    # Step 6: Both processors are active. "BAD" is not present.
    res = pipeline("hello", "world", global_step=6)
    assert res == ("hello", "WORLD")

    # Step 6: Both processors are active. Target gets capitalized to "BAD" which triggers filter.
    res = pipeline("hello", "bad", global_step=6)
    assert res is None

    # Step 15: MockUppercaseProcessor is inactive (stop_step is 10).
    # MockFilteringProcessor is active. Target doesn't get capitalized, "bad" triggers filter.
    res = pipeline("hello", "world", global_step=15)
    assert res == ("hello", "world")


def test_filepath_syntax_loading():
    # Write a quick temporary processor file to verify file-loading plugin architecture
    temp_file = "/tmp/kilo/temp_processor.py"
    os.makedirs(os.path.dirname(temp_file), exist_ok=True)
    with open(temp_file, "w") as f:
        f.write("""
from quickmt_train.processors import TextProcessor
class FileBasedProcessor(TextProcessor):
    def __init__(self, suffix: str = ""):
        self.suffix = suffix
    def __call__(self, src, tgt, step):
        return src + self.suffix, tgt + self.suffix
""")

    processor_path = f"{temp_file}:FileBasedProcessor"
    processor = load_processor(processor_path, {"suffix": "!"})
    
    assert isinstance(processor, TextProcessor)
    res = processor("hello", "world", 0)
    assert res == ("hello!", "world!")


def test_character_noise_processor():
    # Fix seed to ensure reproducible checks
    random.seed(42)

    # Test swaps (highly effective orthographic noise simulating fat-finger slips)
    swap_proc = CharacterNoiseProcessor(swap_prob=1.0)
    res_s, res_t = swap_proc("abcd", "target", 0)
    assert res_s != "abcd"
    assert len(res_s) == 4
    assert res_t == "target"

    # Test deletion
    del_proc = CharacterNoiseProcessor(deletion_prob=1.0)
    res_s, res_t = del_proc("abcd", "target", 0)
    assert len(res_s) < 4
    assert res_t == "target"

    # Test repetition
    ins_proc = CharacterNoiseProcessor(repeat_prob=1.0)
    res_s, res_t = ins_proc("abcd", "target", 0)
    assert len(res_s) > 4
    assert res_t == "target"

    # Test substitution (case swapping via flipcase)
    sub_proc = CharacterNoiseProcessor(flipcase_prob=1.0)
    res_s, res_t = sub_proc("abcd", "target", 0)
    assert res_s == "ABCD"
    assert res_t == "target"


def test_length_filter_processor():
    from quickmt_train.processors import LengthFilterProcessor
    
    filter_proc = LengthFilterProcessor(min_char_length=5, max_char_length=20, length_ratio=2.0)
    
    # Standard valid pair
    assert filter_proc("hello", "world", 0) == ("hello", "world")
    
    # Too short
    assert filter_proc("abc", "world", 0) is None
    
    # Too long
    assert filter_proc("hello world how are you today", "world", 0) is None
    
    # Extreme length ratio (5 vs 15 -> ratio 3.0, exceeds limit 2.0)
    assert filter_proc("hello", "world world world", 0) is None


