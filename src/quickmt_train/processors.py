import importlib
import random
import sys
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any, List


class TextProcessor(ABC):
    """
    Abstract base class for text processors (augmentations and filters).
    
    A TextProcessor takes a source string and a target string, and can:
    1. Filter out the pair (by returning None)
    2. Augment/modify the strings (by returning the modified (src, tgt) tuple)
    3. Pass them through unchanged (by returning the original (src, tgt) tuple)
    """

    def __init__(self, **kwargs):
        """
        Initialize the processor with configuration parameters.
        """
        pass

    @abstractmethod
    def __call__(self, src: str, tgt: str, global_step: int) -> Optional[Tuple[str, str]]:
        """
        Process a source and target sentence pair.

        Args:
            src: Source sentence.
            tgt: Target sentence.
            global_step: Current global training step.

        Returns:
            A tuple of (processed_src, processed_tgt) or None if the sample should be filtered out.
        """
        pass


class CharacterNoiseProcessor(TextProcessor):
    """
    Character-level noise augmentation optimized for source-side NMT.
    Supports:
    - Character deletion (randomly dropping a character)
    - Character repetition (randomly duplicating/repeating a character)
    - Neighboring character swap (simulating typos / keyboard slips)
    - Case-flip (swapping uppercase and lowercase characters)
    """

    def __init__(
        self,
        deletion_prob: float = 0.0,
        repeat_prob: float = 0.0,
        swap_prob: float = 0.0,
        flipcase_prob: float = 0.0,
    ):
        self.deletion_prob = deletion_prob
        self.repeat_prob = repeat_prob
        self.swap_prob = swap_prob
        self.flipcase_prob = flipcase_prob

    def __call__(self, src: str, tgt: str, global_step: int) -> Optional[Tuple[str, str]]:
        if not src:
            return src, tgt

        # If all probabilities are 0, return immediately to bypass word parsing entirely
        if not (self.swap_prob > 0 or self.deletion_prob > 0 or self.repeat_prob > 0 or self.flipcase_prob > 0):
            return src, tgt

        words = src.split()
        new_words = []

        for word in words:
            # We only noise words longer than 1 character to avoid stripping small tokens completely
            if len(word) <= 1:
                new_words.append(word)
                continue

            chars = list(word)
            i = 0
            while i < len(chars):
                r = random.random()

                # 1. Neighboring character swap (highly representative of human typing errors)
                if r < self.swap_prob and i < len(chars) - 1:
                    chars[i], chars[i + 1] = chars[i + 1], chars[i]
                    i += 2  # skip next character as we swapped it

                # 2. Character deletion (captures omissions/misspellings)
                elif r < self.swap_prob + self.deletion_prob:
                    chars.pop(i)
                    # Don't increment i because current index now points to the next char

                # 3. Character repetition (simulates double-presses / key repeats)
                elif r < self.swap_prob + self.deletion_prob + self.repeat_prob:
                    # Insert the same character (simulating double-press)
                    chars.insert(i, chars[i])
                    i += 2

                # 4. Case-flip (simulates caps-lock errors or shift key slips)
                elif r < self.swap_prob + self.deletion_prob + self.repeat_prob + self.flipcase_prob:
                    if chars[i].isalpha():
                        chars[i] = chars[i].swapcase()
                    i += 1
                else:
                    i += 1

            # Avoid producing completely empty words from heavy deletions
            new_word = "".join(chars)
            if new_word:
                new_words.append(new_word)

        return " ".join(new_words), tgt


class LengthFilterProcessor(TextProcessor):
    """
    A filter processor that uses `char_length_match` from `filter_basic.py`
    to filter out pairs with invalid lengths or extreme ratios.
    """

    def __init__(
        self,
        min_char_length: int = 3,
        max_char_length: int = 2000,
        length_ratio: float = 4.0,
    ):
        from .filter_basic import char_length_match
        self.char_length_match = char_length_match
        self.min_char_length = min_char_length
        self.max_char_length = max_char_length
        self.length_ratio = length_ratio

    def __call__(self, src: str, tgt: str, global_step: int) -> Optional[Tuple[str, str]]:
        if self.char_length_match(
            src,
            tgt,
            min_char_length=self.min_char_length,
            max_char_length=self.max_char_length,
            length_ratio=self.length_ratio,
        ):
            return src, tgt
        return None


def load_processor(path: str, kwargs: Dict[str, Any]) -> TextProcessor:
    """
    Dynamically loads a TextProcessor subclass.
    
    Args:
        path: A string in the format 'module.submodule.ClassName' or an absolute filepath
              with the class name, e.g., '/path/to/module.py:ClassName'.
        kwargs: Dictionary of keyword arguments to pass to the processor's constructor.
        
    Returns:
        An instance of TextProcessor.
    """
    if ":" in path:
        # File path syntax: /absolute/path/to/file.py:ClassName
        file_path, class_name = path.rsplit(":", 1)
        import os
        if not os.path.isabs(file_path):
            file_path = os.path.abspath(file_path)
        
        module_name = f"dynamic_module_{hash(file_path) & 0xffffffff}"
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module spec from file: {file_path}")
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    else:
        # Standard python import path: module.submodule.ClassName
        if "." not in path:
            raise ValueError(f"Processor path must be in 'module.ClassName' or 'path/to/file.py:ClassName' format: {path}")
        module_name, class_name = path.rsplit(".", 1)
        module = importlib.import_module(module_name)

    clazz = getattr(module, class_name)
    if not issubclass(clazz, TextProcessor):
        raise TypeError(f"Class {class_name} is not a subclass of TextProcessor")
        
    return clazz(**kwargs)


class ProcessingPipeline:
    """
    Executes an ordered pipeline of TextProcessors, respecting step ranges.
    """

    def __init__(self, processors_config: List[Dict[str, Any]]):
        self.pipeline = []
        for cfg in processors_config:
            path = cfg["path"]
            start_step = cfg.get("start_step", 0)
            stop_step = cfg.get("stop_step", 10000000)
            kwargs = cfg.get("kwargs", {})
            
            processor = load_processor(path, kwargs)
            self.pipeline.append({
                "processor": processor,
                "start_step": start_step,
                "stop_step": stop_step
            })

    def __call__(self, src: str, tgt: str, global_step: int) -> Optional[Tuple[str, str]]:
        current_pair = (src, tgt)
        for stage in self.pipeline:
            if stage["start_step"] <= global_step <= stage["stop_step"]:
                res = stage["processor"](current_pair[0], current_pair[1], global_step)
                if res is None:
                    return None
                current_pair = res
        return current_pair

