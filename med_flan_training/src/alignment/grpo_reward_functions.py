import os
import re
import traceback
from pathlib import Path

import nltk
import numpy as np
import torch

from sentence_transformers import SentenceTransformer


nltk.download("punkt")

try:
    import dill as pkl
except ImportError:
    import pickle as pkl

import gc
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
from nltk import word_tokenize
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
from rouge_score import rouge_scorer

from scipy.special import erf


def save_pkl(data: any, fname: str):
    with open(fname, "wb") as file:
        pkl.dump(data, file, protocol=pkl.HIGHEST_PROTOCOL)


def load_pkl(fname: str):
    with open(fname, "rb") as file:
        data = pkl.load(file)
    return data


def remove_stopwords(text):
    nltk.download("stopwords")
    from nltk.corpus import stopwords

    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text.lower())
    return [w for w in tokens if w.isalnum() and w not in stop_words]


def min_max_normalize(x):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    x_min, x_max = x.min(), x.max()
    if x_max == x_min:
        return x
    return (x - x_min) / (x_max - x_min)


def gaussian_normalize(x):
    """Map an array-like of scores to (0,1) using the Normal CDF after z-score standardization.

    If all values are identical (std == 0), returns an array filled with 0.5 (the CDF at z=0).
    This avoids division by zero and preserves a neutral midpoint reward.
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x, dtype=float)
    else:
        x = x.astype(float)

    std = x.std()
    if std == 0:
        # All entries identical; standardized value would be 0 -> CDF(0) = 0.5
        return np.full_like(x, 0.5, dtype=float)

    # Standardize (z-score) then apply normal CDF using erf (output in (0, 1))
    z = (x - x.mean()) / std
    return 0.5 * (1 + erf(z / np.sqrt(2)))


def get_bleu_score(preds, refs):
    scs = []
    for a, b in zip(preds, refs):
        # Remove stopwords and tokenize for candidate and reference
        cand_tokens = remove_stopwords(a)
        references = remove_stopwords(b)
        # sentence_bleu expects a list of reference token lists and a candidate token list
        score = sentence_bleu(
            [references],  # wrap single reference list
            cand_tokens,
            weights=(0.3, 0.3, 0.3, 0.1),
            smoothing_function=SmoothingFunction().method1,
        )
        scs.append(score)
    return scs


def _cal_meteror_score(pred, ref):
    tokenized_candidate = word_tokenize(pred)
    tokenized_reference = word_tokenize(ref)
    return single_meteor_score(tokenized_reference, tokenized_candidate)


def get_meteor_score(preds, refs):
    scs = []
    for a, b in zip(preds, refs):
        msc = _cal_meteror_score(a, b)
        scs.append(msc)

    return scs


def get_rouge_score(preds, refs):
    scorer = rouge_scorer.RougeScorer(
        rouge_types=["rougeLsum"], use_stemmer=False, tokenizer=None
    )
    scs = []
    for a, b in zip(preds, refs):
        score = scorer.score(a, b)
        scs.append(score["rougeLsum"].fmeasure)
    return scs


def length_penalty(preds, refs, gamma=0.001):
    len_scs = []
    for a, b in zip(preds, refs):
        len_a = len(a)
        len_b = len(b)

        len_ratio = min(len_a, len_b) / max(len_a, len_b)

        # diff = len_a - len_b
        # exp_penalty = np.exp(-gamma * diff**2)
        # penalty = 0.5 * (len_ratio + exp_penalty)

        len_scs.append(len_ratio)

    return len_scs


def tag_count_reward(completions, **kwargs) -> list[float]:
    """Reward function that checks if we produce the desired number of think and answer tags associated with `format_reward()`.

    Adapted from: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb#file-grpo_demo-py-L90
    """

    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("<think>\n") == 1:
            count += 0.25
        if text.count("\n</think>\n") == 1:
            count += 0.25
        if text.count("\n<answer>\n") == 1:
            count += 0.25
        if text.count("\n</answer>") == 1:
            count += 0.25
        return count

    contents = [completion[0]["content"] for completion in completions]
    return [count_tags(c) for c in contents]


def get_repetition_penalty_reward(
    ngram_size: int, max_penalty: float, language: str = "en"
):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://huggingface.co/papers/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    language: Language of the text, defaults to `en`. Used to choose the way to split the text into n-grams.
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    if language == "en":

        def zipngram(text: str, ngram_size: int):
            words = text.lower().split()
            return zip(*[words[i:] for i in range(ngram_size)]), words

    elif language == "zh":
        from transformers.utils.import_utils import _is_package_available

        if not _is_package_available("jieba"):
            raise ValueError("Please install jieba to use Chinese language")

        def zipngram(text: str, ngram_size: int):
            import jieba

            seg_list = list(jieba.cut(text))
            return zip(*[seg_list[i:] for i in range(ngram_size)]), seg_list

    else:
        raise ValueError(
            f"Word splitting for language `{language}` is not yet implemented. Please implement your own zip-ngram function."
        )

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        reward function the penalizes repetitions
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            completions: List of model completions
        """

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "":
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            ngram_array, words = zipngram(completion, ngram_size)

            if len(words) < ngram_size:
                rewards.append(0.0)
                continue

            for ng in ngram_array:
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward


def reward_num_unique_letters(completions, **kwargs):
    """Reward function that rewards completions with more unique letters."""
    completion_contents = [completion[0]["content"] for completion in completions]
    return [float(len(set(content))) for content in completion_contents]


def bioleaflet_reward_function(completions, ground_truth, **kwargs):
    predicted_results = []
    for completion in completions:
        for each in completion:
            if each["role"] == "assistant":
                predicted_results.append(each["content"])
                break

    meteor_score = min_max_normalize(get_meteor_score(predicted_results, ground_truth))
    rouge_score = min_max_normalize(get_rouge_score(predicted_results, ground_truth))
    bleu_score = min_max_normalize(get_bleu_score(predicted_results, ground_truth))
    length_penalty_score = min_max_normalize(
        length_penalty(predicted_results, ground_truth)
    )

    overall_score = (
        0.4 * meteor_score
        + 0.3 * rouge_score
        + 0.2 * bleu_score
        + 0.1 * length_penalty_score
    )

    return overall_score.tolist()


def bioleaflet_reward_function_2(completions, ground_truth, **kwargs):
    predicted_results = []
    for completion in completions:
        for each in completion:
            if each["role"] == "assistant":
                predicted_results.append(each["content"])
                break

    meteor_score = gaussian_normalize(get_meteor_score(predicted_results, ground_truth))
    rouge_score = gaussian_normalize(get_rouge_score(predicted_results, ground_truth))
    length_penalty_score = gaussian_normalize(
        length_penalty(predicted_results, ground_truth)
    )

    overall_score = 0.5 * meteor_score + 0.4 * rouge_score + 0.1 * length_penalty_score

    return overall_score


def get_predicted_results(completions):
    """Utility to extract assistant content from completions."""
    predicted_results = []
    for completion in completions:
        for each in completion:
            if each["role"] == "assistant":
                predicted_results.append(each["content"])
                break
    return predicted_results


def meteor_reward_function(completions, ground_truth, **kwargs):
    predicted_results = get_predicted_results(completions)
    meteor_score = get_meteor_score(predicted_results, ground_truth)

    rewards = [e * 1.5 for e in meteor_score]

    return rewards


def rouge_reward_function(completions, ground_truth, **kwargs):
    predicted_results = get_predicted_results(completions)
    rouge_score = get_rouge_score(predicted_results, ground_truth)
    rewards = [e * 1.5 for e in rouge_score]
    return rewards
    # return min_max_normalize(rouge_score).tolist()


def bleu_reward_function(completions, ground_truth, **kwargs):
    predicted_results = get_predicted_results(completions)
    bleu_score = get_bleu_score(predicted_results, ground_truth)
    return bleu_score
    # return min_max_normalize(bleu_score).tolist()


class PushPullReward:
    def __init__(self, n):
        self.n = n
        self.__name__ = "push_pull_reward_function"

    def _cal_score(self, completions, ground_truth, i):
        group_completions = completions[i : i + self.n]
        group_targets = ground_truth[i : i + self.n]
        rewards = []
        for j, (current_gen, current_target) in enumerate(
            zip(group_completions, group_targets)
        ):
            r_sim = _cal_meteror_score(current_target, current_gen)
            sibling_scores = []
            for k, sibling_gen in enumerate(group_completions):
                if j != k:
                    sibling_scores.append(_cal_meteror_score(current_gen, sibling_gen))
            avg_sibling_sim = np.mean(sibling_scores)
            r_div = 1.0 - avg_sibling_sim
            total_reward = r_sim + 0.2 * r_div
            rewards.append(total_reward)

        return rewards

    def __call__(self, completions, ground_truth, **kwargs):
        rewards = []
        _completions = get_predicted_results(completions)

        with ProcessPoolExecutor(max_workers=12) as executor:
            _rewards = executor.map(
                partial(self._cal_score, _completions, ground_truth),
                range(0, len(_completions), self.n),
            )

        for each in _rewards:
            rewards.extend(each)

        return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [
        re.match(pattern, content, re.DOTALL | re.MULTILINE)
        for content in completion_contents
    ]
    return [1.0 if match else 0.0 for match in matches]


def format_reward_1(completions, **kwargs):
    """
    Reward function that checks if the entire content is enclosed as:
       <think>...</think>
       <answer>...</answer>
    in that exact order (with optional whitespace in between).

    Returns:
      -1 if no match,
       0.0 if there's a match.
    """

    # Use the new pattern to prevent some reward hacking.
    pattern = (
        r"^\s*<think>.*?</think>\s*<answer>(?:(?!<answer>|</answer>).)*?</answer>\s*$"
    )
    format_scores = []

    for comp in completions:
        content = comp[0]["content"]
        # Check if the content matches the entire pattern.
        if re.fullmatch(pattern, content, re.DOTALL | re.MULTILINE):
            format_scores.append(0.0)  # match => 0.0
        else:
            format_scores.append(-2)  # no match => -2

    return format_scores


# aim for training deepseek distilled models
def get_predicted_results_with_template(completions):
    """Utility to extract assistant content from completions, extracting only the answer text from properly formatted responses."""
    predicted_results = []
    pattern = (
        r"^\s*<think>.*?</think>\s*<answer>((?:(?!<answer>|</answer>).)*?)</answer>\s*$"
    )

    for completion in completions:
        for each in completion:
            if each["role"] == "assistant":
                content = each["content"]
                match = re.match(pattern, content, re.DOTALL | re.MULTILINE)
                if match:
                    # Extract only the text between <answer> tags
                    predicted_results.append(match.group(1).strip())
                else:
                    # If format doesn't match, use the entire content
                    predicted_results.append(content)
                break
    return predicted_results


def meteor_reward_with_template_function(completions, ground_truth, **kwargs):
    predicted_results = get_predicted_results_with_template(completions)
    meteor_score = get_meteor_score(predicted_results, ground_truth)
    # You should still normalize each reward to a consistent scale (e.g., 0-1)
    return meteor_score
    # return min_max_normalize(meteor_score).tolist()


def rouge_reward_with_template_function(completions, ground_truth, **kwargs):
    predicted_results = get_predicted_results_with_template(completions)
    rouge_score = get_rouge_score(predicted_results, ground_truth)
    rewards = [e * 1.5 for e in rouge_score]
    return rewards
    # return min_max_normalize(rouge_score).tolist()


class SemanticSimilarityReward:
    """
    Reward function using SentenceTransformers.
    Optimized for Distributed GRPO (Data Parallel) + Memory Saving (Sleep/Wake).
    """

    CACHE_THRESHOLD = 32

    def __init__(
        self,
        model_name,
        tmp_dir="tmp",
        batch_size=4,
        max_model_len=4096,
        **kwargs,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_model_len = max_model_len
        self.cache_path = Path(tmp_dir)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.model = None
        self._is_sleeping = True
        self.__name__ = "semantic_similarity_reward"

        # --- GPU DISTRIBUTION LOGIC ---
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # 2. Assign the specific device
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)  # Explicitly set context
        else:
            self.device = torch.device("cpu")

        print(f"[Process {self.local_rank}] Initialized Reward on {self.device}")

        # 3. Load Model immediately to CPU (to verify path), then sleep
        self._load_model_to_cpu()

    def _load_model_to_cpu(self):
        """Loads model to System RAM (CPU) to verify and prepare."""
        if self.model is None:
            self.model = SentenceTransformer(
                self.model_name,
                # model_kwargs={
                #     "attn_implementation": "flash_attention_2",
                # },
                tokenizer_kwargs={
                    "padding_side": "left",
                    "max_length": self.max_model_len,
                },
                device="cpu",
                trust_remote_code=True,
            )

            self._is_sleeping = True

    def _wake_engine(self):
        """Moves weights from CPU -> GPU"""
        if self._is_sleeping and self.model is not None:
            self.model.to(self.device)
            self._is_sleeping = False

    def _sleep_engine(self):
        """Moves weights from GPU -> CPU to free VRAM for Training"""
        if not self._is_sleeping and self.model is not None:
            self.model.to("cpu")
            self._is_sleeping = True

            # Force cleanup so TRL has full VRAM
            torch.cuda.empty_cache()
            gc.collect()

    def _encode(self, preds, targets):
        """Wake -> Encode -> Sleep"""
        self._wake_engine()
        # print("[DEBUG PREDS]: ", "\n\n".join(preds))
        try:
            pred_embeddings = self.model.encode(
                preds,
                batch_size=self.batch_size,
                device=self.device,
                normalize_embeddings=True,
            )

            target_embeddings = self.model.encode(
                targets,
                batch_size=self.batch_size,
                device=self.device,
                normalize_embeddings=True,
            )

            simlarity_scores = (
                self.model.similarity_pairwise(pred_embeddings, target_embeddings)
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )

            rewards = [r**5 for r in simlarity_scores]

            return rewards
        except:
            traceback.print_exc()
            # instead of return 0 reward scores, we will return None to crash the training process
            return None
        finally:
            self._sleep_engine()

    def __call__(self, completions, ground_truth, **kwargs):
        # 1. Encode Predictions (Local Batch)
        completions_contents = get_predicted_results(completions)
        rewards = self._encode(completions_contents, ground_truth)
        return rewards


reward_function_dict = {
    "bioleaflet_reward_function": bioleaflet_reward_function,
    "bioleaflet_reward_function_2": bioleaflet_reward_function_2,
    "reward_num_unique_letters": reward_num_unique_letters,
    "meteor_reward_function": meteor_reward_function,
    "rouge_reward_function": rouge_reward_function,
    "bleu_reward_function": bleu_reward_function,
    "push_pull_reward_function": PushPullReward,
    "semantic_similarity_reward": SemanticSimilarityReward,
    "format_reward": format_reward_1,
    "meteor_reward_with_template_function": meteor_reward_with_template_function,
    "rouge_reward_with_template_function": rouge_reward_with_template_function,
}


# Deterministic, per-task medical reward for the meddata multitask mix (exact-match
# for classification/QA, doc-level micro-F1 for IE), routed by the `task` column that
# convert_to_grpo now carries through. Imported at the bottom (not module top) to keep
# the meddata-scoring module decoupled from this file's nltk/rouge/sentence-transformers
# imports; the reward is a plain function, so run_grpo.py's no-constructor branch
# (line 148-149) picks it up with no special case.
from .med_grpo_rewards import med_multitask_reward

reward_function_dict["med_multitask_reward"] = med_multitask_reward
