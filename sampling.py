# %%
import os
import sys
from dataclasses import dataclass

import torch as t
import transformers
from einops import rearrange, repeat
from tqdm.auto import tqdm

import utils
import w2d3_test
from w2d3_part1_loading_solution import GPT2, GPT2Block, load_pretrained_weights

os.environ["TOKENIZERS_PARALLELISM"] = "false"
MAIN = __name__ == "__main__"
IS_CI = os.getenv("IS_CI")
if IS_CI:
    sys.exit(0)

# %%
if MAIN:
    my_gpt = load_pretrained_weights().eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")


# %%
"""
## Sampling Boilerplate

The provided functions `sample_tokens` and `sample_next_token` include the boilerplate for sampling from the model. Note that there is a special token `tokenizer.eos_token`, which during training was added to the end of a each article. GPT-2 will generate this token when it feels like the continuation is at a reasonable stopping point, which is our cue to stop generation.

The functions called in `sample_next_token` are not defined yet - you are going to implement them below.
"""
# %%
def sample_next_token(
    model: GPT2,
    input_ids: t.Tensor,
    temperature=1.0,
    freq_penalty=0.0,
    top_k=0,
    top_p=0.0,
    cache=None,
) -> int:
    """Return the next token, sampled from the model's probability distribution with modifiers.

    input_ids: shape (seq,)
    """
    assert input_ids.ndim == 1, "input_ids should be a 1D sequence of token ids"
    assert temperature >= 0
    assert 0 <= top_p <= 1.0, "Top-p must be a probability"
    assert 0 <= top_k, "Top-k must be non-negative"
    assert not (top_p != 0 and top_k != 0), "At most one of top-p and top-k supported"
    model.eval()
    with t.inference_mode():
        all_logits = model(input_ids.unsqueeze(0), cache=cache)
    B, S, V = all_logits.shape
    assert B == 1
    assert S == len(input_ids)
    logits = all_logits[0, -1]
    if temperature == 0:
        return greedy_search(logits)

    logits = apply_temperature(logits, temperature)
    logits = apply_freq_penalty(input_ids, logits, freq_penalty)
    if top_k > 0:
        return sample_top_k(logits, top_k)
    if top_p > 0:
        return sample_top_p(logits, top_p)
    return sample_basic(logits)


def sample_tokens(
    model: GPT2,
    tokenizer,
    initial_text: str,
    max_tokens_generated=30,
    temperature=1.0,
    freq_penalty=0.0,
    stop_at_eos=True,
    top_k=0,
    top_p=0.0,
    cache=None,
) -> str:
    """Sample tokens using sample_next_token until the model outputs `tokenizer.eos_token_id` or the specified token limit is reached.

    Return: the prompt and continuation concatenated
    """
    model.eval()
    input_ids: list = tokenizer(initial_text).input_ids
    generated = []
    device = next(model.parameters()).device
    for _ in tqdm(range(max_tokens_generated)):
        new_token = sample_next_token(
            model,
            t.tensor(input_ids + generated, dtype=t.int64, device=device),
            temperature=temperature,
            freq_penalty=freq_penalty,
            top_k=top_k,
            top_p=top_p,
            cache=cache,
        )
        generated.append(new_token)
        if stop_at_eos and new_token == tokenizer.eos_token_id:
            break
    return tokenizer.decode(input_ids + generated)


"""
## Greedy Search

Returns the most likely next token. If multiple tokens are equally likely, break the tie by returning the smallest token.
"""

# %%
def greedy_search(logits: t.Tensor) -> int:
    """
    logits: shape (vocab_size, )

    Return: the most likely token
    """
    out = logits.argmax().item()
    assert isinstance(out, int)
    return out


if MAIN:
    logits = t.ones(100)
    logits[5] = 10
    logits[8] = 10
    assert greedy_search(logits) == 5

    w2d3_test.test_sample_zero_temperature(my_gpt, tokenizer, sample_tokens)

"""
### Temperature

dividing the logits by the temperature. As temperature goes to zero, this becomes the same as greedy sampling, and as the temperature goes to infinity this becomes the same as sampling from a uniform distribution.

"""
# %%
def apply_temperature(logits: t.Tensor, temperature: float) -> t.Tensor:
    """
    logits: shape (vocab_size, )

    Return: shape (vocab_size, )
    """
    assert temperature > 0
    "SOLUTION"
    return logits / temperature


if MAIN:
    logits = t.tensor([1, 2]).log()
    cold_logits = apply_temperature(logits, 1e-3)
    print('A low temperature "sharpens" or "peaks" the distribution: ', cold_logits)
    utils.allclose(cold_logits, 1e3 * logits)

    hot_logits = apply_temperature(logits, 1e3)
    print("A high temperature flattens the distribution: ", hot_logits)
    utils.allclose(hot_logits, 1e-3 * logits)

# %%
"""
### Frequency Penalty
count the number of occurrences of each token, then subtract `freq_penalty` for each occurrence.
"""
# %%
def apply_freq_penalty(input_ids: t.Tensor, logits: t.Tensor, freq_penalty: float) -> t.Tensor:
    """
    input_ids: shape (seq, )
    logits: shape (vocab_size, )

    Return: shape (vocab_size, )
    """
    (vocab_size,) = logits.shape
    id_freqs = t.bincount(input_ids, minlength=vocab_size)
    return logits - freq_penalty * id_freqs


if MAIN:
    bieber_prompt = "And I was like Baby, baby, baby, oh Like, Baby, baby, baby, no Like, Baby, baby, baby, oh I thought you'd always be mine, mine"
    input_ids = tokenizer(bieber_prompt, return_tensors="pt")["input_ids"][0]

    logits = t.ones(tokenizer.vocab_size)
    penalized_logits = apply_freq_penalty(input_ids, logits, 2.0)

    assert penalized_logits[5156].item() == -11, "Expected 6 occurrences of ' baby' with leading space"
    assert penalized_logits[14801].item() == -5, "Expected 3 occurrences of ' Baby' with leading space"

# %%
"""
## Sampling with `Categorical`
"""
# %%
def sample_basic(logits: t.Tensor) -> int:
    """
    logits: shape (vocab_size, ) - unnormalized log-probabilities

    Return: a sampled token
    """
    out = t.distributions.categorical.Categorical(logits=logits).sample().item()
    assert isinstance(out, int)
    return out


if MAIN:
    N = 20000
    probs = t.linspace(0, 0.4, 5)
    unnormalized_logits = probs.log() + 1.2345
    samples = t.tensor([sample_basic(unnormalized_logits) for _ in range(N)])
    counts = t.bincount(samples, minlength=len(probs)) / N
    print(
        "Checking empirical frequencies (try to increase N if this test fails): ",
        counts,
    )
    utils.allclose_atol(counts, probs, atol=0.01)

# %%
"""
## Sampling - Manual Testing
"""

# %%
if MAIN:
    N_RUNS = 1
    your_prompt = "Jingle bells, jingle bells, jingle all the way"

    cases = [
        ("High freq penalty", dict(freq_penalty=100.0)),
        ("Negative freq penalty", dict(freq_penalty=-1.0)),
        ("Too hot!", dict(temperature=2.0)),
        ("Pleasantly cool", dict(temperature=0.7)),
        ("Pleasantly warm", dict(temperature=0.9)),
    ]

    for name, kwargs in cases:
        for i in range(N_RUNS):
            output = sample_tokens(my_gpt, tokenizer, your_prompt, max_tokens_generated=24, **kwargs)
            print(f"Sample {i} with: {name} ({kwargs}):")
            print(f"Your model said: {repr(output)}")


# %%
"""
## Top-K Sampling

Conceptually, the steps in top-k sampling are:
- Find the `top_k` largest probabilities
- Set all other probabilities to zero
- Normalize and sample
"""
# %%
def sample_top_k(logits: t.Tensor, top_k: int) -> int:
    """
    logits: shape (vocab_size, ) - unnormalized log-probabilities
    top_k: only consider this many of the most likely tokens for sampling

    Return: a sampled token
    """
    top_logits, top_idx = logits.topk(top_k)
    idx = t.distributions.categorical.Categorical(logits=top_logits).sample()
    out = top_idx[idx].item()
    assert isinstance(out, int)
    return out


if MAIN:
    k = 3
    probs = t.linspace(0, 0.4, 5)
    unnormalized_logits = probs.log() + 1.2345
    samples = t.tensor([sample_top_k(unnormalized_logits, k) for _ in range(N)])
    counts = t.bincount(samples, minlength=len(probs)) / N
    expected = probs.clone()
    expected[:-k] = 0
    expected /= expected.sum()
    print(
        "Checking empirical frequencies (try to increase N if this test fails): ",
        counts,
    )
    utils.allclose_atol(counts, expected, atol=0.01)

# %%
"""
### Top-K Sampling - Example
"""
# %%
if MAIN:
    your_prompt = "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English."
    output = sample_tokens(
        my_gpt,
        tokenizer,
        your_prompt,
        temperature=0.7,
        top_k=40,
        max_tokens_generated=64,
    )
    print(f"Your model said: {repr(output)}")

# %%
"""
## Top-p aka Nucleus Sampling

Conceptually, in top-p we:
- Sort the probabilities from largest to smallest
- Find the cutoff point where the cumulative probability first equals or exceeds `top_p`. We do the cutoff inclusively, keeping the first probability above the threshold.
- If the number of kept probabilities is less than `min_tokens_to_keep`, keep that many tokens instead.
- Set all other probabilities to zero
- Normalize and sample
"""

# %%
def sample_top_p(logits: t.Tensor, top_p: float, min_tokens_to_keep: int = 1) -> int:
    """
    logits: shape (vocab_size, ) - unnormalized log-probabilities

    Return: a sampled token
    """
    logits_sorted, indices = logits.sort(descending=True, stable=True)
    cumul_probs = logits_sorted.softmax(-1).cumsum(-1)
    n_keep = t.searchsorted(cumul_probs, top_p, side="right").item() + 1
    n_keep = max(n_keep, min_tokens_to_keep)
    keep_idx = indices[:n_keep]
    keep_logits = logits[keep_idx]
    sample = t.distributions.categorical.Categorical(logits=keep_logits).sample()
    out = keep_idx[sample].item()
    assert isinstance(out, int)
    return out


if MAIN:
    N = 2000
    unnormalized_logits = t.tensor([0.2, 0.3, 0.5]).log() + 2.3456
    samples = t.tensor([sample_top_p(unnormalized_logits, 0.5) for _ in range(N)])
    counts = t.bincount(samples, minlength=len(unnormalized_logits)) / N
    print(
        "top_p of 0.5 or lower should only return token 2: ",
        counts,
    )
    assert counts[0] == 0 and counts[1] == 0

# %%
if MAIN:
    N = 2000
    unnormalized_logits = t.tensor([0.2, 0.3, 0.5]).log() + 2.3456
    samples = t.tensor([sample_top_p(unnormalized_logits, 0.50001) for _ in range(N)])
    counts = t.bincount(samples, minlength=len(unnormalized_logits)) / N
    print(
        "top_p in (0.5, 0.8] should return tokens 1 and 2: ",
        counts,
    )
    assert counts[0] == 0

# %%
if MAIN:
    N = 2000
    top_p = 0.71
    probs = t.linspace(0, 0.4, 5)
    unnormalized_logits = probs.log() + 1.2345
    samples = t.tensor([sample_top_p(unnormalized_logits, top_p) for _ in range(N)])
    counts = t.bincount(samples, minlength=len(probs)) / N
    expected = probs.clone()
    expected[0:2] = 0
    expected /= expected.sum()
    print(
        "Checking empirical frequencies (try to increase N if this test fails): ",
        counts,
    )
    utils.allclose_atol(counts, expected, atol=0.01)

# %%
"""
### Top-P Sampling Example
"""
#
if MAIN:
    your_prompt = "Eliezer Shlomo Yudkowsky (born September 11, 1979) is an American decision and artificial intelligence (AI) theorist and writer, best known for"
    output = sample_tokens(
        my_gpt,
        tokenizer,
        your_prompt,
        temperature=0.7,
        top_p=0.95,
        max_tokens_generated=64,
    )
    print(f"Your model said: {repr(output)}")

# %%
def sample_tokens_with_cache(
    model: GPT2,
    tokenizer,
    initial_text: str,
    max_tokens_generated=30,
    temperature=1.0,
    freq_penalty=2.0,
    stop_at_eos=True,
    top_k=0,
    top_p=0.0,
    cache=None,
) -> str:
    """Does the exact same thing as sample_tokens, but using cache to be faster."""
    device = next(model.parameters()).device
    if cache is None:
        cache = GPT2CacheEntry.new_empty(model)
    input_ids: list = tokenizer(initial_text).input_ids
    generated = []

    x = t.tensor(input_ids, dtype=t.int64, device=device)
    for _ in tqdm(range(max_tokens_generated)):
        next_token = sample_next_token(
            model,
            x,
            temperature=temperature,
            freq_penalty=freq_penalty,
            top_k=top_k,
            top_p=top_p,
            cache=cache,
        )
        generated.append(next_token)
        if stop_at_eos and next_token == tokenizer.eos_token_id:
            break
        x = t.tensor([next_token], dtype=t.int64, device=device)
    return tokenizer.decode(input_ids + generated)


if MAIN:
    w2d3_test.test_identical_output_with_cache(
        my_gpt,
        tokenizer,
        "It is pitch black. You are likely to be eaten by a grue.",
        sample_tokens,
        sample_tokens_with_cache,
    )

# %%
"""
## Beam Search

In beam search, we maintain a list of size `num_beams` completions which are the most likely completions so far as measured by the product of their probabilities. Since this product can become very small, we use the sum of log probabilities instead.

At each iteration, we run the batch of completions through the model and take the log-softmax to obtain `vocab_size` log-probs for each completion, or `num_beams * vocab_size` possible next completions in total.

If we kept all of these, then we would have `num_beams * vocab_size * vocab_size` completions after the next iteration which is way too many, so instead we sort them by their score and loop through from best (highest) log probability to worst (lowest).

For each next completion, if it ends in the end of sequence (EOS) token then we add it to a list of finished completions along with its score. Otherwise, we add it to the list of "to be continued" completions. The iteration is complete when the "to be continued" list has `num_beams` entries in it.

If our finished list now contains at least `num_return_sequences` completions, then we are done. If the length of the completion is now `len(prompt) + max_new_tokens`, then we are also done. Otherwise, we go to the next iteration.
"""
# %%
def beam_search(
    model,
    input_ids: t.Tensor,
    num_return_sequences: int,
    num_beams: int,
    max_new_tokens: int,
    tokenizer,
    verbose=False,
) -> list[tuple[float, t.Tensor]]:
    """
    input_ids: (seq, ) - the prompt

    max_new_tokens: stop after this many new tokens are generated, even if no EOS is generated. In this case, the best incomplete sequences should also be returned.
    verbose: if True, print the current (unfinished) completions after each iteration for debugging purposes

    Return list of length num_return_sequences. Each element is a tuple of (logprob, tokens) where the tokens include both prompt and completion, sorted by descending logprob.
    """
    assert num_return_sequences <= num_beams
    "SOLUTION"
    current_logp = t.tensor([0.0], device=input_ids.device)  # batch,
    (prompt_len,) = input_ids.shape
    current_ids = repeat(input_ids, "s -> 1 s")
    finished = []

    for seq_len in tqdm(range(prompt_len, prompt_len + max_new_tokens)):
        with t.inference_mode():
            token_logp = model(current_ids)[:, -1].log_softmax(-1)
        B, V = token_logp.shape
        # next_scores[V*b + v] = logp[b] + next[b, v]
        next_logp_flat = rearrange(token_logp, "b v -> (b v)") + repeat(current_logp, "b -> (b v)", v=V)

        # Why do we need 2 times beam width?
        # Some of the new tokens might be EOS, and if beam_width of them are EOS then we're guaranteed done.
        # If only beam_width-1 of them are EOS, we need to have beam_width more candidates for the next iteration.
        _, flat_indexes = next_logp_flat.topk(2 * num_beams)
        bs = (flat_indexes / V).long()  # TBD is this just t.div(mode="floor")? Recover small b
        vs = flat_indexes % V  # recover small v

        next_ids = current_ids.new_zeros((num_beams, seq_len + 1))
        next_logp = current_logp.new_zeros((num_beams,))
        beams_used = 0

        for tok, ind in zip(vs, bs):
            score = next_logp_flat[V * ind + tok]
            if tok == tokenizer.eos_token_id:
                finished.append((score.item(), current_ids[ind]))
            else:
                next_ids[beams_used, :seq_len] = current_ids[ind]
                next_ids[beams_used, seq_len] = tok
                next_logp[beams_used] = score
                beams_used += 1

            if beams_used >= num_beams:
                break  # next_ids is full
        else:
            assert False, "Logic error - not enough topk to fill beam!"

        if len(finished) >= num_return_sequences:
            break
        else:
            current_ids = next_ids
            current_logp = next_logp

            if verbose:
                print("Current Completions:\n")
                texts = tokenizer.batch_decode(current_ids)
                for score, text in zip(current_logp, texts):
                    print(f"{score:.3f} {repr(text)}")
                print()

    # If we hit the length limit, add incomplete prompts to finished
    for i in range(num_return_sequences - len(finished)):
        finished.append((current_logp[i].item(), current_ids[i]))

    return sorted(finished, reverse=True)[:num_return_sequences]


if MAIN:
    your_prompt = "I don't want to rule the universe. I just think"
    input_ids = tokenizer(your_prompt, return_tensors="pt", return_attention_mask=False)["input_ids"][0]
    beam_out = w2d3_test.test_beam_search(
        beam_search,
        model=my_gpt,
        input_ids=input_ids,
        num_return_sequences=2,
        num_beams=6,
        max_new_tokens=20,
        tokenizer=tokenizer,
        verbose=True,
    )
    print("Final Completions: ")
    for score, tokens in beam_out:
        print(f"{score:.4f}: {repr(tokenizer.decode(tokens))}")

# %%
# Run this to verify that our reference solution matches HuggingFace
# if MAIN:
#     pretrained_gpt = utils.load_pretrained_gpt()
#     pretrained_out = pretrained_gpt.generate(
#         input_ids.view(1, -1),
#         num_return_sequences=2,
#         num_beams=6,
#         max_new_tokens=20,
#         early_stopping=True,
#         eos_token_id=tokenizer.eos_token_id,
#         pad_token_id=tokenizer.pad_token_id,
#     )
#     print(tokenizer.batch_decode(pretrained_out))
