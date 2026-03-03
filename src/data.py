import torch


def make_induction_tokens(
    batch: int,
    n_ctx_half: int,
    bos_token_id: int,
    vocab_min: int = 1000,
    vocab_max: int = 40000,
    seed: int = 0,
) -> torch.Tensor:
    """
    Generate synthetic induction sequences: [BOS, x0, x1, ..., x_{n-1}, x0, x1, ..., x_{n-1}]

    The second half is an exact copy of the first half.
    If the model has induction heads, it should predict tokens in the
    second half much better — because it can "look back" to the first
    occurrence and copy the next token.

    Args:
        batch: number of sequences
        n_ctx_half: length of each half (so total seq len = 1 + 2*n_ctx_half)
        bos_token_id: the model's BOS token id
        vocab_min: lower bound of token id range to sample from (avoids special tokens)
        vocab_max: upper bound of token id range to sample from
        seed: random seed for reproducibility

    Returns:
        tokens: shape [batch, 1 + 2*n_ctx_half]
    """
    gen = torch.Generator()
    gen.manual_seed(seed)

    # Sample random token IDs for the first half
    first_half = torch.randint(
        vocab_min, vocab_max, (batch, n_ctx_half), generator=gen
    )

    # Second half is an exact copy
    second_half = first_half.clone()

    # Prepend BOS token
    bos = torch.full((batch, 1), bos_token_id, dtype=torch.long)

    # [BOS, first_half, second_half]
    tokens = torch.cat([bos, first_half, second_half], dim=1)
    return tokens
