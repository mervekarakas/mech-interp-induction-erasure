import torch


def corrupt_prefix_random_replace(
    tokens: torch.Tensor,
    n_ctx_half: int,
    vocab_min: int = 1000,
    vocab_max: int = 40000,
    eps: float = 0.5,
    seed: int = 0,
) -> torch.Tensor:
    """
    Corrupt the first half of the sequence by replacing a fraction `eps` of
    tokens with random token IDs.

    Only touches positions [1, 1+n_ctx_half) — the first half, excluding BOS.
    The second half stays identical to the (now corrupted) first half? No!
    The second half stays AS-IS (still a copy of the ORIGINAL first half).

    This means the second half still repeats the original pattern, but the
    first half no longer matches — so the model can't "look back" and find
    the pattern it expects. This breaks induction.

    Args:
        tokens: shape [batch, 1 + 2*n_ctx_half], original clean tokens
        n_ctx_half: length of each half
        vocab_min: min token id for random replacements
        vocab_max: max token id for random replacements
        eps: fraction of first-half tokens to replace (0.0 = no corruption, 1.0 = all replaced)
        seed: random seed

    Returns:
        corrupted tokens (new tensor, does not modify the input)
    """
    gen = torch.Generator()
    gen.manual_seed(seed)

    corrupted = tokens.clone()
    batch = tokens.shape[0]

    # Create a mask: which positions in the first half get replaced?
    # Shape: [batch, n_ctx_half]
    mask = torch.rand(batch, n_ctx_half, generator=gen) < eps

    # Generate random replacement tokens
    replacements = torch.randint(vocab_min, vocab_max, (batch, n_ctx_half), generator=gen)

    # Apply: only replace where mask is True, only in first half (positions 1 to n_ctx_half)
    corrupted[:, 1 : 1 + n_ctx_half][mask] = replacements[mask]

    return corrupted


def corrupt_prefix_fixed_token(
    tokens: torch.Tensor,
    n_ctx_half: int,
    fixed_token_id: int,
    eps: float = 0.5,
    seed: int = 0,
) -> torch.Tensor:
    """
    Same idea as random replacement, but replace with a single fixed token
    (e.g., the token for "the"). This is our robustness check — if results
    hold across both corruption types, the finding is more trustworthy.

    Args:
        tokens: shape [batch, 1 + 2*n_ctx_half]
        n_ctx_half: length of each half
        fixed_token_id: the token id to use as replacement
        eps: fraction of first-half tokens to replace
        seed: random seed

    Returns:
        corrupted tokens (new tensor)
    """
    gen = torch.Generator()
    gen.manual_seed(seed)

    corrupted = tokens.clone()
    batch = tokens.shape[0]

    mask = torch.rand(batch, n_ctx_half, generator=gen) < eps
    corrupted[:, 1 : 1 + n_ctx_half][mask] = fixed_token_id

    return corrupted
