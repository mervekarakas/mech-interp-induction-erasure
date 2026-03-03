import torch


def induction_positions_and_targets(
    tokens: torch.Tensor, n_ctx_half: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Identify which positions to evaluate and what the correct next token is.

    In our sequence [BOS, x0, ..., x_{n-1}, x0, ..., x_{n-1}]:
    - The second half starts at index (1 + n_ctx_half).
    - At position (1 + n_ctx_half + i), the model should predict x_{i+1}
      (the token that followed x_i in the first half).
    - We can evaluate positions i=0, ..., n_ctx_half-2
      (the last position in the second half has no "next" within our sequence).

    Args:
        tokens: shape [batch, 1 + 2*n_ctx_half]
        n_ctx_half: length of each half

    Returns:
        positions: 1D tensor of positions to evaluate, shape [n_ctx_half - 1]
        target_ids: the correct next token at each position, shape [batch, n_ctx_half - 1]
    """
    # Positions in the second half where we evaluate (excluding the very last)
    second_half_start = 1 + n_ctx_half
    positions = torch.arange(second_half_start, second_half_start + n_ctx_half - 1)

    # The correct next token at position (second_half_start + i) is the token
    # at position (second_half_start + i + 1), which equals x_{i+1}
    target_ids = tokens[:, second_half_start + 1 : second_half_start + n_ctx_half]

    return positions, target_ids


def logit_diff(
    logits: torch.Tensor,
    positions: torch.Tensor,
    target_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the average logit difference: logit(correct) - mean(all logits).

    This measures how much the model "prefers" the correct token over the
    average alternative. Higher = stronger induction behavior.

    Args:
        logits: model output, shape [batch, seq_len, vocab_size]
        positions: which positions to evaluate, shape [n_positions]
        target_ids: correct token ids, shape [batch, n_positions]

    Returns:
        scalar: mean logit difference across batch and positions
    """
    # Grab logits at the positions we care about: [batch, n_positions, vocab_size]
    pos_logits = logits[:, positions, :]

    # Logit of the correct token at each position: [batch, n_positions]
    correct_logits = pos_logits.gather(2, target_ids.unsqueeze(2)).squeeze(2)

    # Mean logit across entire vocab at each position: [batch, n_positions]
    mean_logits = pos_logits.mean(dim=2)

    # Logit difference: how much correct exceeds the average
    diff = correct_logits - mean_logits

    return diff.mean()


def logit_diff_per_position(
    logits: torch.Tensor,
    positions: torch.Tensor,
    target_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Same as logit_diff but returns per-(batch, position) values instead
    of averaging. Used for conditional analysis (good vs bad positions).

    Returns:
        tensor of shape [batch, n_positions]
    """
    pos_logits = logits[:, positions, :]
    correct_logits = pos_logits.gather(2, target_ids.unsqueeze(2)).squeeze(2)
    mean_logits = pos_logits.mean(dim=2)
    return correct_logits - mean_logits
