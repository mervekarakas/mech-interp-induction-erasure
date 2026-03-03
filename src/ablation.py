import torch
from transformer_lens import HookedTransformer
from typing import Callable


def ablate_head(
    model: HookedTransformer,
    tokens: torch.Tensor,
    layer: int,
    head: int,
    metric_fn: Callable,
) -> float:
    """
    Run the model with one attention head zeroed out, and return the metric.

    "Ablation" = disable a component and see what happens. If the metric
    drops, that head was helping. If the metric RISES, that head was
    actually hurting (pushing the model toward wrong answers).

    We zero out the head's z tensor (pre-output-projection activations)
    at all positions. This effectively removes that head's contribution
    to the residual stream.

    Args:
        model: the transformer
        tokens: input tokens
        layer: which layer
        head: which head to ablate
        metric_fn: function(logits) -> scalar

    Returns:
        metric value with the head ablated
    """
    hook_name = f"blocks.{layer}.attn.hook_z"

    def ablate_hook(activation, hook):
        # activation: [batch, seq_len, n_heads, d_head]
        # Zero out this head's contribution
        activation[:, :, head, :] = 0.0
        return activation

    logits = model.run_with_hooks(
        tokens,
        fwd_hooks=[(hook_name, ablate_hook)],
    )
    return metric_fn(logits).item()


def ablate_head_logits(
    model: HookedTransformer,
    tokens: torch.Tensor,
    layer: int,
    head: int,
) -> torch.Tensor:
    """
    Same as ablate_head but returns the raw logits tensor instead of a
    scalar metric. Used for per-position conditional analysis.
    """
    hook_name = f"blocks.{layer}.attn.hook_z"

    def ablate_hook(activation, hook):
        activation[:, :, head, :] = 0.0
        return activation

    logits = model.run_with_hooks(
        tokens,
        fwd_hooks=[(hook_name, ablate_hook)],
    )
    return logits


def compute_ablation_effects(
    model: HookedTransformer,
    tokens: torch.Tensor,
    metric_fn: Callable,
    candidate_heads: list[tuple[int, int]] = None,
) -> dict:
    """
    For each head, compute how much ablating it changes the metric.

    contribution = metric(normal) - metric(ablated)
      positive → head helps (ablation hurts performance)
      negative → head hurts (ablation improves performance!)

    Args:
        model: the transformer
        tokens: input tokens
        metric_fn: function(logits) -> scalar
        candidate_heads: list of (layer, head) tuples to test.
                        If None, tests ALL heads.

    Returns:
        dict with:
            'baseline': metric without any ablation
            'ablated_metrics': dict mapping (layer, head) -> metric when ablated
            'contributions': dict mapping (layer, head) -> contribution value
    """
    # Baseline: no ablation
    with torch.no_grad():
        baseline_logits = model(tokens)
    baseline = metric_fn(baseline_logits).item()

    if candidate_heads is None:
        candidate_heads = [
            (l, h)
            for l in range(model.cfg.n_layers)
            for h in range(model.cfg.n_heads)
        ]

    ablated_metrics = {}
    contributions = {}

    for layer, head in candidate_heads:
        abl_metric = ablate_head(model, tokens, layer, head, metric_fn)
        ablated_metrics[(layer, head)] = abl_metric
        contributions[(layer, head)] = baseline - abl_metric

    return {
        'baseline': baseline,
        'ablated_metrics': ablated_metrics,
        'contributions': contributions,
    }
