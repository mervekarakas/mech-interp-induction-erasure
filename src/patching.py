import torch
from transformer_lens import HookedTransformer
from typing import Callable


def get_clean_cache(model: HookedTransformer, clean_tokens: torch.Tensor):
    """
    Run the model on clean tokens and cache all activations.
    We'll use these cached activations to "patch" into corrupted runs.

    Returns:
        clean_logits: model output on clean tokens
        clean_cache: dictionary mapping hook names to activation tensors
    """
    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    return clean_logits, clean_cache


def patch_resid_pre_at_position(
    model: HookedTransformer,
    corrupted_tokens: torch.Tensor,
    clean_cache,
    layer: int,
    position: int,
    metric_fn: Callable,
) -> float:
    """
    Run the model on corrupted tokens, but at one specific (layer, position),
    replace the residual stream activation with the clean cached value.

    The "residual stream" (resid_pre) at layer L is the sum of all outputs
    from layers 0..L-1 plus the original embedding. It's the main information
    highway through the transformer — everything reads from and writes to it.

    Patching at (layer, position) answers: "Does the information at this
    specific location matter for the behavior?"

    Args:
        model: the transformer
        corrupted_tokens: input with corrupted first half
        clean_cache: cached activations from the clean run
        layer: which layer's resid_pre to patch
        position: which token position to patch
        metric_fn: function(logits) -> scalar, our evaluation metric

    Returns:
        metric value after patching
    """
    # The hook name for the residual stream at the input to layer L
    hook_name = f"blocks.{layer}.hook_resid_pre"

    def patch_hook(activation, hook):
        # activation shape: [batch, seq_len, d_model]
        # Replace just one position with the clean version
        activation[:, position, :] = clean_cache[hook_name][:, position, :]
        return activation

    # Run with the hook active — model processes corrupted tokens but
    # at (layer, position) it sees the clean activation
    patched_logits = model.run_with_hooks(
        corrupted_tokens,
        fwd_hooks=[(hook_name, patch_hook)],
    )

    return metric_fn(patched_logits).item()


def compute_patching_heatmap(
    model: HookedTransformer,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    metric_fn: Callable,
) -> tuple[torch.Tensor, float, float]:
    """
    Compute the full (layer x position) patching heatmap.

    For each (layer, position), we patch clean activations into the corrupted
    run and measure how much the metric recovers.

    We normalize to a "recovery fraction":
        0.0 = patching here didn't help at all (metric stays at corrupted level)
        1.0 = patching here fully restored the clean metric

    Args:
        model: the transformer
        clean_tokens: clean repeated-token sequences
        corrupted_tokens: sequences with corrupted first half
        metric_fn: function(logits) -> scalar

    Returns:
        heatmap: tensor [n_layers, seq_len] of recovery fractions
        clean_metric: metric on clean tokens
        corrupted_metric: metric on corrupted tokens
    """
    n_layers = model.cfg.n_layers
    seq_len = clean_tokens.shape[1]

    # Step 1: get clean cache and clean metric
    clean_logits, clean_cache = get_clean_cache(model, clean_tokens)
    clean_metric = metric_fn(clean_logits).item()

    # Step 2: get corrupted metric (no patching)
    with torch.no_grad():
        corrupted_logits = model(corrupted_tokens)
    corrupted_metric = metric_fn(corrupted_logits).item()

    print(f"Clean metric:     {clean_metric:.3f}")
    print(f"Corrupted metric: {corrupted_metric:.3f}")
    print(f"Gap to recover:   {clean_metric - corrupted_metric:.3f}")
    print(f"\nPatching {n_layers} layers x {seq_len} positions...")

    # Step 3: patch each (layer, position) and record recovery
    heatmap = torch.zeros(n_layers, seq_len)

    for layer in range(n_layers):
        for pos in range(seq_len):
            patched_metric = patch_resid_pre_at_position(
                model, corrupted_tokens, clean_cache, layer, pos, metric_fn
            )
            # Normalize: how much of the gap did patching recover?
            if clean_metric - corrupted_metric != 0:
                heatmap[layer, pos] = (patched_metric - corrupted_metric) / (
                    clean_metric - corrupted_metric
                )
            else:
                heatmap[layer, pos] = 0.0

        print(f"  Layer {layer:2d} done — max recovery: {heatmap[layer].max():.3f}")

    return heatmap, clean_metric, corrupted_metric


def compute_head_patching_heatmap(
    model: HookedTransformer,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    metric_fn: Callable,
) -> tuple[torch.Tensor, float, float]:
    """
    Patch each attention head's output (across ALL positions at once)
    from the clean run into the corrupted run.

    This is more informative than position-level patching for induction,
    because induction depends on information from many positions simultaneously.

    We use the hook point "blocks.{layer}.attn.hook_z" which stores
    each head's output before the final projection, with shape
    [batch, seq_len, n_heads, d_head]. This is always cached by default.

    Returns:
        heatmap: tensor [n_layers, n_heads] of recovery fractions
        clean_metric: metric on clean tokens
        corrupted_metric: metric on corrupted tokens
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Step 1: get clean cache and metrics
    clean_logits, clean_cache = get_clean_cache(model, clean_tokens)
    clean_metric = metric_fn(clean_logits).item()

    # Debug: verify the hook names we expect actually exist in the cache
    test_hook = "blocks.0.attn.hook_z"
    assert test_hook in clean_cache, (
        f"'{test_hook}' not in cache. Available keys: "
        + str([k for k in clean_cache.keys() if 'attn' in k][:10])
    )
    print(f"Cache check OK. hook_z shape: {clean_cache[test_hook].shape}")

    with torch.no_grad():
        corrupted_logits = model(corrupted_tokens)
    corrupted_metric = metric_fn(corrupted_logits).item()
    gap = clean_metric - corrupted_metric

    print(f"Clean metric:     {clean_metric:.3f}")
    print(f"Corrupted metric: {corrupted_metric:.3f}")
    print(f"Gap to recover:   {gap:.3f}")
    print(f"\nPatching {n_layers} layers x {n_heads} heads = {n_layers * n_heads} runs...")

    heatmap = torch.zeros(n_layers, n_heads)

    for layer in range(n_layers):
        for head in range(n_heads):
            hook_name = f"blocks.{layer}.attn.hook_z"

            # Use a factory function to capture the current hook_name and head
            # (avoids Python closure-over-loop-variable bugs)
            def make_hook(cached_hook_name, h):
                def patch_hook(activation, hook):
                    # activation: [batch, seq_len, n_heads, d_head]
                    # Replace just this one head's z output (all positions)
                    activation[:, :, h, :] = clean_cache[cached_hook_name][:, :, h, :]
                    return activation
                return patch_hook

            patched_logits = model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks=[(hook_name, make_hook(hook_name, head))],
            )
            patched_metric = metric_fn(patched_logits).item()

            if gap != 0:
                heatmap[layer, head] = (patched_metric - corrupted_metric) / gap

        top_head = heatmap[layer].argmax()
        print(f"  Layer {layer:2d} done — best head: {top_head.item()} (recovery: {heatmap[layer, top_head]:.3f})")

    return heatmap, clean_metric, corrupted_metric
