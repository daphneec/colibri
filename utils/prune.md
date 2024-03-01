# PRUNE.py
_by Lut99_

How does `prune.py` work?


## `PruneInfo` & `PruneInfoTransformer`
Everything revolves around the [`PruneInfo`](./prune.py#L50) and [`PruneInfoTransformer`](./prune.py#L12) classes.

They both carry information about a model's modules (seems only [`InvertedResidualChannels`](../models/secure_mobilenet_base.py#L333)) that are eligible for pruning.

They are essentially wrappers around a dictionary that relates a particular weight to a dictionary of arbitrary properties. I.e.,
```python
class PruneInfo:
    _info: {
        "features.5.branches.2.0.ops.2.1.1.weight": {
            "compress_masked": False,
            "penalty": 0.8,
            "per_channel_flops": 5
        }
    }
```

In vanilla HR-NAS, `PruneInfo`'s dictionary carries the following properties:
1. `compress_masked`: Some kind of flag that relates to compression (see below).
2. `penalty`: A list of penalties that the module has accumulated. Computed at [lines 143-186](./prune.py#L143-186), and depending on the [`n_macs`](./prune.py#L161-162), divided by the number of hidden channels, and some [normalization](./prune.py#L186). Thus, encodes `per_channel_flops` (see below).
3. `per_channel_flops`: The number of flops (e.g., `n_macs`) per channel for the module. Same as `penalty` (see above), but not normalized. In fact, directly copies the value from it (ergo, change `penalty` and `per_channel_flops` is automatically updated).
4. `mask`: A dynamically updated value that determines which blocks to keep (assigned at [`secure_train.py`](../secure_train.py#L529)). This relates to which weights are too small to matter; i.e., only weights that are above a certain threshold are marked with a 0, others are marked with a 1. ([`cal_mask_network_slimming_by_threshold()`](./prune.py#L276-281)).

I won't note down `PruneInfoTransformer` for now, as all configs disable that (the `use_transformer`-option in any YAML file in `configs/`).


## Usage
Then, the `PruneInfo` is created in [`get_bn_to_prune()`](./prune.py#L129) and distributed through `*_train.py` as `FLAGS._bn_to_prune` (see [line 488](../secure_train.py#L488)). It is used in calls to various other `prune.py` functions.

Every epoch, the `PruneInfo` is updated as follows:
1. The `mask`s of all weights are computed (see above; [`cal_mask_network_slimming_by_threshold()`](./prune.py#L276-281)).
2. The number of total flops for all weights are computed, including a variant where weights that are mask'ed out (i.e., their mask value is 1) are ignored ([`cal_pruned_flops()`](./prune.py#L285-299)). This is returned as a pair of:
   - the number of flops that are pruned (i.e., their mask is 1).
   - a list of: `[weight name, #total weights, #pruned weights, total flops, pruned weight flops, #pruned weights / #total weights]` lists.
3. The model is then shrinked if _either_ the number of flops that are pruned are larger-to-or-equal to some threshold (`FLAGS.model_shrink_delta_flops`), _or_ it's the last epoch. This shrinking is done in [`shrink_model()`](./secure_train.py#L38) in `*_train.py`:
   1. First, another two masks are computed: one after convolutions have been applied depth-wise ([lines 59-62](../secure_train.py#59-62)), and another for an exponential moving average (ema; [lines 64-68](../secure_train.py#L64-68)).
   2. The blocks are "compressed" based on this mask by calling [`InvertedResidualChannels.compress_by_mask()`](../models/secure_mobilenet_base.py#L537-541), which calls [`copmress_inverted_residual_channels()`](../models/secure_compress_utils.py#L183-303) in compress_utils. That just removes weights (=layers :thinking:) which are too small.


## Adapting
Therefore, the trick is as follows: we change prune's `get_bn_to_prune()` function to use seconds to compute the penalty instead of n_macs.
