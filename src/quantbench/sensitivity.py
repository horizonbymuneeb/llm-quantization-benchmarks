"""Per-layer sensitivity: which layers can be compressed and which cannot.

The question a practitioner actually asks is not "what happens at INT8" but
"where in the network does the damage happen".  This module answers it three
ways, because they disagree in informative ways:

1. **Weight-space sensitivity** quantises one layer at a time, keeping every
   other layer at full precision, and measures the drop in test accuracy.  It
   is cheap and it isolates the layer, but it ignores error accumulation.
2. **Activation-space sensitivity** compares the layer's output before and
   after quantisation of its weights.  It catches a layer whose weights look
   harmless but whose outputs feed a nonlinearity near a decision boundary.
3. **Cumulative sensitivity** quantises layers 0..k and reports accuracy at
   each prefix.  Error accumulation is superadditive in practice, and this is
   the curve that shows it.

All three are run on a fixed test split so the numbers are comparable between
layers.
"""
