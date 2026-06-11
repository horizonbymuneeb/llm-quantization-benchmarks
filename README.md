# llm-quantization-benchmarks

A benchmark harness for the question that matters when you compress a model:
**what changed about its behaviour, not just its accuracy.** Post-training
quantisation is usually reported as a single accuracy delta, which hides the
case where a model keeps its accuracy while changing its answer on a large
fraction of inputs. This project measures that directly — per-input decision
churn, confidence rank correlation, margin shrinkage, and per-layer sensitivity
— on a small numpy MLP trained from a fixed seed, so the whole thing runs on a
CPU with no framework and no downloads.

The harness simulates FP32, BF16, INT8 and INT4 storage; supports symmetric and
asymmetric schemes at per-tensor and per-channel granularity; implements four
calibration methods (min-max, percentile, entropy/KL, MSE); and reports the
accuracy delta, churn rate, Spearman and Kendall rank agreement, expected
calibration error, and per-layer sensitivity ranking.

## What is in the box

| Module | Job |
|---|---|
| `src/quantbench/dtypes.py` | Storage-format simulation: BF16 rounding, FP16, INT8/INT4 grids, byte accounting and compression ratios. |
| `src/quantbench/config.py` | Typed, validated config objects loaded from `config.yaml`; unknown keys are an error, not a silent default. |
| `src/quantbench/data.py` | Seeded synthetic datasets (Gaussian clusters, moons, rings) and a stratified train/test split. |
| `src/quantbench/model.py` | The numpy MLP under test: He/Xavier init, forward with recorded intermediates, hand-written backprop, Adam. |
| `src/quantbench/calibration.py` | Activation range selection: min-max, percentile, entropy/KL search, and MSE-optimal clipping. |
| `src/quantbench/quantize.py` | Weight quantisation: symmetric/asymmetric, per-tensor/per-channel, nearest or stochastic rounding. |
| `src/quantbench/metrics.py` | Behavioural metrics: accuracy delta, decision churn, confidence deltas, margin shift, Spearman/Kendall, ECE. |
| `src/quantbench/sensitivity.py` | Per-layer, cumulative, and mixed-precision sensitivity analysis. |
| `src/quantbench/sweep.py` | The experiment driver: train once, evaluate the whole quantisation matrix, assemble rows. |
| `src/quantbench/report.py` | JSON report assembly, strict-JSON sanitisation, report diffing, CSV export. |
| `src/quantbench/cli.py` | Command-line interface: `sweep`, `inspect`, `compare`, `calibrate`. |

## Quickstart

```bash
python -m pip install -r requirements.txt

# Full sweep: trains the MLP, evaluates every configuration, writes a report.
PYTHONPATH=src python -m quantbench.cli sweep --config config.yaml

# Same thing through make, plus a flat CSV of the rows.
make sweep

# One configuration, in detail.
PYTHONPATH=src python -m quantbench.cli inspect --config config.yaml --format int4

# Print the calibrated activation ranges per layer.
PYTHONPATH=src python -m quantbench.cli calibrate --config config.yaml

# Diff two reports.
PYTHONPATH=src python -m quantbench.cli compare reports/old.json reports/new.json

make test
```

`make sweep` writes `reports/sweep_report.json` and prints a digest:

```
reference accuracy: 0.697222
configurations: 10
  int8-symmetric-per_channel         x4.00   acc=0.6972  delta=+0.0000 flips=0.0056
  int4-symmetric-per_tensor          x8.00   acc=0.6903  delta=-0.0069 flips=0.0833
  int4-asymmetric-per_channel        x8.00   acc=0.7000  delta=+0.0028 flips=0.0639
churn outliers (3):
  int4-asymmetric-per_channel: flip_rate=0.0639 vs accuracy_delta=0.0028
```

That last block is the reason the project exists: `int4-asymmetric-per_channel`
loses *nothing* in accuracy (it is marginally better, which is noise at this
test-set size) while changing the prediction on 6.4% of inputs. An accuracy-only
report would call that configuration free.

## Design notes

**Quantised layers stay in float32; the benchmark measures information loss, not
storage.** Every quantiser returns a dequantised `float32` tensor. This means a
compressed model runs through exactly the same numpy forward pass as the
reference, and the only difference between two rows of the report is the numbers
the layers see. Packing is simulated separately, by `storage_bytes` and
`compression_ratio`, when a compression factor is reported. The alternative —
packing to real `int8` arrays and writing kernels for them — would have measured
my matmul rather than the format.

**The entropy calibrator scores a saturating assignment, not a truncation.** The
first version of `entropy_calibration` truncated the histogram at each candidate
threshold and compared the surviving mass against the reference. It always chose
the tightest allowed clip, because a small threshold gave each surviving bin its
own code and therefore looked lossless — the clipped tail simply vanished from
the comparison. The fix models what hardware actually does: the retained range
`[0, cut)` is divided into `n_levels` equal steps (so a tighter `cut` means finer
resolution in the dense region) and every bin at or beyond `cut` saturates onto
the last code. With saturation in the objective the search has a real trade-off
and finds a minimum near four standard deviations on a Gaussian with an outlier
tail. `tests/test_calibration.py` pins that behaviour.

**Per-channel scales are derived along axis 0 because every weight matrix is
`(out_features, in_features)`.** One scale per output unit is the layout
hardware kernels expect, and it is also what makes the per-tensor/per-channel
comparison meaningful: the tensor-level scale has to cover the row with the
largest dynamic range, so rows with small weights collapse toward zero. A test
constructs exactly that case (one row spanning ±50, one spanning ±0.01) and
asserts that per-channel keeps the small row's error under 1e-3 while per-tensor
does not.

**Biases stay in float unless `weight_layers_only` is false.** Bias quantisation
is a separate design axis, and including it by default would confound the
attribution of damage to *weight* precision. The switch exists so the
alternative can be measured, but the default isolates the thing under test.

**The reference model is trained once and every configuration is measured
against those weights.** Training is expensive relative to evaluating a
quantised copy, and more importantly it makes report rows comparable: any
difference between two rows is caused by the quantisation settings, not by a
different training run. `test_run_sweep_is_deterministic` asserts that two
sweeps of the same config produce identical rows.

**Sensitivity is ranked by output error, not by accuracy drop.** Test accuracy
moves in steps of `1/n_test`, so with a few hundred test rows many layers tie
exactly on accuracy drop and the ranking becomes arbitrary. The relative L2
change in a layer's output is continuous and ranks stably; accuracy drop is still
reported next to it.

**NaN is converted to `null` in reports.** Python's `json` emits bare `NaN`,
which is not valid JSON and is rejected by several editors and every strict
parser. `report.py` sanitises the tree before writing and the writer is called
with `allow_nan=False` so a missed value fails loudly instead of producing an
invalid file.

## Tests

267 tests, run with `python -m pytest -q`. They assert real behaviour:

- **Exact values.** BF16 round-half-to-even on hand-picked bit patterns; the
  analytic KL divergence of `(1,0)` against `(0.5,0.5)` = 1 bit; hand-computed
  cross-entropy; Kendall tau of a three-element example counted by hand.
- **Gradients.** `backward` is checked against a finite-difference estimate of
  the loss for both a weight and a bias entry.
- **Numerical tolerances.** The BF16 relative error bound of 2⁻⁸; the He
  initialisation standard deviation of √(2/fan_in); the half-scale bound on
  nearest-rounding error; monotone error ordering FP32 < BF16 < INT8 < INT4.
- **Shapes and ranges.** Every quantiser, cache, confusion matrix and report row.
- **Error paths.** Invalid axes, degenerate scales, unknown formats/schemes/
  methods/round modes, missing config files, unknown YAML keys, mismatched
  shapes and lengths, absent report files, bad percentiles, out-of-range labels.
- **The headline behaviour.** A test constructs two prediction sets with
  identical accuracy but different answers and asserts the churn metric reports
  it while the accuracy delta does not.
- **End to end.** `run_sweep` on a small config, report round-trip through strict
  JSON, report diffing, CSV export, and all four CLI subcommands including their
  failure exits.

## Limitations

- The model under test is a three-layer numpy MLP on synthetic data. It is a
  vehicle for the measurement machinery, not a claim about any real network. The
  absolute churn numbers here do not transfer to a transformer.
- Quantised tensors are stored as `float32`. Reported compression ratios are
  computed from `storage_bytes` and are not achieved in memory.
- Per-channel granularity means axis 0 only. There is no per-group (block-wise)
  quantisation, no mixed INT4/INT8 within a layer, and no KV-cache quantisation.
- Biases are excluded by default, and the output layer is quantised like any
  other layer. Selective layer-skipping is only available through the
  mixed-precision recommendation, which keeps the top-k most sensitive layers at
  BF16 and quantises the rest — it is a heuristic, not a searched optimum.
- Activation quantisation is calibrated and reported, but the forward pass is
  not simulated with quantised activations; only weights are quantised during
  evaluation. The calibration module is exercised and tested, and the numbers it
  produces are what a real activation-quantising deployment would start from.
- The dataset generators are separable enough that a three-layer MLP reaches
  roughly 70% accuracy; that is intentional (a hard problem shows churn), but it
  means the accuracy deltas are noisy at this test-set size.
- No GPU path, no real INT8 kernels, and no support for models loaded from disk.
