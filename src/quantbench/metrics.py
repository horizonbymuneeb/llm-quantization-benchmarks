"""Behavioural metrics: accuracy, decision churn, and confidence rank agreement.

Aggregate accuracy is a lossy summary of what compression did.  A model can keep
its accuracy to three decimal places while changing its mind on a sixth of the
inputs, and those flips are what a user of the model actually experiences.  This
module therefore reports three families of metric:

* **accuracy** — the classic aggregate, plus the per-class breakdown so a
  regression concentrated in one class is visible.
* **decision churn** — per-input comparison of the compressed model's argmax
  against the reference argmax: flips, net flips, confidence deltas, and the
  margin change for the inputs that flipped.
* **rank agreement** — Spearman and Kendall correlation between the reference
  and compressed confidence orderings.  Two models can agree on every argmax
  while ordering the rest of the distribution completely differently, and the
  rank metrics catch that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
