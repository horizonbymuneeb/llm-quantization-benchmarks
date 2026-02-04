"""Synthetic datasets for the benchmark, generated with numpy only.

No dataset downloads and no network: every split is produced by a seeded
generator, which means a report is reproducible from its config file alone.

Three generators are provided because they stress different parts of the
classifier.  ``clusters`` is separable Gaussians and is the main workload.
``moons`` has a curved decision boundary, so quantisation error in the first
layer shows up as churn rather than as an accuracy drop.  ``ring`` is
concentric and deliberately *hard* near the boundary, which is where a
compressed model changes its mind most often.
"""
