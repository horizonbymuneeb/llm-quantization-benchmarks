"""End-to-end tests: run_sweep, report writing, and the CLI.

These are the slowest tests in the suite because they train a model and run the
whole configuration matrix.  The config is deliberately small (three formats,
one scheme, ten epochs) so the whole module finishes in a few seconds while
still exercising every code path a real run takes.
"""
