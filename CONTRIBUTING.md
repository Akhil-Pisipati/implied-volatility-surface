# Contributing

Thanks for helping improve the Implied Volatility Surface Builder.

1. Create a focused branch and describe the motivation in your pull request.
2. Install development dependencies with `pip install -e '.[dev]'`.
3. Run `pytest -q` and `ruff check .` before submitting.
4. Add tests for numerical behavior, quote validation, and regression fixes.
5. Do not commit API credentials, cached private data, or environment files.

Market-data behavior varies by provider. Document assumptions and limitations in code and the README. Numerical arbitrage checks should include the tested domain and tolerances.
