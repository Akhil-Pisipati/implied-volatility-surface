# Contributing

Thanks for helping improve the Indian Options Implied-Volatility Surface.

1. Make focused changes and describe the motivation and data-source assumptions in the pull request.
2. Install development dependencies with `python -m pip install -e ".[dev]"`.
3. Run `pytest -q`, `ruff check .`, and `black --check .` before submitting.
4. Add offline tests for provider parsing, India-market timing, quote validation, and numerical behavior. CI must not call broker or NSE APIs.
5. Fixtures must be licensed/authorized and traceable. The committed NSE-shaped JSON files are explicitly synthetic examples, not captured market data. Do not add large raw exchange dumps.
6. Never commit `.env`, API tokens, cached private data, or account details.

Market-data behavior and access limits vary by provider. Document assumptions and limitations in code and the README. Numerical arbitrage checks should include the tested domain and tolerances. Check current NSE data-use terms before collecting, storing, or redistributing exchange content.
