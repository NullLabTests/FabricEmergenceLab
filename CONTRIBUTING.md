# Contributing to FabricEmergenceLab

Thank you for your interest! This project is an experimental research platform for studying emergent behaviors in predictive-coding agents.

## How to Contribute

### Reporting Issues
- Open a GitHub issue describing the bug or feature request
- Include logs or steps to reproduce

### Code Contributions
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Run existing experiments to verify nothing is broken:
   ```bash
   python experiments/memory_maze.py
   ```
5. Commit with a descriptive message
6. Open a pull request

### Adding a New Experiment
1. Create a `.py` file in `experiments/`
2. Import fabricpc via relative path
3. Log to `logs/` as JSONL
4. Add documentation to `docs/`

### Adding a New Metric
1. Add the computation to `experiments/memory_maze.py` (or relevant experiment)
2. Include it in the JSONL log output
3. Add analysis to `logs/analysis.py`
4. Add to the report in `scripts/generate_report.py`

## Engineering Principles

- Prefer measurable experimental infrastructure over speculative claims
- Every claim of emergence should be backed by logged evidence
- Fixed seeds for reproducibility
- Append-only JSONL logging

## Code of Conduct

Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
