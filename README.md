# SWE-ReBench

Evaluation framework for LLM-based software engineering agents. Built on top of the [SWE-rebench benchmark](https://arxiv.org/abs/2505.20411) (21K+ real GitHub issues).

Modern LLM benchmarks have three problems:
1. **Data contamination** - models train on test data
2. **Black-box agents** - we don't understand why they work (or don't)
3. **No causal analysis** - correlation ≠ causation

This repo addresses all three with novel research contributions.

```bash
# Install
pip install -r requirements.txt

# Run evaluation
python cli.py evaluate --model deepseek/deepseek-r1 --provider openrouter --api-key $YOUR_API_KEY --max-tasks 5

```
## File Structure

```
SWE_ReBench/
├── results/
├── agent_interface.py
├── api.py
├── cli.py
├── dataset_loader.py
├── evaluation_engine.py
├── execution_env.py
├── models.py
├── requirements.txt
├── tests.py
└── README.md   
```


## Citation

```bibtex
@software{jayswal2025swerebench,
  author = {Jayswal, Divyesh},
  title = {SWE-ReBench: Causal Analysis and Interpretability for Agent Evaluation},
  year = {2025},
  url = {https://github.com/DivyeshJayswal/SWE_ReBench}
}
```

And the original benchmark:
```bibtex
@article{badertdinov2025swerebench,
  title={SWE-rebench: An Automated Pipeline for Task Collection},
  author={Badertdinov, Ibragim and others},
  journal={arXiv:2505.20411},
  year={2025}
}
```

## Requirements

- Python 3.9+
- Docker
- 16GB RAM (32GB for full benchmark)
- GPU optional (faster but not required)

## License

MIT - see LICENSE file

---

