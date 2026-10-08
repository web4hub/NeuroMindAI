cat-readme [https://github.com/web4hub/NeuroMindAI.git]
# Install the repository in editable mode
python -m pip install -e .

# Run a validation training run for 100 steps
python -m training.train --steps 100

# Run the test suite quietly
pytest -q
PYTHONPATH=src python3 -m bayes_irt_gsm8k.suite \
  --data-dir . \
  --item-mode file \
  --items-file rigorous_residual_outputs/perturbation_candidate_items.csv \
  --sensitivity standard \
  --steps 1000 \
  --batch-size 50000 \
  --holdout-fraction 0.1 \
  --seed 123 \
  --output-dir bayes_irt_suite_outputs \
  --top-n 1000
