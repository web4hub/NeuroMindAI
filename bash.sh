cat-readme [https://github.com/web4hub/NeuroMindAI.git]
# Install the repository in editable mode
python -m pip install -e .

# Run a validation training run for 100 steps
python -m training.train --steps 100

# Run the test suite quietly
pytest -q
