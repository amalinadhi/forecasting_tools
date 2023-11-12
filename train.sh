#!/bin/sh
# Activate the venv
source forecasting_tool_env/bin/activate

# Run the python script
python src/data_pipeline.py
python src/data_processing.py
python src/data_modelling.py