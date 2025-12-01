@echo off
echo Running Research Lab Tests...
echo ============================

python -m pytest tests/ -v --ignore=venv --ignore=data --ignore=research_lab/venv
