echo "Begin installation"

conda create -n intragen python=3.8 -y

conda activate intragen

pip install -e .

pip install setuptools==69.5.1

pip install -e ".[train]"

pip install flash-attn --no-build-isolation

pip install -e '.[dev]'

echo "Installation Finished!"

