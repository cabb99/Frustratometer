ARG PYTHON_VERSION=3.12
FROM condaforge/miniforge3:latest

# Install OS-level deps
RUN apt-get update -qq \
    && apt-get install -qq -y --no-install-recommends \
        gcc g++ gfortran git wget hmmer \
    && rm -rf /var/lib/apt/lists/*

# Create conda env with the requested Python version
ARG PYTHON_VERSION
RUN conda create -n test -y python=${PYTHON_VERSION} && conda clean -afy
SHELL ["conda", "run", "-n", "test", "/bin/bash", "-c"]

WORKDIR /repo

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --quiet --no-cache-dir -r requirements.txt
RUN pip install --quiet --no-cache-dir pytest pytest-cov
RUN conda install -n test -y -c conda-forge pdbfixer \
    || echo "WARNING: pdbfixer install failed – pdbfixer tests will be skipped"

# On Python <3.10, old prody needs numpy<2; conda's pdbfixer may have upgraded it
RUN python -c "import sys; sys.exit(0 if sys.version_info < (3,10) else 1)" \
    && conda install -n test -y "numpy<2" \
    || true

# Install pyDCA (optional – continues on failure)
RUN pip install --quiet --no-cache-dir "git+https://github.com/cabb99/pydca.git" \
    || echo "WARNING: pydca install failed – pyDCA tests will be skipped"

# Install the package itself
COPY . .
RUN pip install --quiet --no-cache-dir --no-deps -e .
