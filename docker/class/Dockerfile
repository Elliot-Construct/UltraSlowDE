FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    make \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install project + modeling dependencies (including CLASS/CAMB backends)
COPY pyproject.toml README.md /workspace/
COPY src /workspace/src
RUN python -m pip install --upgrade pip && \
    python -m pip install -e . && \
    python -m pip install camb classy tqdm matplotlib corner

CMD ["bash"]