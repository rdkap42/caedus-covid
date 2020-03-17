FROM python:3.8 as base

ENV PYTHONPATH=${PYTHONPATH}:/app/app \
    PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PYTHONUNBUFFERED=1

WORKDIR /app

FROM base as builder

# Set build-only env variables
ENV PIP_DEFAULT_TIMEOUT=100 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=1.0.5

# Install dependencies
RUN pip install "poetry==$POETRY_VERSION"
RUN python -m venv /venv

COPY pyproject.toml poetry.lock ./
RUN poetry export -f requirements.txt | /venv/bin/pip install -r /dev/stdin

# Copy and build app
COPY ./app /app
RUN poetry build && /venv/bin/pip install dist/*.whl

FROM base as final

# Copy startup scripts
COPY docker_scripts/ /

# Make sure to add virtual environment
COPY --from=builder /venv /venv