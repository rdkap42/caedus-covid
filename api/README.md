# Caedus Covid API
This directory contains the API for Caedus Covid.

## Usage
### Setup
Install development dependencies:
- [Docker](https://docs.docker.com/docker-for-mac/install/)
- [Poetry](https://python-poetry.org/docs/#installation)

### Running the app
To run the API on its own, run:
```
docker-compose run api
```

### Managing dependencies
We use Poetry to manage dependencies. Installations via Poetry follow the same pattern as pip and pipenv, albeit with different commands.

| Poetry command | pip command |
| - | - |
| `poetry add <dependency>` | `pip install -r <dependency>` |
| `poetry remove <dependency>` | `pip uninstall <dependency>` |