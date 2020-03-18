# Caedus Covid
Data Science Modeling for COVID-19 Pandemic

## Getting Started
### Setup
Install development dependencies:
- [Docker](https://docs.docker.com/docker-for-mac/install/)
- [Poetry](https://python-poetry.org/docs/#installation)
- [Yarn](https://classic.yarnpkg.com/docs/install)

### Running the project
To run both the api and frontend, run the following command in your shell of choice:
```
docker-compose up
```

The backend will be ready when you see this message:
```
api_1       | INFO:     Uvicorn running on http://0.0.0.0:80 (Press CTRL+C to quit)
```

The frontend will be ready when you see this message:
```
frontend_1  |  READY  Server listening on http://0.0.0.0:80
```

Once the app finishes launching, you can visit the following URLs:
- API: http://localhost:8000
- Frontend: http://localhost:9000

### Rebuilding the project
If you have added new dependencies and need to rebuild the docker image, run:
```
docker-compose build
```

### API
See [the api's documentation](api/README.md).

### Frontend
See [the frontend's documentation](frontend/README.md).