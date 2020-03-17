import os
from importlib import import_module

from fastapi import FastAPI
from fastapi_sqlalchemy import DBSessionMiddleware

# Import app config
config_module = import_module(os.getenv('APP_CONFIG'))
config = config_module.Config()

# Create app
app = FastAPI()

# Add middleware
app.add_middleware(DBSessionMiddleware, db_url=config.DATABASE_URL)