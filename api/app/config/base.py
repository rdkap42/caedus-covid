import os

class BaseConfig:
  DATABASE_URL = os.getenv("DATABASE_URL")