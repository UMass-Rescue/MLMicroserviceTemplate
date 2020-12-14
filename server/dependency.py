from pydantic import BaseSettings


class Settings(BaseSettings):
    ready_to_predict = False


class PredictionException(Exception):
    pass