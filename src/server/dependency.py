from concurrent.futures.thread import ThreadPoolExecutor

from pydantic import BaseSettings


class Settings(BaseSettings):
    ready_to_predict = False


model_settings = Settings()


class PredictionException(Exception):
    pass


connected = False
shutdown = False
pool = ThreadPoolExecutor(10)
WAIT_TIME = 10
