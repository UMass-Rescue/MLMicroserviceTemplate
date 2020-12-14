from PIL import Image
import time

from fastapi.logger import logger

from server.dependency import Settings


def init():
    """
    This method will be run once on startup. You should check if the supporting files your
    model needs have been created, and if not then you should create/fetch them.
    """
    logger.debug('STARTING TO WAIT')
    time.sleep(10)
    logger.debug('DONE WAITING')
    Settings.ready_to_predict = True


def predict(image_file):
    """
    Interface method between model and server. This signature must not be
    changed and your model must be able to predict given a file-like object
    with the image as an input.
    """

    image = Image.open(image_file.name, mode='r')

    return {
        "someResultCategory": "actualResultValue",
    }
