import time

from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware

import logging

from starlette.responses import JSONResponse

from src.model.model import predict, init
from dotenv import load_dotenv
import os

from src.server import dependency
from src.server.dependency import model_settings, PredictionException, pool, connected
from src.server.server_connection import register_model_to_server

app = FastAPI()


# Must have CORSMiddleware to enable localhost client and server
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:5057",
    "http://localhost:5000",
    "http://localhost:6379",
]

logger = logging.getLogger("api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(PredictionException)
async def prediction_exception_handler(request: Request, exc: PredictionException):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "status": 'failure',
            "detail": "Model is not ready to receive predictions."
        },
    )


@app.get("/")
async def root():
    """
    Default endpoint for testing if the server is running
    :return: Positive JSON Message
    """
    return {"MLMicroserviceTemplate is Running!"}


@app.on_event("startup")
def initial_startup():
    """
    Calls the init() method in the model and prepares the model to receive predictions. The init
    task may take a long time to complete, so the settings field ready_to_predict will be updated
    asynchronously when init() completes. This will also begin the background registration task
    to the server.

    :return: {"status": "success"} upon startup completion. No guarantee that init() is done processing.
    """
    # Run startup task async
    load_dotenv()

    # Register the model to the server in a separate thread to avoid meddling with
    # initializing the service which might be used directly by other client later on
    # We will only run the registration once the model init is complete.
    def init_model_helper():
        logger.debug('Beginning Model Initialization Process.')
        init()
        model_settings.ready_to_predict = True
        logger.debug('Finishing Model Initialization Process.')
        pool.submit(register_model_to_server, os.getenv('SERVER_PORT'), os.getenv('PORT'), os.getenv('NAME'))

    pool.submit(init_model_helper)
    return {"status": "success", 'detail': 'server startup in progress'}


@app.on_event('shutdown')
def on_shutdown():
    model_settings.ready_to_predict = False

    dependency.shutdown = True  # Send shutdown signal to threads
    pool.shutdown()  # Clear any non-processed jobs from thread queue

    return {
        'status': 'success',
        'detail': 'Deregister complete and server shutting down.',
    }


@app.get("/status")
async def check_status():
    """
    Checks the current prediction status of the model. Predictions are not able to be made
    until this method returns {"result": "True"}.

    :return: {"result": "True"} if model is ready for predictions, else {"result": "False"}
    """

    if not model_settings.ready_to_predict:
        raise PredictionException()

    return {
        'status': 'success',
        'detail': 'Model ready to receive prediction requests.'
    }


@app.post("/predict")
async def create_prediction(filename: str = ""):
    """
    Creates a new prediction using the model. This method must be called after the init() method has run
    at least once, otherwise this will fail with a HTTP Error. When given a filename, the server will create a
    file-like object of the image file and pass that to the predict() method.

    :param filename: Image file name for an image stored in shared Docker volume photoanalysisserver_images
    :return: JSON with field "result" containing the results of the model prediction.
    """

    # Ensure model is ready to receive prediction requests
    if not model_settings.ready_to_predict:
        raise PredictionException()

    # Attempt to open image file
    try:
        image_file = open('src/images/' + filename, 'r')
        image_file.close()
    except IOError:
        logger.debug('Unable to open file: ' + filename)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": 'failure',
                "detail": "Invalid file name provided: [" + filename + "]. Unable to find image on server."
            }
        )

    # Create prediction with model
    result = predict(image_file)

    return {
        'status': 'success',
        "result": result
    }



