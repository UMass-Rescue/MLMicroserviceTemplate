import requests
import time

from fastapi.logger import logger

from server import dependency


def ping_server(server_port, model_port, model_name):
    """
    Periodically ping the server to make sure that
    it is active.
    """

    while dependency.connected and not dependency.shutdown:
        try:
            r = requests.get('http://host.docker.internal:' + str(server_port) + '/')
            r.raise_for_status()
            time.sleep(dependency.WAIT_TIME)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            dependency.connected = False
            logger.debug("Server " + model_name + " is not responsive. Retry registering...")
    if not dependency.shutdown:
        register_model_to_server(server_port, model_port, model_name)


def register_model_to_server(server_port, model_port, model_name):
    """
    Send notification to the server with the model name and port to register the microservice
    It retries until a connection with the server is established
    """
    while not dependency.connected and not dependency.shutdown:
        try:
            r = requests.post('http://host.docker.internal:' + str(server_port) + '/model/register',
                              json={"modelName": model_name, "modelPort": model_port})
            r.raise_for_status()
            dependency.connected = True
            logger.debug('Registering to server succeeds.')
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            logger.debug('Registering to server fails. Retry in ' + str(dependency.WAIT_TIME) + ' seconds')
            time.sleep(dependency.WAIT_TIME)
            continue
    if not dependency.shutdown:
        ping_server(server_port, model_port, model_name)
