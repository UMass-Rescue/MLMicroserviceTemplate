# MLMicroserviceTemplate

## Overview

This is a template application for developing a machine learning model and serving the
 results of it on a webserver. This allows the model to work with a server such as [\[This One\]](https://github.com/UMass-Rescue/PhotoAnalysisServer)

The entire application is containerized with Docker, so the entire development and 
deployment process is portable.


## Initial Setup

To run this application, there are some commands that must be run. All of these should be
done via the command line in the root directory of the project folder.

Ensure that you have Docker running on your computer before attempting these commands.


### Configure Application

When creating a model with this template, there are two options that must be configured before
you can get full functionality from the template. These options are set once per model, and will
not need to be changed by you in the future during development.

#### [Configuration] Step 1
Open the file `.env` in the root directory. Change the first line from:
`NAME=example_model` to a more descriptive name for your model.

> [**Example**]
> If your model is able to detect soda cans, the first line of your `.env` file may look like the following:
> ```text
> NAME=soda_detect
> ```

#### [Configuration] Step 2 (Optional)
This step must only be done if you are running multiple template Docker containers on the same computer.

Open the file `.env` in the root directory. Change the first line from:
`PORT=5005` to another port that you have not already used.


### Build and Run Application

Once you have configured the template, you must run these commands to build and run the model.

**As mentioned above, all of these commands must be run in the root directory of the project.**

#### [Build & Run] Quick Setup

You may run `setup/setup.sh` in your terminal to quickly set up this application. If that fails, detailed steps
are included below for manual setup. If this is successful however, you may skip all of the following steps.

#### [Build & Run] Step 0
If this is the first time you are setting up a model template, you must manually create a "volume".
This is used to track and share uploaded images.

**You only need to run this command once per computer, regardless of how many different models
 you will be running on the machine**

```cmd
docker volume create --name=photoanalysisserver_images
``` 


#### [Build & Run] Step 1
Download dependencies with Docker and build container

```cmd
docker-compose build
```
#### [Build & Run] Step 2
Start application
```cmd
docker-compose up -d
```

---

## Model Development

When working with models in this template, there are some useful commands and intricacies 
to keep in mind.

### [Dev Feature] Creating a Model
This template is designed to allow easy deployment of pre-trained machine learning models. To successfully add a model
to this template, there are two methods in the `model.py` file that must be implemented:
- `init`
- `predict`

The `init` method in `model.py` will be run once upon initial startup. This method should fetch all needed files and ensure that
the model is ready to make predictions.

The `predict` method is what is used by the template to return prediction results. This method is passed a [file-like object](https://docs.python.org/3/library/io.html)
containing the input image. The output of this method is a dictionary containing the results of the model prediction. The output should look something like this: `return {"someResultCategory": "actualResultValue"}`


You may create any number of additional helper files in this template, and they will be automatically included in the
Docker container. However, note that the signatures and return types of `init` and `predict` in `model.py` must not be changed.


### [Dev Feature] Installing Packages
When working with models, it is likely that you will need to install additional packages such as numpy.

To install additional requirements, add them to the requirements.txt file locally. Then, you can rerun the `docker-compose build` command and
the packages will now be installed.

**NOTE:** Do **NOT** connect to the container and then install packages with pip. If you do this, then the installed packages will be lost when you restart the container! 


### [Dev Command] Connect to Application's Terminal
In the initial setup, you set the name of the application. Using that name as `$NAME`, you may run
the following command to connect your terminal window to that of the Docker container. This allows you to run commands such as
`python` to debug and test your model.
```cmd
docker exec -it $NAME bash
```
> [**Example**]
> If your model is named soda_detect, run the following command to connect to the terminal for the model
> ```cmd
> docker exec -it soda_detect bash
> ```

---

## Web Server Configuration

This template serves the results of model predictions on a web API that can be easily accessed by other
applications. When a remote server wishes to use the model contained in the template, it will do the following

1. `POST /status`: This will run the `init` method in `model.py` to ensure the model is ready for predicting.
2. `GET /status`: This will ensure that the model is ready to make predictions.
3. `POST /predict`: This will create a new predicted based off of the passed image file name in the request body. 

#### Testing Model Predictions
If you wish to test the webserver results, a good tool to use is [Postman](https://postman.com). This is used for simulating API
calls and viewing the results. To do the steps above, you may do the following in Postman:

**Note:** `$PORT` in the steps below refers to the port number defined on line 2 of the `.env` file. `$NAME` in the steps 
below refers to the container name defined on the first line of the `.env` file

1. Send a `POST` request to `localhost:$PORT/status` with an empty body.
2. Send a `GET` request to `localhost:$PORT/status`. You should receive `"result": "True"` as a response.
3. Upload image file to Docker volume and `POST` the `/predict` endpoint:
    - Run the following command to transfer an image file from your local computer to the Docker image volume.
   `docker cp $LOCALPATH/$IMAGE $NAME:images/$IMAGE`
        - In this example, `$IMAGE` refers to the image file that is being transferred and `$LOCALPATH` refers to the
   path leading up to the image file (example: `$LOCALPATH=~/Downloads/images`, `$IMAGE=my_picture.png`).
   
   - Send a `POST` request to `localhost:$PORT/predict` with the following arguments key:`filename` body:`$IMAGE`.
