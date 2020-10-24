import asyncio
from PIL import Image
from SceneDetection.scene_detect_model import SceneDetectionModel

model = None

async def init():
    """
    This method will be run once on startup. You should check if the supporting files your
    model needs have been created, and if not then you should create/fetch them.
    """
    await asyncio.sleep(2)
    global model

    print('Loading SceneDetection model')
    model = SceneDetectionModel()

def predict(image_file):
    """
    Interface method between model and server. This signature must not be
    changed and your model must be able to predict given a file-like object
    with the image as an input.
    """
    global model
    if model == None:
        raise RuntimeError("SceneDetection model is not loaded properly")
    
    model.load_image(image_file.name)
    scene_detect_result = model.predict_scene()

    return {
        "Category Confidence": scene_detect_result['category_results'],
        "Scene Attributes": scene_detect_result['attributes_result'],
        "Environment Type": scene_detect_result['environment']
    }
