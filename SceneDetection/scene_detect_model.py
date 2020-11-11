# using a resnet 18 model to calculate to predict the scene category, (an
# indoor/outdoor image classifier using the scene category) and the scene 
# attributes of a given input image
import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F

import os
import sys
import numpy as np
from PIL import Image

class SceneDetectionModel:

    ## Model URLs
    CSAILVISION_URL = 'https://raw.githubusercontent.com/csailvision/places365/master/'
    SCENE_ATTRIBUTE_WIDERESNET18_URL = 'http://places2.csail.mit.edu/models_places365/'
    WIDERESNET18_TAR_URL = 'http://places2.csail.mit.edu/models_places365/'

    ## Supporting files names
    CATEGORIES_FILE_NAME = 'categories_places365.txt'
    INDOOR_OUTDOOR_PLACES_FILE_NAME = 'IO_places365.txt'
    LABELS_SUNATTRIBUTE_FILE_NAME = 'labels_sunattribute.txt'
    WIDERESNET18_FILE_NAME = 'wideresnet.py'
    WIDERESNET18_SCENE_ATTRIBUTES_FILE_NAME = 'W_sceneattribute_wideresnet18.npy'
    WIDERESNET18_TAR_FILE_NAME = 'wideresnet18_places365.pth.tar'
    MODEL_DIRECTORY = 'SceneDetection'

    ## Default constants
    IMG_WIDTH = 224
    IMG_HEIGHT = 224

    def __init__(self):
        self.classes = self.download_classes()
        self.labels_indoor_outdoor = self.download_labels_indoor_outdoor()
        self.labels_attribute = self.download_labels_attributes()
        self.W_attribute = self.download_wideresnet18_attributes()
        self.download_model()
        self.input_img = None

    # function to ensure presence of the list of scene categories
    def download_classes(self):
        # fetching the list of scene categories if not already present
        if not os.access(os.path.join(self.MODEL_DIRECTORY, self.CATEGORIES_FILE_NAME), os.W_OK):
            os.system('wget ' + self.CSAILVISION_URL + self.CATEGORIES_FILE_NAME + ' -P ' + self.MODEL_DIRECTORY)

        # listing all the scene categories in a tuple
        classes = list()
        with open(os.path.join(self.MODEL_DIRECTORY, self.CATEGORIES_FILE_NAME)) as class_file:
            for line in class_file:
                classes.append(line.strip().split(' ')[0][3:])
        classes = tuple(classes)
        return classes

    # function to ensure presence of the lookup table of scene category 
    # classifying as indoor or outdoor scene
    def download_labels_indoor_outdoor(self):
        # fetching the lookup table given an input scene to check if its indoor or
        # outdoor 
        if not os.access(os.path.join(self.MODEL_DIRECTORY, self.INDOOR_OUTDOOR_PLACES_FILE_NAME), os.W_OK):
            os.system('wget ' + self.CSAILVISION_URL + self.INDOOR_OUTDOOR_PLACES_FILE_NAME + ' -P ' + self.MODEL_DIRECTORY)

        # listing the table in an array
        with open(os.path.join(self.MODEL_DIRECTORY, self.INDOOR_OUTDOOR_PLACES_FILE_NAME)) as f:
            lines = f.readlines()
            labels_indoor_outdoor = []
            for line in lines:
                items = line.rstrip().split()
                labels_indoor_outdoor.append(int(items[-1]) -1) # 0 is indoor, 1 is outdoor
        labels_indoor_outdoor = np.array(labels_indoor_outdoor)
        return labels_indoor_outdoor
    
    # function to ensure presence of the list of scene attributes
    def download_labels_attributes(self):
        # fetching scene attributes if not already present
        if not os.access(os.path.join(self.MODEL_DIRECTORY, self.LABELS_SUNATTRIBUTE_FILE_NAME), os.W_OK):
            os.system('wget ' + self.CSAILVISION_URL + self.LABELS_SUNATTRIBUTE_FILE_NAME + ' -P ' + self.MODEL_DIRECTORY)
        labels_attribute = list()
        with open(os.path.join(self.MODEL_DIRECTORY, self.LABELS_SUNATTRIBUTE_FILE_NAME)) as f:
            lines = f.readlines()
            labels_attribute = [item.rstrip() for item in lines]
        return labels_attribute

    # function to ensure presence of the list of wideresnet model scene attributes
    def download_wideresnet18_attributes(self):
        if not os.access(os.path.join(self.MODEL_DIRECTORY, self.WIDERESNET18_SCENE_ATTRIBUTES_FILE_NAME), os.W_OK):
            os.system('wget ' + self.SCENE_ATTRIBUTE_WIDERESNET18_URL + self.WIDERESNET18_SCENE_ATTRIBUTES_FILE_NAME + ' -P ' + self.MODEL_DIRECTORY)
        W_attribute = np.load(os.path.join(self.MODEL_DIRECTORY, self.WIDERESNET18_SCENE_ATTRIBUTES_FILE_NAME))
        return W_attribute

    # transform the image as required by the resnet model
    def returnTF(self):
        tf = trn.Compose([
            trn.Resize((self.IMG_WIDTH, self.IMG_HEIGHT)),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return tf

    # download wideresnet18 model
    def download_model(self):
        # fetch the pretrained weights of the model if not already present
        if not os.access(os.path.join(self.MODEL_DIRECTORY, self.WIDERESNET18_TAR_FILE_NAME), os.W_OK):
            os.system('wget ' + self.WIDERESNET18_TAR_URL + self.WIDERESNET18_TAR_FILE_NAME + ' -P ' + self.MODEL_DIRECTORY)
            os.system('wget ' + self.CSAILVISION_URL + self.WIDERESNET18_FILE_NAME + ' -P ' + self.MODEL_DIRECTORY)

    # fetch and load the model, model specified inside the function itself and can
    # be modified to load a different model
    def load_model(self, features_blobs):
        # create feature list given module, input and output
        def hook_feature(module, input, output):
            features_blobs.append(np.squeeze(output.data.cpu().numpy()))

        # imported here since this model file may not be present in the beginning
        # and has been downloaded right before this
        # makes the model ready to run the forward pass
        from SceneDetection import wideresnet
        model = wideresnet.resnet18(num_classes=365)
        checkpoint = torch.load(os.path.join(self.MODEL_DIRECTORY, self.WIDERESNET18_TAR_FILE_NAME), map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
        model.eval()

        # hook the feature extractor
        features_names = ['layer4','avgpool'] # this is the last conv layer of the resnet
        for name in features_names:
            model._modules.get(name).register_forward_hook(hook_feature)
        return model

    # method to load a single image into the model for prediction
    def load_image(self, image_file_name):
        tf = self.returnTF() 
        img = Image.open(image_file_name)

        if img.mode != 'RGB':        
            img = img.convert("RGB")
        self.input_img = V(tf(img).unsqueeze(0))

    def forward_pass(self):
        # Common list required for hooking features later in the model
        features_blobs = []

        # load the model using all helper functions defined before this
        model = self.load_model(features_blobs)

        # get the softmax weight
        params = list(model.parameters())
        weight_softmax = params[-2].data.numpy()
        weight_softmax[weight_softmax<0] = 0

        # forward pass
        logit = model.forward(self.input_img)
        h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        probs = probs.numpy()
        idx = idx.numpy()
        return idx, probs, features_blobs

    def get_indoor_outdoor_prediction_from_forward_pass(self, idx):
        # output the indoor/outdoor prediction
        io_image = np.mean(self.labels_indoor_outdoor[idx[:10]]) # vote for the indoor or outdoor
        return 'indoor' if io_image < 0.5 else 'outdoor'

    def get_scene_categories_from_forward_pass(self, idx, probs):
        # output the scene categories
        scene_categories_dict = {}
        for i in range(0, 5):    
            scene_categories_dict[self.classes[idx[i]]] = str(probs[i])
        scene_categories_prob = {k: v for k, v in sorted(scene_categories_dict.items(), key=lambda item: item[1], reverse=True)}
        return scene_categories_prob
    
    def get_scene_attributes_from_forward_pass(self, features_blobs):
        # output the scene attributes
        responses_attribute = self.W_attribute.dot(features_blobs[1])
        idx_a = np.argsort(responses_attribute)
        return [self.labels_attribute[idx_a[i]] for i in range(-1,-10,-1)]

    # method to predict if the scene is indoor or outdoor
    def predict_environment_type(self):
        idx, _, _ = self.forward_pass()
        return self.get_indoor_outdoor_prediction_from_forward_pass(idx)

    # method to predict the scene's categories
    def predict_categories(self):
        idx, probs, _ = self.forward_pass()
        return self.get_scene_categories_from_forward_pass(idx, probs)

    # method to predict the scene's attributes
    def predict_scene_attributes(self):
        _, _, features_blobs = self.forward_pass()
        return self.get_scene_attributes_from_forward_pass(features_blobs)

    # the main function which given an input image predicts the scene categories,
    # scene attributes and an indoor or outdoor image using the output of the 
    # above two
    def predict_scene(self):
        idx, probs, features_blobs = self.forward_pass()
        scene_categories_prob = self.get_scene_categories_from_forward_pass(idx, probs)
        env_type = self.get_indoor_outdoor_prediction_from_forward_pass(idx)
        scene_attributes = self.get_scene_attributes_from_forward_pass(features_blobs)
        return {"category_results": scene_categories_prob, "attributes_result": scene_attributes, "environment": env_type}
