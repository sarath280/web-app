import os
import json
import urllib.request
import h5py
import numpy as np
import pickle

import keras 
from keras.models import Sequential,load_model,model_from_json
from keras.applications.vgg16 import VGG16
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img
from keras.utils.data_utils import get_file

# from azure.storage.blob import BlockBlobService
# container_name = 'akash'

# block_blob_service = BlockBlobService(account_name='code', account_key='JkBDjAZRJJSHibDT10kS1JlVo9fMsnhua1qF0lXoNmn4n1DRHrtVUqCng3HDIv4IJPhuMGOPLy5zdIsDCdSydw==')

location = './static/models/'

# Load models and support
first_gate = VGG16(weights='imagenet')

with open(location+'second_gate_model_json.json','r') as f:
    second_gate_json = f.read()
second_gate = model_from_json(second_gate_json)
second_gate.load_weights(location+'second_gate_model_weights.h5')

with open(location+'location_model_json.json','r') as f:
    location_model_json = f.read()
location_model = model_from_json(location_model_json)
location_model.load_weights(location+'location_model_weights.h5')

with open(location+'severity_model_json.json','r') as f:
    severity_model_json = f.read()
severity_model = model_from_json(severity_model_json)
severity_model.load_weights(location+'severity_model_weights.h5')


with open(location+'vgg16_cat_list.pk','rb') as f:
    cat_list = pickle.load(f)

print('cat-list loaded')

# from Keras GitHub  
CLASS_INDEX = None
CLASS_INDEX_PATH = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'

def get_predictions(preds, top=5):
    global CLASS_INDEX
    if len(preds.shape) != 2 or preds.shape[1] != 1000:
        raise ValueError('`decode_predictions` expects '
                         'a batch of predictions '
                         '(i.e. a 2D array of shape (samples, 1000)). '
                         'Found array with shape: ' + str(preds.shape))
    if CLASS_INDEX is None:
        fpath = get_file('imagenet_class_index.json',
                         CLASS_INDEX_PATH,
                         cache_subdir='models')
        CLASS_INDEX = json.load(open(fpath))
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    return results

def prepare_img_244(img_path):
    #urllib.request.urlretrieve(img_url,'save.jpg')
    img = load_img(img_path,target_size=(224,224))
    x = img_to_array(img)
    x = np.expand_dims(x,axis=0)
    x = preprocess_input(x)
    return x

def car_categories_gate(img244,model):
    print("Validating that this is a picture of your car...")
    out = model.predict(img244)
    top = get_predictions(out,top=5)
    for j in top[0]:
        if j[0:2] in cat_list:
            # print j[0:2]
            return True # "Validation complete - proceed to damage evaluation"
    return False # "Are you sure this is a picture of your car? 
                 # Please take another picture (try a different angle or lighting) and try again."

def prepare_img_256(img_path):
    img = load_img(img_path,target_size=(256,256))
    x = img_to_array(img)
    x = x.reshape((1,)+x.shape)  
    return x

def car_damage_gate(img_256,model):
    print("Validating that damage exists...")
    pred = model.predict(img_256)
    if pred[0][0] <=.5:
        return True # print "Validation complete - proceed to location and severity determination"
    else:
        return False
        # print "Are you sure that your car is damaged? Please submit another picture of the damage."
        # print "Hint: Try zooming in/out, using a different angle or different lighting"

def location_assessment(img_256,model):
    print("Determining location of damage...")
    pred = model.predict(img_256)
    pred_label = np.argmax(pred,axis=1)
    d = {0: 'Front', 1: 'Rear', 2: 'Side'}
    for key in d.keys():
        if pred_label[0] == key:
        	return d[key]
            #print("Assessment: {} damage to vehicle".format(d[key]))
    #print("Location assessment complete.")

def severity_assessment(img_256, model):
    print("Determining severity of damage...")
    pred = model.predict(img_256)
    pred_label = np.argmax(pred, axis=1)
    d = {0: 'Minor', 1: 'Moderate', 2: 'Severe'}
    for key in d.keys():
        if pred_label[0] == key:
            return d[key]
            #print("Assessment: {} damage to vehicle".format(d[key]))
    #print("Severity assessment complete.")

# load models
def engine(img_path):
    img_244 = prepare_img_244(img_path)
    g1 = car_categories_gate(img_244,first_gate)
    if g1 is False:
        result = {'gate1': 'Car validation check: ', 
        'gate1_result': 0, 
        'gate1_message': {0: 'Are you sure this is a picture of your car? Please retry your submission.', 
        1: 'Hint: Try zooming in/out, using a different angle or different lighting'},
        'gate2': None,
        'gate2_result': None,
        'gate2_message': {0: None, 1: None},
        'location': None,
        'severity': None,
        'final': 'Damage assessment unsuccessful!'}
        return result

    img_256 = prepare_img_256(img_path)
    g2 = car_damage_gate(img_256,second_gate)
    if g2 is False:
        result = {'gate1': 'Car validation check: ', 
        'gate1_result': 1, 
        'gate1_message': {0: None, 1: None},
        'gate2': 'Damage presence check: ',
        'gate2_result': 0,
        'gate2_message': {0: 'Are you sure that your car is damaged? Please retry your submission.',
        1: 'Hint: Try zooming in/out, using a different angle or different lighting.'},
        'location': None,
        'severity': None,
        'final': 'Damage assessment unsuccessful!'}
        return result

    x = location_assessment(img_256,location_model)
    y = severity_assessment(img_256,severity_model)

    result = {'gate1': 'Car validation check: ', 
	'gate1_result': 1, 
	'gate1_message': {0: None, 1: None},
	'gate2': 'Damage presence check: ',
	'gate2_result': 1,
	'gate2_message': {0: None, 1: None},
	'location': x,
	'severity': y,
	'final': 'Damage assessment complete!'}
    return result
