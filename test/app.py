import io
import json

from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
app = Flask(__name__)

DEBUG = False

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files["file"]  # get the file from the request
        img_bytes = file.read()  # convert the file into bytes
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_id':class_id, 'class_name': class_name})
 

def transform_image(image_bytes):
    """takes image data in bytes, applies the 
    series of transforms and returns a tensor.
    """
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def debug():
    if DEBUG:
        with open("./images", 'rb') as f:
            image_bytes = f.read()
            tensor = transform_image(image_bytes=image_bytes)
            print(tensor)

        # Before using imagenet_class_index dictionary, 
        # first we will convert tensor value to a string value,
        # since the keys in the imagenet_class_index dictionary are strings. 
        with open("./images", 'rb') as f:
            image_bytes = f.read()
            print(get_prediction(image_bytes=image_bytes))
        

# Make sure to pass `pretrained` as `True` to use the pretrained weights:
model = models.densenet121(pretrained=True)
# Since we are using our model only for inference, switch to `eval` mode:
model.eval()
# This file contains the mapping og ImageNet class id to ImageNet class name
imagenet_class_index = json.load(open('./imagenet_class_index.json'))


def get_prediction(image_bytes):
    """The tensor y_hat will contain the index of the predicted class id.
    """
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_index = str(y_hat.item())
    return imagenet_class_index[predicted_index]


if __name__ == '__main__':
    debug()
    app.run()