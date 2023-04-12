from pprint import pprint
import shutil
import os
import cv2
from pprint import pprint

import identify_layers
import ai_utils

img_path = r'C:\Users\gufra\Desktop\Work\Projects\MultiThemed\CamNN\Datasets\test_img8.jpg'
cache_path = r'C:\Users\gufra\Desktop\Work\Projects\MultiThemed\CamNN\Scripts\cache'

if os.path.exists(cache_path):
    shutil.rmtree(cache_path)
os.mkdir(cache_path)

img = cv2.imread(img_path)
layers = identify_layers.extract_layers(img,cache_path)

text_in_layer = ai_utils.get_text(layers)

prompt = "write tensorflow/keras code for a CNN using following layers\n\n"
for i in text_in_layer:
    layer_text = i["text"]
    for t in layer_text:
        prompt+= t + " "
    prompt+="\n"

# code = ai_utils.build_code(prompt)

print(prompt)
print()
# pprint(code)