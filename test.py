from PIL import ImageDraw
from PIL import Image
import json

block = 0
index = 1
image_width = 224
image_height = 224

# image = Image.open('./data/images/%02d/%02d-%03d.png' % (block, block, index))
image = Image.open('./data/images/%03d/%03d-%06d.png' % (block, block, index))
anno = []
# open('data/annotations/%02d.json' % block)
with open('data/annotations/%06d.json' % block) as f:
    for line in f:
        anno.append(json.loads(line))
draw = ImageDraw.Draw(image)
bbox =  anno[index]['bbox']
shape = [(bbox[0], image_height - bbox[1] - bbox[3]), (bbox[0] + bbox[2], image_height - bbox[1])] 
draw.rectangle(shape, outline ="red") 
image.show()