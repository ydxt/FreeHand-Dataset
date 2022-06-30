# FreeHand-Dataset

![Sample images](samples.png)

FreeHand-Dataset is synthesized hand pose images generated by Blender.

## Prerequisites

- Blender 2.80 or above

## How to use

`render_batch.py` is a Blender file to make images of hands in 224x224 PNG
format and write annotations of hand pose and bounding box. `hand.blend` is
a Blender file which has a hand 3D model. Run the following command to
make just 20 images to verify the set up. Use `--full` to make 60,000
images.

```s
$ blender hand.blend --background --python render_batch.py -- --test
```

## Features

- [x] Output the annotation file in Labelme format

## 3D model

The 3D model and texture image for the hand was created with
[MakeHuman 1.1.1](http://www.makehumancommunity.org/).