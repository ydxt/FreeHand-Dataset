# Copyright 2019 Katsuya Iida. All rights reserved.
# See LICENSE file in the project root for full license information.

from importlib.resources import path
import bpy
import mathutils

import sys
import os
import random
import json
from math import pi, cos, sin
import numpy as np


BONE_NAMES = (
    'Camera',  # roll
    'Camera',  # elevation
    'Camera',  # azmith
    'wrist.R', # hori
    'wrist.R', # vert
    'finger1.R',
    'finger2-1.R',
    'finger3-1.R',
    'finger4-1.R',
    'finger5-1.R',
    )
    
BONE_MIN = np.array([
    -30,
    0,
    0,
    -10,
    -30,
    -5,
    -5,
    -5,
    -5,
    -5
    ])

BONE_MAX = np.array([
    90,
    360,
    360,
    10,
    30,
    40,
    40,
    40,
    40,
    40
    ])

BONE_RAND_RANGE = BONE_MAX - BONE_MIN

BBOX_BONE_NAMES = (['wrist.R'] + 
    ['finger{}-{}.R'.format(i, j)
     for i in range(1, 6)
     for j in (1, 3)
    ])

BBOX_FULL_BONE_NAMES = []
BBOX_FULL_BONE_NAMES.append('wrist.R')
for i in range(1, 6):
    for j in range(0, 4):
        if j==0 and i==1:
            continue
        elif j!=0 or i==1:
            BBOX_FULL_BONE_NAMES.append('finger{}-{}.R'.format(i, j))
        else:
            BBOX_FULL_BONE_NAMES.append('metacarpal{}.R'.format(i-1))
    
def setup():
    '''Changes to the POSE mode.'''
    view_layer = bpy.context.view_layer
    ob = bpy.data.objects['Hand']
    ob.select_set(True)
    view_layer.objects.active = ob
    # bpy.context.scene.objects.active = ob # 2.7x
    bpy.ops.object.mode_set(mode='POSE')
    
    
def random_angles():
    '''Returns random angles as a numpy array.'''
    angles = np.random.random(len(BONE_NAMES)) * BONE_RAND_RANGE + BONE_MIN
    for i in range(6, 9):
        t = random.random()
        if t < 0.4:
            angles[i] = 0.0
        elif t < 0.6:
            angles[i] = BONE_MAX[i]
    if random.random() < 0.8:
        # Outer fingers are easier to be flexed. 
        angles[7] = max(angles[6], angles[7])
        angles[8] = max(angles[7], angles[8])
    angles[9] = angles[8] # ring and baby move together.
    return angles


def apply_handpose(angles):
    '''Applies angles to the hand bones.'''
    ob = bpy.data.objects['Hand']
    for i in range(3, 10):
        bonename = BONE_NAMES[i]
        bone = ob.pose.bones[bonename]
        angle = angles[i]
        if i == 4: # vert
            bone.rotation_quaternion.z = angle * pi / 180
        else:
            bone.rotation_quaternion.x = angle * pi / 180


def apply_camerapose(angles):
    '''Rotate the camera bone to move the camera.'''
    ob = bpy.data.objects['Hand']
    bone = ob.pose.bones['camera']
    bone.rotation_euler.x = angles[0] * pi / 180
    bone.rotation_euler.y = angles[1] * pi / 180
    bone.rotation_euler.z = angles[2] * pi / 180
    

def apply_lights():
    """Changes lights strength."""
    for i in range(1, 6):
        ob = bpy.data.objects['Light{}'.format(i)]
        if random.random() < 0.3:
            ob.data.energy = 0.0
        else:
            ob.data.energy = random.random() * 60.0


def apply_dark_lights():
    """Changes lights strength."""
    for i in range(1, 6):
        ob = bpy.data.objects['Light{}'.format(i)]
        ob.data.energy = (2.0*random.random()-1.0)*5.0


def get_render_pos(mat, pos):
    p = mat @ pos
    vx = p.x / -p.z * 3.888
    vy = p.y / -p.z * 3.888
    return vx, vy


def get_bounding_box(image_width, image_height):
    """
    Returns the bounding box of the hand in the image coordinate.
    The origin of the bbox is the left-bottom corner.

    return left_x, bottom_y, width, height
    """
    min_vx, min_vy, max_vx, max_vy = 1.0, 1.0, -1.0, -1.0
    ob = bpy.data.objects['Camera']
    mat = ob.matrix_world.normalized().inverted()
    ob = bpy.data.objects['Hand']

    for bonename in BBOX_BONE_NAMES:
        bone = ob.pose.bones[bonename]
        for pt in (bone.head, bone.tail):
            vx, vy = get_render_pos(mat, pt)
            if min_vx > vx:
                min_vx = vx
            if max_vx < vx:
                max_vx = vx
            if min_vy > vy:
                min_vy = vy
            if max_vy < vy:
                max_vy = vy

    # Translate to the image coordinate.
    min_x = round((min_vx + 0.5) * image_width)
    min_y = round((min_vy + 0.5) * image_height)
    max_x = round((max_vx + 0.5) * image_width)
    max_y = round((max_vy + 0.5) * image_height)
    
    return min_x, min_y, max_x - min_x, max_y - min_y


def get_finger_pose(image_width, image_height):
    """
    Returns finger pose list.
    The origin point is the left-bottom corner.
    """
    ob = bpy.data.objects['Camera']
    mat = ob.matrix_world.normalized().inverted()
    ob = bpy.data.objects['Hand']

    finger_21pts_list=[]
    for idx, bonename in enumerate(BBOX_FULL_BONE_NAMES):
        bone = ob.pose.bones[bonename]
        if 'wrist.R'==bonename:
            vx, vy = get_render_pos(mat, bone.head)
            cur_x = (vx + 0.5) * image_width
            cur_y = (vy + 0.5) * image_height
            bone_thumb_1 = ob.pose.bones[BBOX_FULL_BONE_NAMES[idx+1]]
            vx_thumb_1, vy_thumb_1 = get_render_pos(mat, bone_thumb_1.tail)
            next_x = (vx_thumb_1 + 0.5) * image_width
            next_y = (vy_thumb_1 + 0.5) * image_height
            finger_21pts_list.append([cur_x,cur_y])
            finger_21pts_list.append([(cur_x+next_x)/2,(cur_y+next_y)/2])
        else:
            vx, vy = get_render_pos(mat, bone.tail)
            cur_x = (vx + 0.5) * image_width
            cur_y = (vy + 0.5) * image_height
            finger_21pts_list.append([cur_x,cur_y])

    return finger_21pts_list


def labelme_format(imgfilename, image_width, image_height, bbox, finger_21pts_list):
    """
    将标注数据格式转化为labelme格式
    Args
        args：命令行输入的参数
            - imgfilename 保存的图片名称
            - bbox 输入数据格式为lx,by,w,h
            - finger_21pts_list 为手指21点坐标
            ...

        return: labelme format
        备注：
            `v=0, x=0, y=0`表示该点不可见且未标注，`v=1`表示该点有标注但不可见，`v=2`表示该点有标注且可见，
            v 即：visible_id
    """
    box_label=['hand']
    hand_keypoints_name = ["WRIST","THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP", \
    "INDEX_FINGER_MCP","INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP", \
    "MIDDLE_FINGER_MCP","MIDDLE_FINGER_PIP","MIDDLE_FINGER_DIP","MIDDLE_FINGER_TIP",\
    "RING_FINGER_MCP","RING_FINGER_PIP","RING_FINGER_DIP","RING_FINGER_TIP",\
    "PINKY_FINGER_MCP","PINKY_FINGER_PIP","PINKY_FINGER_DIP","PINKY_FINGER_TIP"]
    assert len(finger_21pts_list)==len(hand_keypoints_name)

    shapes_new=[]
    lx,by,w,h=bbox
    x1, y1 = lx, image_height - by - h
    x2,y2 = x1+w, image_height - by
    box_id = 1
    # box shape
    box_shape={}
    box_shape["label"]=box_label[0]
    box_shape["points"]=[]
    box_shape["points"].append([float(x1), float(y1)])
    box_shape["points"].append([float(x2), float(y2)])
    box_shape["group_id"]=box_id
    box_shape["shape_type"]="rectangle"
    box_shape["flags"]={}
    box_shape["flags"]["left"] = False
    box_shape["flags"]["right"] = True
    box_shape["visible_id"]=2
    shapes_new.append(box_shape)
    # point shape
    for idx, pt in enumerate(finger_21pts_list):
        pt_label = hand_keypoints_name[idx].lower()
        point_shape={}
        point_shape["label"]=pt_label
        point_shape["points"]=[]
        x_tmp,y_tmp = pt
        point_shape["points"].append([float(x_tmp), float(image_height - y_tmp)])
        point_shape["group_id"]=box_id
        point_shape["shape_type"]="point"
        point_shape["flags"]={}
        point_shape["visible_id"]=2
        shapes_new.append(point_shape)

    json_data={}
    json_data["version"]="5.0.1"
    json_data["flags"] = {}
    json_data["shapes"] = shapes_new
    json_data["imagePath"] = imgfilename
    json_data["imageData"] = None
    json_data["imageHeight"] = float(image_height)
    json_data["imageWidth"] = float(image_width)
    return json_data
        

def render_scene(dirpath, filename):
    bpy.context.scene.render.filepath = os.path.join(dirpath, filename)
    bpy.ops.render.render(write_still=True)


def process_once(dirpath, filename, annotations):
    angles = random_angles()
    apply_handpose(angles)
    apply_camerapose(angles)
    apply_lights()
    render_scene(dirpath, filename)
    #dg = bpy.context.evaluated_depsgraph_get()
    #dg.update() 
    #scene = bpy.data.scenes['Scene']
    #scene.update() # 2.7x
    #bpy.context.view_layer.update() #2.8x
    image_width = bpy.context.scene.render.resolution_x
    image_height = bpy.context.scene.render.resolution_y
    bbox = get_bounding_box(image_width, image_height)
    finger_21pts_list = get_finger_pose(image_width, image_height)
    labelme_json_data = labelme_format(filename, image_width, image_height, bbox, finger_21pts_list)
    anno = {
        'file_name': filename,
        'pose': list(angles),
        'bbox': bbox
        }
    annotations.append(anno)

    return labelme_json_data
    

def write_annotations(annotations, dirpath, filename):
    with open(os.path.join(dirpath, filename), 'w') as f:
        for anno in annotations:
            line = json.dumps(anno)
            f.write(line + '\n')


def _save_json(instance, save_path):
    """ 保存 coco json文件
    """
    json.dump(instance, open(save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        
        
def main(mode='test'):
    setup()

    if mode == 'test':
        num_blocks = 2
        num_images_per_block = 10
    else:
        num_blocks = 60
        num_images_per_block = 1000
        
    annotations_dirpath = bpy.path.abspath('//data/annotations')
    if not os.path.exists(annotations_dirpath):
        os.makedirs(annotations_dirpath)

    for i in range(num_blocks):
        annotations = []

        image_dirpath = bpy.path.abspath('//data/images/{:03d}'.format(i))
        if not os.path.exists(image_dirpath):
            os.makedirs(image_dirpath)

        for j in range(num_images_per_block):
            image_filename = '{:03d}-{:06d}.png'.format(i, j)
            labelme_json_data = process_once(image_dirpath, image_filename, annotations)
            labelme_json_filename = image_filename.replace('.png', '.json')
            _save_json(labelme_json_data,os.path.join(image_dirpath,labelme_json_filename))

        annotations_filename = '{:06d}.json'.format(i)
        write_annotations(annotations, annotations_dirpath, annotations_filename)

        
if __name__ == '__main__':
    mode = 'full' if '--full' in sys.argv[1:] else 'test'
    print('Mode is {}'.format(mode))
    main(mode=mode)
