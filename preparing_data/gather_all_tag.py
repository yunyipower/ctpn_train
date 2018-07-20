#coding:utf-8
#########################################
# Description: Transform RectLabel object info to the mode we use 
# Author: John Zhou
#########################################

import os
import json


current_file_absolute_path = os.path.abspath(__file__)
current_dir_path = os.path.dirname(current_file_absolute_path)
images_path = os.path.abspath(os.path.join(current_dir_path, "./images"))
labels_path = os.path.abspath(os.path.join(current_dir_path, "./label"))
transform_label_path = os.path.abspath(os.path.join(current_dir_path, "./transform_label"))

_all_tag_info_list = []
for each_tag_file in os.listdir(labels_path):
  if not each_tag_file.endswith(".json"):
    continue 
  _cur_file_dict = {}
  _cur_file = os.path.join(labels_path, each_tag_file)
  _label_name = each_tag_file.split(".json")[0]+".txt"
  _saved_file = os.path.join(transform_label_path, _label_name)
  _pw = open(_saved_file, "wr+")

  print "=" * 20
  print _cur_file
  print "=" * 20
  _cur_orig_dict = json.loads(open(_cur_file, "r").read())
  _cur_file_dict["img_key"] = _cur_orig_dict["filename"]
  _tag_list = []
  _orig_objects_list = _cur_orig_dict["objects"]
  for each_object_dict in _orig_objects_list:
    _lang = "Default_Lang"
    _width = each_object_dict["x_y_w_h"][2]
    _height = each_object_dict["x_y_w_h"][3]
    _info_list = []

    _left_upper_x = each_object_dict["x_y_w_h"][0]
    _left_upper_y = each_object_dict["x_y_w_h"][1]
    _right_upper_x = _left_upper_x + _width
    _right_upper_y = _left_upper_y 
    _right_bottom_x = _right_upper_x
    _right_bottom_y = _right_upper_y + _height
    _left_bottom_x = _left_upper_x
    _left_bottom_y = _right_bottom_y

    #clockwise
    _info_list.append(str(_left_upper_x))
    _info_list.append(str(_left_upper_y))
    _info_list.append(str(_right_upper_x))
    _info_list.append(str(_right_upper_y))
    _info_list.append(str(_right_bottom_x))
    _info_list.append(str(_right_bottom_y))
    _info_list.append(str(_left_bottom_x))
    _info_list.append(str(_left_bottom_y))

    _info_list.append(_lang)
    _info_list.append("###")
    _pw.write((",").join(_info_list))
    _pw.write("\n")
  _pw.close() 
