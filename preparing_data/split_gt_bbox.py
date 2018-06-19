from __future__ import division
import os
import numpy as np
import math
import cv2 as cv

#download the dataset from here: 
#  http://rrc.cvc.uab.es/?ch=4&com=downloads

g_image_path = '/tmp/ICDAR_2015/training_images'
g_gt_path = '/tmp/ICDAR_2015/gt_text'
g_out_path = '/tmp/image_icdar_2015'
g_label_path = '/tmp/label_icdar_2015'

g_min_size = 600
g_max_size = 1000

g_b_debug = False


if not os.path.exists(g_out_path):
    os.makedirs(g_out_path)
files = os.listdir(g_image_path)
files.sort(reverse=True)

g_stopping_langs = set(['Arabic', 'Bangla', 'Japanese', 'Korean'])

for file in files:
    _, basename = os.path.split(file)
    if basename.lower().split('.')[-1] not in ['jpg', 'png']:
        continue
    stem, ext = os.path.splitext(basename)
    gt_file = os.path.join(g_gt_path, 'gt_' + stem + '.txt')
    img_image_path = os.path.join(g_path, file)

    _is_train_data = True
    for line in open(gt_file):
        _ret = line.split(',')
        _lang = _ret[8]
        if _lang in g_stopping_langs:
            _is_train_data = False
            break
    if not _is_train_data:
        print("stopped by language: %s", gt_file)
        continue

    print(img_image_path)
    img = cv.imread(img_image_path)
    img_size = img.shape
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])

    im_scale = float(g_min_size) / float(im_size_min)
    if np.round(im_scale * im_size_max) > g_max_size:
        im_scale = float(g_max_size) / float(im_size_max)
    re_im = cv.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv.INTER_LINEAR)
    re_size = re_im.shape
    cv.imwrite(os.path.join(g_out_path, stem) + '.jpg', re_im)

    with open(gt_file, 'r') as f:
        lines = f.readlines()
    line_idx = 0
    for line in lines:
        line_idx += 1
        if '\xef\xbb\xbf' in line:
            line = line.replace('\xef\xbb\xbf', '')

        splitted_line = line.strip().lower().split(',')
        pt_x = np.zeros((4, 1))
        pt_y = np.zeros((4, 1))
        pt_x[0, 0] = int(float(splitted_line[0]) / img_size[1] * re_size[1])
        pt_y[0, 0] = int(float(splitted_line[1]) / img_size[0] * re_size[0])
        pt_x[1, 0] = int(float(splitted_line[2]) / img_size[1] * re_size[1])
        pt_y[1, 0] = int(float(splitted_line[3]) / img_size[0] * re_size[0])
        pt_x[2, 0] = int(float(splitted_line[4]) / img_size[1] * re_size[1])
        pt_y[2, 0] = int(float(splitted_line[5]) / img_size[0] * re_size[0])
        pt_x[3, 0] = int(float(splitted_line[6]) / img_size[1] * re_size[1])
        pt_y[3, 0] = int(float(splitted_line[7]) / img_size[0] * re_size[0])

        ind_x = np.argsort(pt_x, axis=0)
        pt_x = pt_x[ind_x]
        pt_y = pt_y[ind_x]

        if pt_y[0] < pt_y[1]:
            pt1 = (pt_x[0], pt_y[0])
            pt3 = (pt_x[1], pt_y[1])
        else:
            pt1 = (pt_x[1], pt_y[1])
            pt3 = (pt_x[0], pt_y[0])

        if pt_y[2] < pt_y[3]:
            pt2 = (pt_x[2], pt_y[2])
            pt4 = (pt_x[3], pt_y[3])
        else:
            pt2 = (pt_x[3], pt_y[3])
            pt4 = (pt_x[2], pt_y[2])

        #fix bug_1, for xmin, it should be point 1 and point 3 
        xmin = int(min(pt1[0], pt3[0]))

        ymin = int(min(pt1[1], pt2[1]))
        xmax = int(max(pt2[0], pt4[0]))
        ymax = int(max(pt3[1], pt4[1]))

        if xmin < 0:
            xmin = 0
        if xmax > re_size[1] - 1:
            xmax = re_size[1] - 1
        if ymin < 0:
            ymin = 0
        if ymax > re_size[0] - 1:
            ymax = re_size[0] - 1

        width = xmax - xmin
        height = ymax - ymin

        step_size = 16.0
        x_left = []
        x_right = []
        x_left.append(xmin)

        # x_left_start may equal xmin+1, which means xmin =x_left[0]-1
        x_left_start = int(math.ceil(xmin / step_size) * step_size)

        if x_left_start == xmin:
            x_left_start = xmin + 16
        for i in np.arange(x_left_start, xmax, 16):
            x_left.append(i)
        x_left = np.array(x_left)

        #fix bug_2, x_left_start may be larger than xmax
        if x_left_start <= xmax:
            x_right.append(x_left_start - 1)

        if g_b_debug and x_left_start - xmin != 16:
            print "[debug_youlie]line_idx:{} xmin: {}, x_left_start:{}, xmax:{}".format(line_idx, xmin, x_left_start, xmax)

        for i in range(1, len(x_left) - 1):
            x_right.append(x_left[i] + 15)
        x_right.append(xmax)
        x_right = np.array(x_right)

        if g_b_debug:
            print "debug_youlie_v2: left_len:{}, right_len:{}, equals: {}".\
			    format(len(x_left), len(x_right), len(x_left)==len(x_right))

        idx = np.where(x_left == x_right)
 
        x_left = np.delete(x_left, idx, axis=0)
        x_right = np.delete(x_right, idx, axis=0)

        if not os.path.exists(g_label_path):
            os.makedirs(g_label_path)
        with open(os.path.join(g_label_path, stem) + '.txt', 'a') as f:
            _xside_num1 = 0
            _xside_num2 = 0

            if g_b_debug and len(x_left) <= 1:
                print "[debug_youlie]Small number of left_side len:{} v:{}, right_side len:{} v:{}, xmin: {}, xmax: {}".\
                    format(len(x_left), x_left, len(x_right), x_right, xmin, xmax)

            for i in range(len(x_left)):
                _center = x_left[i] + (x_right[i] - x_left[i])/2
                _center2left = _center - xmin
                _center2right = xmax - _center
                
                _xside = xmin if _center2left < _center2right else xmax
                if g_b_debug and _xside +1 == x_left[0]:
                    print "[debug_youlie] _xside:{}, xmin: {}, x_left_0:{}".format(_xside, xmin, x_left[0])

                if g_b_debug and _xside+1 != x_left[0] and _xside != x_left[0] and _xside != x_right[len(x_right)-1]:
                    print "[debug_youlie] Error occurs._xside:{}, xmin: {}, x_left_0:{}, x_right_last:{}".\
                        format(_xside, xmin, x_left[0], x_right[len(x_right)-1])

                if g_b_debug:
                    print "[debug_youlie] line_idx:{}, x_left:{}, x_right:{}, xmin:{}, xmax:{}, \
					    i:{},l_len{},r_len{},_xside:{}".\
						format(line_idx, x_left[i], x_right[i], xmin, xmax, i, len(x_left), len(x_right), _xside)

                f.writelines("text\t")
                f.writelines(str(int(x_left[i])))
                f.writelines("\t")
                f.writelines(str(int(ymin)))
                f.writelines("\t")
                f.writelines(str(int(x_right[i])))
                f.writelines("\t")
                f.writelines(str(int(ymax)))
                f.writelines("\t")
                f.writelines(str(int(_xside)))
                f.writelines("\n")
            if g_b_debug:
                print "_xside_num1:{}, _xside_num2:{}".format(_xside_num1, _xside_num2) 
