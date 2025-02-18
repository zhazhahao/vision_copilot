import os, sys
import math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import argparse

def init_args():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser()
    # params for prediction engine
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--gpu_mem", type=int, default=500)
    parser.add_argument("--warmup", type=str2bool, default=True)

    # params for text detector
    parser.add_argument("--image_dir", type=str,default=None)
    parser.add_argument("--det_algorithm", type=str, default='DB')
    parser.add_argument("--det_model_path", type=str, default="/home/portable-00/VisionCopilot/pharmacy/checkpoints/ocr/ch_ptocr_v4_det_server_infer.pth")
    parser.add_argument("--det_limit_side_len", type=float, default=960)
    parser.add_argument("--det_limit_type", type=str, default='max')

    # DB parmas
    parser.add_argument("--det_db_thresh", type=float, default=0.3)
    parser.add_argument("--det_db_box_thresh", type=float, default=0.6)
    parser.add_argument("--det_db_unclip_ratio", type=float, default=1.5)
    parser.add_argument("--max_batch_size", type=int, default=10)
    parser.add_argument("--use_dilation", type=str2bool, default=False)
    parser.add_argument("--det_db_score_mode", type=str, default="fast")

    # params for text recognizer
    parser.add_argument("--rec_algorithm", type=str, default='CRNN')
    parser.add_argument("--rec_model_path", type=str, default="/home/portable-00/VisionCopilot/pharmacy/checkpoints/ocr/ch_ptocr_v4_rec_server_infer.pth")
    parser.add_argument("--rec_image_inverse", type=str2bool, default=True)
    parser.add_argument("--rec_image_shape", type=str, default="3, 48, 640")
    parser.add_argument("--rec_char_type", type=str, default='ch')
    parser.add_argument("--rec_batch_num", type=int, default=6)
    parser.add_argument("--max_text_length", type=int, default=25)

    parser.add_argument("--use_space_char", type=str2bool, default=True)
    parser.add_argument("--drop_score", type=float, default=0.5)
    parser.add_argument("--limited_max_width", type=int, default=1280)
    parser.add_argument("--limited_min_width", type=int, default=16)

    parser.add_argument(
        "--rec_char_dict_path",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                             '/home/portable-00/VisionCopilot/pharmacy/dependency/ocr/toolkit/ppocr_keys_v1.txt'))

    # params for text classifier
    parser.add_argument("--use_angle_cls", type=str2bool, default=True)
    parser.add_argument("--cls_model_path", type=str,default="/home/portable-00/VisionCopilot/pharmacy/checkpoints/ocr/ch_ptocr_mobile_v2.0_cls_infer.pth")
    parser.add_argument("--cls_image_shape", type=str, default="3, 48, 192")
    parser.add_argument("--label_list", type=list, default=['0', '180'])
    parser.add_argument("--cls_thresh", type=float, default=0.9)

    # params .yaml
    parser.add_argument("--det_yaml_path", type=str, default="/home/portable-00/VisionCopilot/pharmacy/configs/ocr/ch_PP-OCRv4_det_teacher.yml")
    parser.add_argument("--rec_yaml_path", type=str, default="/home/portable-00/VisionCopilot/pharmacy/configs/ocr/ch_PP-OCRv4_rec_hgnet.yml")
    parser.add_argument("--cls_yaml_path", type=str, default=None)

    return parser

def parse_args():
    parser = init_args()
    return parser.parse_args()

def get_default_config(args):
    return vars(args)


def read_network_config_from_yaml(yaml_path, char_num=None):
    if not os.path.exists(yaml_path):
        raise FileNotFoundError('{} is not existed.'.format(yaml_path))
    import yaml
    with open(yaml_path, encoding='utf-8') as f:
        res = yaml.safe_load(f)
    if res.get('Architecture') is None:
        raise ValueError('{} has no Architecture'.format(yaml_path))
    if res['Architecture']['Head']['name'] == 'MultiHead' and char_num is not None:
        res['Architecture']['Head']['out_channels_list'] = {
            'CTCLabelDecode': char_num,
            'SARLabelDecode': char_num + 2,
            'NRTRLabelDecode': char_num + 3
        }
    return res['Architecture']

def AnalysisConfig(weights_path, yaml_path=None, char_num=None):
    if not os.path.exists(os.path.abspath(weights_path)):
        raise FileNotFoundError('{} is not found.'.format(weights_path))

    if yaml_path is not None:
        return read_network_config_from_yaml(yaml_path, char_num=char_num)

    weights_basename = os.path.basename(weights_path)
    weights_name = weights_basename.lower()

    # supported_weights = ['ch_ptocr_server_v2.0_det_infer.pth',
    #                      'ch_ptocr_server_v2.0_rec_infer.pth',
    #                      'ch_ptocr_mobile_v2.0_det_infer.pth',
    #                      'ch_ptocr_mobile_v2.0_rec_infer.pth',
    #                      'ch_ptocr_mobile_v2.0_cls_infer.pth',
    #                    ]
    # assert weights_name in supported_weights, \
    #     "supported weights are {} but input weights is {}".format(supported_weights, weights_name)

    if weights_name == 'ch_ptocr_mobile_v2.0_cls_infer.pth':
        network_config = {'model_type':'cls',
                          'algorithm':'CLS',
                          'Transform':None,
                          'Backbone':{'name':'MobileNetV3', 'model_name':'small', 'scale':0.35},
                          'Neck':None,
                          'Head':{'name':'ClsHead', 'class_dim':2}}

    elif weights_name == 'ch_ptocr_v3_rec_infer.pth':
        network_config = {'model_type':'rec',
           'algorithm':'CRNN',
           'Transform':None,
           'Backbone':{'name':'MobileNetV1Enhance',
                       'scale':0.5,
                       'last_conv_stride': [1, 2],
                       'last_pool_type': 'avg'},
           'Neck':{'name':'SequenceEncoder',
                   'dims': 64,
                   'depth': 2,
                   'hidden_dims': 120,
                   'use_guide': True,
                   'encoder_type':'svtr'},
           'Head':{'name':'CTCHead', 'fc_decay': 2e-05}
           }

    elif weights_name == 'ch_ptocr_v3_det_infer.pth':
        network_config = {'model_type': 'det',
                          'algorithm': 'DB',
                          'Transform': None,
                          'Backbone': {'name': 'MobileNetV3', 'model_name': 'large', 'scale': 0.5, 'disable_se': True},
                          'Neck': {'name': 'RSEFPN', 'out_channels': 96, 'shortcut': True},
                          'Head': {'name': 'DBHead', 'k': 50}}

    elif 'om_' in weights_name and '_rec_' in weights_name:
        network_config = {'model_type': 'rec',
                          'algorithm': 'CRNN',
                          'Transform': None,
                          'Backbone': {'model_name': 'small', 'name': 'MobileNetV3', 'scale': 0.5,
                                       'small_stride': [1, 2, 2, 2]},
                          'Neck': {'name': 'SequenceEncoder', 'hidden_size': 48, 'encoder_type': 'om'},
                          'Head': {'name': 'CTCHead', 'fc_decay': 4e-05}}

    else:
        network_config = {'model_type': 'rec',
                          'algorithm': 'CRNN',
                          'Transform': None,
                          'Backbone': {'model_name': 'small', 'name': 'MobileNetV3', 'scale': 0.5,
                                       'small_stride': [1, 2, 2, 2]},
                          'Neck': {'name': 'SequenceEncoder', 'hidden_size': 48, 'encoder_type': 'rnn'},
                          'Head': {'name': 'CTCHead', 'fc_decay': 4e-05}}
        # raise NotImplementedError

    return network_config

def draw_text_det_res(dt_boxes, img_path):
    src_im = cv2.imread(img_path)
    for box in dt_boxes:
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
    return src_im


def resize_img(img, input_size=600):
    """
    resize img and limit the longest side of the image to input_size
    """
    img = np.array(img)
    im_shape = img.shape
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(input_size) / float(im_size_max)
    img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale)
    return img


def draw_ocr_box_txt(image,
                     boxes,
                     txts,
                     scores=None,
                     drop_score=0.5,
                     font_path="./doc/simfang.ttf"):
    h, w = image.height, image.width
    img_left = image.copy()
    img_right = Image.new('RGB', (w, h), (255, 255, 255))

    import random

    random.seed(0)
    draw_left = ImageDraw.Draw(img_left)
    draw_right = ImageDraw.Draw(img_right)
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        if scores is not None and scores[idx] < drop_score:
            continue
        color = (random.randint(0, 255), random.randint(0, 255),
                 random.randint(0, 255))
        draw_left.polygon(box, fill=color)
        draw_right.polygon(
            [
                box[0][0], box[0][1], box[1][0], box[1][1], box[2][0],
                box[2][1], box[3][0], box[3][1]
            ],
            outline=color)
        box_height = math.sqrt((box[0][0] - box[3][0])**2 + (box[0][1] - box[3][
            1])**2)
        box_width = math.sqrt((box[0][0] - box[1][0])**2 + (box[0][1] - box[1][
            1])**2)
        if box_height > 2 * box_width:
            font_size = max(int(box_width * 0.9), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            cur_y = box[0][1]
            for c in txt:
                # char_size = font.getsize(c)
                draw_right.text(
                    (box[0][0] + 3, cur_y), c, fill=(0, 0, 0), font=font)
                cur_y += 10
        else:
            font_size = max(int(box_height * 0.8), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            draw_right.text(
                [box[0][0], box[0][1]], txt, fill=(0, 0, 0), font=font)
    img_left = Image.blend(image, img_left, 0.5)
    img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(img_right, (w, 0, w * 2, h))
    return np.array(img_show)


def str_count(s):
    """
    Count the number of Chinese characters,
    a single English character and a single number
    equal to half the length of Chinese characters.
    args:
        s(string): the input of string
    return(int):
        the number of Chinese characters
    """
    import string
    count_zh = count_pu = 0
    s_len = len(s)
    en_dg_count = 0
    for c in s:
        if c in string.ascii_letters or c.isdigit() or c.isspace():
            en_dg_count += 1
        elif c.isalpha():
            count_zh += 1
        else:
            count_pu += 1
    return s_len - math.ceil(en_dg_count / 2)


def text_visual(texts,
                scores,
                img_h=400,
                img_w=600,
                threshold=0.,
                font_path="./doc/simfang.ttf"):
    """
    create new blank img and draw txt on it
    args:
        texts(list): the text will be draw
        scores(list|None): corresponding score of each txt
        img_h(int): the height of blank img
        img_w(int): the width of blank img
        font_path: the path of font which is used to draw text
    return(array):
    """
    if scores is not None:
        assert len(texts) == len(
            scores), "The number of txts and corresponding scores must match"

    def create_blank_img():
        blank_img = np.ones(shape=[img_h, img_w], dtype=np.int8) * 255
        blank_img[:, img_w - 1:] = 0
        blank_img = Image.fromarray(blank_img).convert("RGB")
        draw_txt = ImageDraw.Draw(blank_img)
        return blank_img, draw_txt

    blank_img, draw_txt = create_blank_img()

    font_size = 20
    txt_color = (0, 0, 0)
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")

    gap = font_size + 5
    txt_img_list = []
    count, index = 1, 0
    for idx, txt in enumerate(texts):
        index += 1
        if scores[idx] < threshold or math.isnan(scores[idx]):
            index -= 1
            continue
        first_line = True
        while str_count(txt) >= img_w // font_size - 4:
            tmp = txt
            txt = tmp[:img_w // font_size - 4]
            if first_line:
                new_txt = str(index) + ': ' + txt
                first_line = False
            else:
                new_txt = '    ' + txt
            draw_txt.text((0, gap * count), new_txt, txt_color, font=font)
            txt = tmp[img_w // font_size - 4:]
            if count >= img_h // gap - 1:
                txt_img_list.append(np.array(blank_img))
                blank_img, draw_txt = create_blank_img()
                count = 0
            count += 1
        if first_line:
            new_txt = str(index) + ': ' + txt + '   ' + '%.3f' % (scores[idx])
        else:
            new_txt = "  " + txt + "  " + '%.3f' % (scores[idx])
        draw_txt.text((0, gap * count), new_txt, txt_color, font=font)
        # whether add new blank img or not
        if count >= img_h // gap - 1 and idx + 1 < len(texts):
            txt_img_list.append(np.array(blank_img))
            blank_img, draw_txt = create_blank_img()
            count = 0
        count += 1
    txt_img_list.append(np.array(blank_img))
    if len(txt_img_list) == 1:
        blank_img = np.array(txt_img_list[0])
    else:
        blank_img = np.concatenate(txt_img_list, axis=1)
    return np.array(blank_img)


def base64_to_cv2(b64str):
    import base64
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.fromstring(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data


def draw_boxes(image, boxes, scores=None, drop_score=0.5):
    if scores is None:
        scores = [1] * len(boxes)
    for (box, score) in zip(boxes, scores):
        if score < drop_score:
            continue
        box = np.reshape(np.array(box), [-1, 1, 2]).astype(np.int64)
        image = cv2.polylines(np.array(image), [box], True, (255, 0, 0), 2)
    return image
