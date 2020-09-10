import os
import sys
import numpy as np
import cv2
import torch
import gc
from torch.autograd import Variable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image
from retinaface_pytorch.retinaface import load_retinaface_mbnet, RetinaFace_MobileNet
from retinaface_pytorch.utils import RetinaFace_Utils
from retinaface_pytorch.align_trans import get_reference_facial_points, warp_and_crop_face
import time

from torchvision import transforms as trans

def sort_list(list1): 
    z = [list1.index(x) for x in sorted(list1, reverse = True)] 
    return z 
def img_process(img, target_size, max_size):
    im_shape = img.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    # im = im.astype(np.float32)

    im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]), dtype=np.float32)
    for i in range(3):
        im_tensor[0, i, :, :] = im[:, :, 2 - i] 
    im_scale = [im_scale, im_scale]
    return im_tensor, im_scale

def img_process_tensorrt(img, target_size, max_size):
    im_shape = img.shape

    img = cv2.resize(img, (320, 256))
    im_size_min = target_size #np.min(im_shape[0:2])
    im_size_max = max_size #np.max(im_shape[0:2])
    im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]), dtype=np.float32)
    for i in range(3):
        im_tensor[0, i, :, :] = im[:, :, 2 - i] 

    im_scale = np.zeros((2, 1))

    im_scale[0] = max_size/im_shape[1] 
    im_scale[1] = target_size / im_shape[0]
    return im_tensor, im_scale

class Retinaface_Detector(object):
    def __init__(self):
        
        self.threshold = 0.6
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = RetinaFace_MobileNet()
        self.model = self.model.to(self.device)
        checkpoint = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'retinaface_pytorch/checkpoint.pth'))
        self.model.load_state_dict(checkpoint['state_dict'])
        del checkpoint
        gc.collect()
        self.img_process = img_process
        self.target_size = 480
        self.max_size = 640
        self.model.eval()
        self.pixel_means = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.pixel_stds = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.pixel_scale = float(1.0)
        self.refrence = get_reference_facial_points(default_square= True)
        self.utils = RetinaFace_Utils()

    def get_head_pose_status(self):
        return self.head_pose_face

    def change_head_pose(self, w, h):
        if self.get_head_pose_status():
            self.head_pose_predict.change_camera((h, w))

    def align(self, img, limit = None, min_face_size=None, thresholds = None, nms_thresholds=None):
        dict_output = self.align_multi(img)
        if len(dict_output['faces']) > 0:
            return dict_output["bboxs"][0], dict_output['faces'][0]
        return None, None

    def align_multi(self, img, limit = None, min_face_size=None, thresholds = None, nms_thresholds=None):
        faces = []
        faces_2_tensor = []
        lst_head_pose = [] 
        dict_result = {}
        sort = True
        img = np.array(img)
        im, im_scale = self.img_process(img, self.target_size, self.max_size)
        im = torch.from_numpy(im).to(self.device)
        im_tensor = Variable(im.contiguous())
        output = self.model(im_tensor)
        boxes, landmarks = self.utils.detect(im, output, self.threshold, im_scale)          
        if limit:
            boxes, landmarks = boxes[:limit], landmarks[:limit]  
            
        boxes = boxes.astype(np.int)
        landmarks = landmarks.astype(np.int)
        face_area = (boxes[:,2] - boxes[:,0])*(boxes[:,3]-boxes[:, 1])
        if len(boxes) > 0 and sort:
            face_area = (boxes[:,2] - boxes[:,0])*(boxes[:,3]-boxes[:, 1])
            indexs = np.argsort(face_area)[::-1]
            boxes = boxes[indexs]
            landmarks = landmarks[indexs]
            for i, landmark in enumerate(landmarks):
                warped_face, face_img_tranform = warp_and_crop_face(img, landmark, self.refrence, crop_size=(112,112))
                # face = Image.fromarray(warped_face)
                faces.append(warped_face)
                faces_2_tensor.append(torch.FloatTensor(face_img_tranform).contiguous().to(self.device).unsqueeze(0))

        num_face = len(boxes)
        dict_result["num_face"] = num_face
        dict_result["bboxs"] = boxes
        dict_result["faces"] = faces
        dict_result["faces_to_tensor"] = faces_2_tensor
        return dict_result


if __name__ == '__main__':
    import time
    reti = Retinaface_Detector()
    with open("train.txt", "r") as f:
        for line in f.readlines():
            line = line.rstrip()
            path = line.split(" ")[0]
            img = cv2.imread(path)
            dicta = reti.align_multi(img)
            for face in dicta["faces"]:
                cv2.imwrite(path, face)
                # face.save(path)
