import core as corelib
from efficientnet_pytorch import EfficientNet
import albumentations as alt
import torchvision.transforms as transforms
import torch
import albumentations as alt
from albumentations.pytorch import ToTensorV2 as ToTensor
from imutils.video import VideoStream
from retinaface import  Retinaface_Detector
from PIL import Image
import cv2
import argparse



parser = argparse.ArgumentParser(description="Face anti smoofing")
parser.add_argument("--webcam", action='store_true', default=False, help="camera id 0")
parser.add_argument("--v", type=str, default='')
args = parser.parse_args()

retina = Retinaface_Detector()
class Anti_smoofing_track:
    def __init__(self, device):

        self.device = device
        self.model = {}
        self.model['lgsc'] = corelib.LGSC(drop_ratio=0.4)
        checkpoint = torch.load("weights/lgsc_siw_pretrained.pth", map_location=lambda storage, loc: storage)
        new_dict = {}
        for key in  checkpoint['backbone'].keys():
            new_dict[key.replace("module.", "")] = checkpoint['backbone'][key]
        self.model['lgsc'].load_state_dict(new_dict)
        self.model['lgsc'].to(self.device)
        self.model['lgsc'].eval()   # core

        self.model['classtify'] = EfficientNet.from_name('efficientnet-b0', num_classes=2)
        checkpoint = torch.load("weights/Iter_035000_net.ckpt", map_location=lambda storage, loc: storage)
        # print(checkpoint.keys())
        self.model['classtify'].load_state_dict(checkpoint['net_state_dict'])
        self.model['classtify'].to(self.device)
        self.model['classtify'].eval()
        self.tranform = {}
        self.tranform["classtify"] = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # range [0.0, 1.0] -> [-1.0,1.0]
        ])

        self.tranform["lgsc"] = alt.Compose([alt.Resize(224, 224, p=1), alt.Normalize(), ToTensor()])
        # self.

    def predict(self, image):
        img = self.tranform["classtify"](Image.fromarray(image.astype('uint8'), 'RGB'))
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        raw_logits = self.model['classtify'](img.to(self.device))
        raw_logits = torch.nn.functional.softmax(raw_logits)
        score, predict = torch.max(raw_logits.data, 1)
        if predict[0] == 1 and score[0] > 0.8:
            phase2 = self.tranform['lgsc'](image=image)['image'].unsqueeze(0)
            imgs_feat, clf_out = self.model['lgsc'](phase2.to(self.device))
            spoof_score = torch.mean(torch.abs(imgs_feat[-1])).item()

            if spoof_score < 0.0095:
                return score[0], spoof_score, 1
        return score[0], 1, 0

def test_cam(path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Smoofing_detect = Anti_smoofing_track(device)
    vid = VideoStream(path).start()
    frame = vid.read()
    while True:
        frame = vid.read()
        if frame is None:
            continue
        dicta = retina.align_multi(frame)
        for box in dicta["bboxs"]:
            face =  frame[box[1]:box[3], box[0]: box[2]]
            score_c, score_l, result = Smoofing_detect.predict(face)
            # print(score_c, score_l, result)
            if result:
                smoof_stt = "real"
            else:
                smoof_stt = "smoofing"

            frame = cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 1)
            frame = cv2.putText(frame, smoof_stt, (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (255, 0, 0), 1, cv2.LINE_AA) 
        cv2.imshow("window_name", frame) 
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    if args.webcam:
        args.v = 0
    test_cam(args.v)

