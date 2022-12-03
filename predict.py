from typing import List
from sieve.types import FrameSingleObject, BoundingBox, FrameFetcher, Object
from sieve.predictors import ObjectPredictor
from sieve.types.constants import FRAME_NUMBER, BOUNDING_BOX, SCORE, CLASS, START_FRAME, END_FRAME, OBJECT
from sieve.types.outputs import StaticClassification
import skimage
from fastai.vision.all import *
import cv2
import torch

class EmotionPredictor(ObjectPredictor):

    def setup(self):
        url='https://huggingface.co/spaces/arturxmet/emotion-detector-space/resolve/main/export.pkl'
        res = requests.get(url, stream = True)
        with open('age.pkl', 'wb') as f:
            for chunk in res.iter_content(chunk_size = 1024*1024):
                if chunk:
                    f.write(chunk)
        if torch.cuda.is_available():
            print("Using GPU")
            self.learn = load_learner('age.pkl', cpu=False)
        else:
            print("Using CPU")
            self.learn = load_learner('age.pkl', cpu=True)
        self.labels = self.learn.dls.vocab

    def predict(self, frame_fetcher: FrameFetcher, object: Object) -> StaticClassification:
        object_start_frame, object_end_frame = object.get_static_attribute(START_FRAME), object.get_static_attribute(END_FRAME)
        frame_number = (object_start_frame + object_end_frame)

        object_bbox: BoundingBox = object.get_temporal_attribute(BOUNDING_BOX, frame_number)
        frame_data = frame_fetcher.get_frame(frame_number)

        out_dict = {OBJECT: object}

        frame_data = frame_data[int(object_bbox.y1):int(object_bbox.y2), int(object_bbox.x1):int(object_bbox.x2)]
        if frame_data.shape[0] == 0 or frame_data.shape[1] == 0:
            out_dict["age"] = "unknown"
            return StaticClassification(**out_dict)
        
        frame = cv2.resize(frame_data, (512, 512))
        frame = PILImage.create(frame)

        pred, pred_idx, probs = self.learn.predict(frame)

        emotion = pred

        out_dict['emotion'] = emotion

        return StaticClassification(**out_dict)