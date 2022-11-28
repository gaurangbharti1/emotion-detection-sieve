from deepface import DeepFace
import cv2
from sieve.predictors import TemporalPredictor
from typing import List
from sieve.types import FrameSingleObject, BoundingBox
from sieve.types.outputs import Detection
from sieve.types.constants import FRAME_NUMBER, SCORE, CLASS, BOUNDING_BOX


class EmotionDetector(TemporalPredictor):
    def setup(self):
        self.model = DeepFace
        img = cv2.imread('happy.jpg')
        attributes = ['emotion']
        self.model.analyze(img, attributes)

    def predict(self, frame: FrameSingleObject) -> List[Detection]:
        frame_number = frame.get_temporal().frame_number
        frame_data = frame.get_temporal().get_array()
        attributes = ['emotion']
        emotion = self.model.analyze(frame_data, attributes)

        output_objects = []

        for k, emotion in emotion.items():
            out_cls = emotion['dominant_emotion']
            out_bbox = BoundingBox.from_array(emotion['region'])
            out_score = emotion['emotion']
            out_dict = {
                FRAME_NUMBER: frame_number,
                BOUNDING_BOX: out_bbox,
                SCORE: out_score,
                CLASS: out_cls
            }

            output_objects.append(
                Detection(
                    **out_dict
                )
            )
        return output_objects


