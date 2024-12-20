import os
import onnxruntime
from PIL import Image
import numpy as np
import cv2
import copy
from inswapper.swapper import *


class FounderSwap:
    def __init__(self, model_dir: str):
        providers = onnxruntime.get_available_providers()
        self.face_analyser = getFaceAnalyser(model_dir, providers)
        self.face_analyser.prepare(ctx_id=0)
        model_path = os.path.join(model_dir, 'inswapper_128.onnx')
        self.face_swapper = getFaceSwapModel(model_path, providers=providers)
        self.source_face = None

    def set_source_face(self, source_image: np.array):
        # source_faces = get_many_faces(self.face_analyser, source_image)
        # self.source_face = source_faces[0]
        self.source_face = get_one_face(
            self.face_analyser, source_image)

    def swap_frame(self, frame: np.array):
        target_faces = get_many_faces(self.face_analyser, frame)
        num_target_faces = len(target_faces)
        if target_faces is not None:
            temp_image = copy.deepcopy(frame)
            if self.source_face is None:
                raise Exception("No source faces found!")
            for i in range(num_target_faces):
                temp_image = self.face_swapper.get(
                    temp_image, target_faces[i], self.source_face, paste_back=True)
            # result_image = Image.fromarray(
            #     cv2.cvtColor(temp_image, cv2.COLOR_BGR2RGB))
            return temp_image
        else:
            raise Exception("Need source image!")


if __name__ == "__main__":
    model_dir = "D:/Pycharm_Project/founder_faceswap-1/checkpoints"
    source_img_path = "D:/Pycharm_Project/founder_faceswap/inswapper/data/man1.jpeg"
    target_img_path = "D:/Pycharm_Project/founder_faceswap/inswapper/data/man2.jpeg"
    source_img = cv2.imread(source_img_path)
    target_img = cv2.imread(target_img_path)
    face_swap = FounderSwap(model_dir=model_dir)
    face_swap.set_source_face(source_img)
    result = face_swap.swap_frame(target_img)
    cv2.imshow("result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
