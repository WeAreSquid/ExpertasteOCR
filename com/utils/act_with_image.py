import cv2
import os
from paddleocr import PaddleOCR

class ActWithImage():
    def show_image(self, image):
        cv2.imshow('image', image)
        
    def set_image(self, _image):
        self.image = _image
        return self.image
    
    def get_image(self, image_path):
        return cv2.imread(image_path)
    
    def store_image(self, image, file_name, prefix = ''):
        path = r"./" + "{}_{}".format(prefix,file_name) 
        cv2.imwrite(path, image)
    
    def delete_image(self, file_name):
        path = r"./images/" + file_name
        os.remove(path)
    
    def instance_model(self):
        ocr_model = PaddleOCR(ocr_version="PP-OCRv3",lang='en', rec_algorithm = 'CRNN', rec_char_type = 'en', use_angle_cls=True)
        return ocr_model
