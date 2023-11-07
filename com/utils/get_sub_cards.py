from com.service.ocr_service import ActWithImage
from com.utils.work_points import workPoints

class GetSubCards(ActWithImage):    
    def get_cards(self, samplers_points, image):
        work_with_points = workPoints()
        _sampler_points = work_with_points.order_points(samplers_points)
        image_dictio = work_with_points.crop_cards(_sampler_points, image)
        return image_dictio
    
    
                

                    
                         
                
       