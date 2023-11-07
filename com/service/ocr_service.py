from com.utils.act_with_image import ActWithImage
from com.utils.get_sub_cards import GetSubCards
from com.utils.run_ocr import RunOcr
from com.utils.process_cards import ProcessCards


run_ocr = RunOcr()
get_sub_cards = GetSubCards()
process_cards = ProcessCards()

class OCRService(ActWithImage):
    def main_execution(self,sampler_name, img):
        self.set_image(img)
        self.ocr_model = self.instance_model()
        try:
            #card_info_dict = get_sub_cards.get_cards(self.image)
            ocr_entities_output = run_ocr.execute_paddleocr(sampler_name, self.ocr_model, img)
            sampler_final_json = process_cards.crop_on_cards(ocr_entities_output, self.ocr_model, img)
            return sampler_final_json
        except Exception as e:
            print(e)
            return {'message': e}
        #return final_json_list
        return
    
