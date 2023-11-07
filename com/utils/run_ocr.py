from paddleocr import PaddleOCR, draw_ocr
from com.utils.act_with_image import ActWithImage

class RunOcr(ActWithImage):
    def execute_paddleocr(self, sampler_name, ocr_model, img):
        # Perform OCR
        image_dictionary = {}
        image_dictionary['name'] = sampler_name
        print(f'Processing {sampler_name}')
        result = ocr_model.ocr(img, cls=True)
        # Process OCR results
        card_out = []
        if result != [None]:
            for line in result:
                for word_info in line:
                    word = word_info[0]
                    confidence = word_info[1]
                    card_out.append((confidence[0], word))
        
            image_dictionary['entity_results'] = card_out
            
            # Visualization (optional)
            boxes = [elements[0] for line in result for elements in line]
            pairs = [elements[1] for line in result for elements in line]
            txts = [pair[0] for pair in pairs]
            scores = [pair[1] for pair in pairs]
        
        else:
            card_out.append(('',[[]]))
            image_dictionary['entity_results'] = card_out

        return image_dictionary