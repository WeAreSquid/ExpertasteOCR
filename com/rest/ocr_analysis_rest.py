from fastapi import APIRouter
import json, urllib.parse, cv2, base64
import numpy as np 

from com.models.input_model import InputModel
from com.service.ocr_service import OCRService

router = APIRouter()
@router.post('/ocr_analysis')
async def ocr_analysis_rest(input_data: InputModel):
    #data_form = data.dict(exclude = {'file'})
    #file_bytes = await data.file.read()
    #file = cv2.imdecode(np.frombuffer(file_bytes, dtype=np.uint8), flags=1)
    try:
        decoded_bytes = base64.b64decode(input_data.cropped_card)
        array = np.frombuffer(decoded_bytes, dtype=np.uint8)
        original_shape = input_data.array_shape  # For example
        image_array = array.reshape((original_shape[0], original_shape[1], original_shape[2]))
        
        ocr_service = OCRService()
        final_json = ocr_service.main_execution(input_data.name, image_array)
        json_string = json.dumps(final_json)
        return json.loads(json_string)
    except Exception as e:
        print(e)
        return {'message': e}
        
    """
    try:
        #Calling next service
        URL = 'url for next service'
        header = {'content-type':'application/json'}
        header = json.loads(header)
        async with httpx.AsyncClient() as client:
            response = await client.request('POST', URL, json = data_form, header = header, follow_redirects = True, timeout = None)
            return {'message': 'Service completed', 'status code': response.status_code}
    except httpx.HTTPStatusError as exc:
        print('Failed!')
    """
        
        
    
    