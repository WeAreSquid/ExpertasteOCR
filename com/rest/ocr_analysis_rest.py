from fastapi import APIRouter, UploadFile, File, Depends, Form
import json, urllib.parse, cv2, base64
import numpy as np
from typing import List
import ast

from com.service.ocr_service import OCRService

router = APIRouter()
@router.post('/ocr_analysis')
async def ocr_analysis_rest(name: str = Form(...), file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)

        # Decode the NumPy array into an OpenCV image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        ocr_service = OCRService()
        final_json = ocr_service.main_execution(name, img)
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
        
        
    
    