import cv2, Levenshtein
import numpy as np
from PIL import Image
import cv2
import io
from rembg import remove
from com.utils.entities_position import EntityPosition
entity_position = EntityPosition()

from com.utils.blob_detection import BlobDetector
blob_detector = BlobDetector()

class ProcessCards():       
    def crop_on_cards(self, card_info_list, ocr_model, img):
        card_info_complete = {}
        image_card = img.copy()
        card_info_complete['name'] = card_info_list['name']
        card_info_complete['entity_results'] = {}

        labels_to_extract = ['GENDER', 'AGE', 'ETHNICITY', 'HEARD OF', 'TRIED', 'PACKS', 'RATING']
        # Iterate through the data and extract the elements for the specified labels
        extracted_elements = {label: '' for label in labels_to_extract}
        for element in labels_to_extract:
            card_info_complete['entity_results'][element] = {}

        for ideal in labels_to_extract:
            distance_list = []
            for label, coords in card_info_list['entity_results']:
                if len(label) >= 3:
                    distance = Levenshtein.distance(label, ideal)
                    similarity = 1 - (distance / max(len(label), len(ideal)))
                    if similarity >= 0.5:
                        distance_list.append((similarity, ideal, coords))
                
            sorted_list = sorted(distance_list, key=lambda x: x[0])
            extracted_elements[sorted_list[0][1]] = sorted_list[0][2]

        extracted_elements_clean = {key: value for key, value in extracted_elements.items() if value!=''}

        for key in extracted_elements_clean.keys():
            if key == 'ETHNICITY':
                print('PROCESSING ETHNICITY!')
                points = np.array(extracted_elements_clean[key])
                centroid = np.mean(points, axis=0)
                sorted_points = sorted(points, key=lambda point: (-np.arctan2(point[1] - centroid[1], point[0] - centroid[0])))
                max_per_coordinate = np.max(sorted_points, axis=0)
                min_per_coordinate = np.min(sorted_points, axis=0)
                
                width = int(max_per_coordinate[0] - min_per_coordinate[0])
                height = int(max_per_coordinate[1] - min_per_coordinate[1])
                
                #original[y:y+h, x:x+w]
                cropped_image = image_card[ int(centroid[1]) - int(height): int(centroid[1]) + int(3*height), int(centroid[0]) - int(1.4*width): int(min_per_coordinate[0]) + int(1.8*width)]
                #cv2.imwrite( r"./cropped_image_ethnicity.jpg", cropped_image)
                
                result = ocr_model.ocr(cropped_image, cls=True)
                try:
                    blobs_center_list = blob_detector.detect_blobs(cropped_image)
                except Exception as e:
                    print(e)

                result_list = entity_position.match_main_entity(key, result)
                _expected = ['AA', 'H', 'C', 'O', 'A']
                selected_item = entity_position.parse_positions(key, result_list, blobs_center_list,  _expected)
                result_to_send = entity_position.build_response(key, selected_item, _expected, result_list)
                card_info_complete['entity_results'][key] = result_to_send

            elif key == 'AGE':
                print('PROCESSING AGE!')
                points = np.array(extracted_elements_clean[key])
                centroid = np.mean(points, axis=0)
                sorted_points = sorted(points, key=lambda point: (-np.arctan2(point[1] - centroid[1], point[0] - centroid[0])))
                max_per_coordinate = np.max(sorted_points, axis=0)
                min_per_coordinate = np.min(sorted_points, axis=0)
                
                width = int(max_per_coordinate[0] - min_per_coordinate[0])
                height = int(max_per_coordinate[1] - min_per_coordinate[1])
                
                #original[y:y+h, x:x+w]
                cropped_image = image_card[ int(centroid[1]) - int(0.5*height): int(centroid[1]) + int(2.8*height), int(centroid[0]) - int(3.8*width): int(centroid[0]) + int(3.8*width)]
                #cv2.imwrite( r"./cropped_image_age.jpg", cropped_image)
                
                result = ocr_model.ocr(cropped_image, cls=True)
                
                result_list = entity_position.match_main_entity(key, result)
                blobs_center_list = blob_detector.detect_blobs(cropped_image)
                
                _expected = ['+20', '+30', '+40', '+50']
                selected_item = entity_position.parse_positions(key, result_list, blobs_center_list,  _expected)

                try:
                    result_to_send = entity_position.build_response(key, selected_item, _expected, result_list)
                except Exception as e:
                    print('EXCEPTION HERE!')
                    print(e)   
                card_info_complete['entity_results'][key] = result_to_send


            elif key == 'GENDER':
                print('PROCESSING GENDER!')
                points = np.array(extracted_elements_clean[key])
                centroid = np.mean(points, axis=0)
                
                sorted_points = sorted(points, key=lambda point: (-np.arctan2(point[1] - centroid[1], point[0] - centroid[0])))

                max_per_coordinate = np.max(sorted_points, axis=0)
                min_per_coordinate = np.min(sorted_points, axis=0)

                weight = int(max_per_coordinate[0] - min_per_coordinate[0])
                height = int(max_per_coordinate[1] - min_per_coordinate[1])

                #original[y:y+h, x:x+w]
                cropped_image = image_card[ int(centroid[1]) - int(0.5*height): int(centroid[1]) + int(2.5*height), int(centroid[0]) - int(0.7*weight): int(centroid[0]) + int(0.7*weight)]
                
                #cv2.imwrite( r"./cropped_image_gender.jpg", cropped_image)
                result = ocr_model.ocr(cropped_image, cls=True)
                result_list = entity_position.match_main_entity(key, result)
                blobs_center_list = blob_detector.detect_blobs(cropped_image)
                
                #EXPECTED LIST IN LOWER CASE ALWAYS!!! IN THE SAME ORDER AS TEMPLATE
                _expected = ['F', 'M']
                selected_item = entity_position.parse_positions(key, result_list, blobs_center_list, _expected)

                result_to_send = entity_position.build_response(key, selected_item, _expected, result_list)  
                card_info_complete['entity_results'][key] = result_to_send
                        
            elif key == 'HEARD OF':
                print('PROCESSING HEARD OF!')
                points = np.array(extracted_elements_clean[key])
                centroid = np.mean(points, axis=0)
                sorted_points = sorted(points, key=lambda point: (-np.arctan2(point[1] - centroid[1], point[0] - centroid[0])))
                max_per_coordinate = np.max(sorted_points, axis=0)
                min_per_coordinate = np.min(sorted_points, axis=0)
                
                weight = int(max_per_coordinate[0] - min_per_coordinate[0])
                height = int(max_per_coordinate[1] - min_per_coordinate[1])
                
                #original[y:y+h, x:x+w]
                cropped_image = image_card[ int(centroid[1]) - int(0.7*height): int(centroid[1]) + int(0.7*height), int(centroid[0]) - int(0.6*weight): int(centroid[0]) + int(1.8*weight)]
                #cv2.imwrite( r"./cropped_image_heardOF.jpg", cropped_image)
                
                result = ocr_model.ocr(cropped_image, cls=True)
                result_list = entity_position.match_main_entity(key, result)
                blobs_center_list = blob_detector.detect_blobs(cropped_image)
                
                #EXPECTED LIST IN LOWER CASE ALWAYS!!! IN THE SAME ORDER AS TEMPLATE
                _expected = ['Y', 'N']
                selected_item = entity_position.parse_positions(key, result_list, blobs_center_list, _expected)

                result_to_send = entity_position.build_response(key, selected_item, _expected, result_list)  
                card_info_complete['entity_results'][key] = result_to_send
                    
            elif key == 'TRIED':
                print('PROCESSING TRIED!')
                points = np.array(extracted_elements_clean[key])
                centroid = np.mean(points, axis=0)
                sorted_points = sorted(points, key=lambda point: (-np.arctan2(point[1] - centroid[1], point[0] - centroid[0])))
                max_per_coordinate = np.max(sorted_points, axis=0)
                min_per_coordinate = np.min(sorted_points, axis=0)
                
                weight = int(max_per_coordinate[0] - min_per_coordinate[0])
                height = int(max_per_coordinate[1] - min_per_coordinate[1])
                
                #original[y:y+h, x:x+w]
                cropped_image = image_card[ int(centroid[1]) - int(0.7*height): int(centroid[1]) + int(0.7*height), int(centroid[0]) - int(0.6*weight): int(centroid[0]) + int(3*weight)]
                #cv2.imwrite( r"./cropped_image_tried.jpg", cropped_image)
                
                result = ocr_model.ocr(cropped_image, cls=True)
                result_list = entity_position.match_main_entity(key, result)

                blobs_center_list = blob_detector.detect_blobs(cropped_image)

                #EXPECTED LIST IN LOWER CASE ALWAYS!!! IN THE SAME ORDER AS TEMPLATE
                _expected = ['Y', 'N']
                selected_item = entity_position.parse_positions(key, result_list, blobs_center_list, _expected)

                result_to_send = entity_position.build_response(key, selected_item, _expected, result_list)  
                card_info_complete['entity_results'][key] = result_to_send
            
            elif key == 'PACKS':
                print('PROCESSING PACKS!')
                card_info_complete['entity_results'][key] = {}
                points = np.array(extracted_elements_clean[key])
                centroid = np.mean(points, axis=0)
                sorted_points = sorted(points, key=lambda point: (-np.arctan2(point[1] - centroid[1], point[0] - centroid[0])))
                max_per_coordinate = np.max(sorted_points, axis=0)
                min_per_coordinate = np.min(sorted_points, axis=0)
                
                weight = int(max_per_coordinate[0] - min_per_coordinate[0])
                height = int(max_per_coordinate[1] - min_per_coordinate[1])
                
                #original[y:y+h, x:x+w]
                cropped_image = image_card[ int(centroid[1]) - int(0.6*height): int(centroid[1]) + int(6.1*height), int(centroid[0]) - int(2.5*weight): int(centroid[0]) + int(2.5*weight)]
                #cv2.imwrite( r"./cropped_image_packs.jpg", cropped_image)
                
                result = ocr_model.ocr(cropped_image, cls=True)
                results_list = [(recognized_points[0],recognized_points[1][0], np.mean(np.array(recognized_points[0]), axis=0), recognized_points[1][1]) for sublist in result for recognized_points in sublist]

                clean_list = results_list[1:]
                
                sorted_list = sorted(clean_list, key=lambda x: x[2][1])
                y_coordinates = [x[2][1] for x in sorted_list]
                y_center = np.mean(y_coordinates)

                first_group = []
                second_group = []
                third_group = []              

                for entry in sorted_list:
                    _result_json = {}
                    _result_json['entity'] = entry[1]
                    _result_json['confidence'] = round(entry[3], 2)*100
                    _result_json['position'] = np.array([entry[2][0], entry[2][1]])
                    if entry[2][1] <= y_center-weight/2:
                        #original[y:y+h, x:x+w]
                        first_group.append(_result_json)
                    elif entry[2][1] >= y_center-weight/2 and entry[2][1] <= y_center + weight/2:
                        second_group.append(_result_json)                            
                    elif entry[2][1] >= y_center + weight/2:
                        third_group.append(_result_json)

                blobs_center_list = blob_detector.detect_blobs(cropped_image)

                first_blob, second_blob, third_blob = [], [], []
                for blob in blobs_center_list:
                    if int(blob[1]) <= y_center-weight/2:
                        first_blob.append(blob)
                    if int(blob[1]) >= y_center-weight/2 and int(blob[1]) <= y_center+weight/2:
                        second_blob.append(blob)
                    if int(blob[1]) >= y_center + weight/2:
                        third_blob.append(blob)

                _expected, sections, group_list, blob_lists, response_labels = (['1p', '4p', '6p', '12p'], ['PROD-1', 'PROD-2', 'PROD-3'], [first_group, second_group, third_group], [first_blob, second_blob, third_blob], ['first_PACKS', 'second_PACKS', 'third_PACKS'])
                
                for i in range(3):
                    try:
                        selected_item = entity_position.parse_positions(sections[i], group_list[i], blob_lists[i], _expected)
                        result_to_send = entity_position.build_response(sections[i], selected_item, _expected, group_list[i])
                        card_info_complete['entity_results'][key][response_labels[i]] = result_to_send
                    except Exception as e:
                        print(e, sections[i]) 

            elif key == 'RATING':
                print('PROCESSING RATING!')
                card_info_complete['entity_results'][key] = {}
                points = np.array(extracted_elements_clean[key])
                centroid = np.mean(points, axis=0)
                sorted_points = sorted(points, key=lambda point: (-np.arctan2(point[1] - centroid[1], point[0] - centroid[0])))

                max_per_coordinate = np.max(sorted_points, axis=0)
                min_per_coordinate = np.min(sorted_points, axis=0)
                
                weight = int(max_per_coordinate[0] - min_per_coordinate[0])
                height = int(max_per_coordinate[1] - min_per_coordinate[1])
                
                #original[y:y+h, x:x+w]
                cropped_image = image_card[ int(centroid[1]) - int(0.6*height): int(centroid[1]) + int(7*height), int(centroid[0]) - int(1.5*weight): int(centroid[0]) + int(1.5*weight)]
                #cv2.imwrite( r"./cropped_image_rating.jpg", cropped_image)
                
                result = ocr_model.ocr(cropped_image, cls=True)
                results_list = [(recognized_points[0],recognized_points[1][0], np.mean(np.array(recognized_points[0]), axis=0), recognized_points[1][1]) for sublist in result for recognized_points in sublist]

                clean_list = results_list[1:]
                
                sorted_list = sorted(clean_list, key=lambda x: x[2][1])
                y_coordinates = [x[2][1] for x in sorted_list]
                y_center = np.mean(y_coordinates)
                
                first_group = []
                second_group = []
                third_group = []

                for entry in sorted_list:
                    _result_json = {}
                    _result_json['entity'] = entry[1]
                    _result_json['confidence'] = round(entry[3], 2)*100
                    _result_json['position'] = np.array([entry[2][0], entry[2][1]])
                    if entry[2][1] <= y_center-weight/4:
                        first_group.append(_result_json)
                    elif entry[2][1] >= y_center-weight/4 and entry[2][1] <= y_center + weight/4:
                        second_group.append(_result_json)                           
                    elif entry[2][1] >= y_center + weight/4:
                        third_group.append(_result_json)

                blobs_center_list = blob_detector.detect_blobs(cropped_image)

                first_blob, second_blob, third_blob = [], [], []
                for blob in blobs_center_list:
                    if int(blob[1]) <= y_center-weight/4:
                        first_blob.append(blob)
                    if int(blob[1]) >= y_center-weight/4 and int(blob[1]) <= y_center+weight/4:
                        second_blob.append(blob)
                    if int(blob[1]) >= y_center + weight/4:
                        third_blob.append(blob)

                _expected, sections, group_list, blob_lists, response_labels = (['1', '2', '3', '4', '5'], ['', '', ''], [first_group, second_group, third_group], [first_blob, second_blob, third_blob], ['first_RATING', 'second_RATING', 'third_RATING'])
                
                for i in range(3):
                    try:
                        selected_item = entity_position.parse_positions(sections[i], group_list[i], blob_lists[i], _expected)
                        result_to_send = entity_position.build_response(sections[i], selected_item, _expected, group_list[i])
                        result_to_send_sorted = sorted(result_to_send, key=lambda x: int(x['entity']))
                        card_info_complete['entity_results'][key][response_labels[i]] = result_to_send_sorted
                    except Exception as e:
                        print(e) 
        return card_info_complete