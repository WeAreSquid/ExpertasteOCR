import cv2
import numpy as np

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

        for label, coords in card_info_list['entity_results']:
            if label in labels_to_extract:
                extracted_elements[label] = coords

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
                result_list = []
                for line in result:
                    for word_info in line:
                        _result_json = {}
                        _result_json['entity'] = word_info[1][0]
                        _result_json['confidence'] = round(word_info[1][1], 2)*100
                        result_list.append(_result_json)
                        
                card_info_complete['entity_results'][key] = result_list

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
                _result_json = {}
                result_list = []
                for line in result:
                    for word_info in line:
                        _result_json = {}
                        _result_json['entity'] = word_info[1][0]
                        _result_json['confidence'] = round(word_info[1][1], 2)*100
                        result_list.append(_result_json)
                        
                card_info_complete['entity_results'][key] = result_list


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
                result_list = []
                for line in result:
                    for word_info in line:
                        _result_json = {}
                        _result_json['entity'] = word_info[1][0]
                        _result_json['confidence'] = round(word_info[1][1], 2)*100
                        result_list.append(_result_json)
                        
                card_info_complete['entity_results'][key] = result_list
                        
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
                result_list = []
                for line in result:
                    for word_info in line:
                        _result_json = {}
                        _result_json['entity'] = word_info[1][0]
                        _result_json['confidence'] = round(word_info[1][1], 2)*100
                        result_list.append(_result_json)
                        
                card_info_complete['entity_results'][key] = result_list
                    
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
                result_list = []
                for line in result:
                    for word_info in line:
                        _result_json = {}
                        _result_json['entity'] = word_info[1][0]
                        _result_json['confidence'] = round(word_info[1][1], 2)*100
                        result_list.append(_result_json)
                        
                card_info_complete['entity_results'][key] = result_list
            
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
                    if entry[2][1] <= y_center-weight/2:
                        first_group.append(_result_json)
                    elif entry[2][1] >= y_center-weight/2 and entry[2][1] <= y_center + weight/2:
                        second_group.append(_result_json)                            
                    elif entry[2][1] >= y_center + weight/2:
                        third_group.append(_result_json)
                
                
                card_info_complete['entity_results'][key]['first_PACKS'] = first_group
                card_info_complete['entity_results'][key]['second_PACKS'] = second_group
                card_info_complete['entity_results'][key]['third_PACKS'] = third_group

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
                    if entry[2][1] <= y_center-weight/2:
                        first_group.append(_result_json)
                    elif entry[2][1] >= y_center-weight/2 and entry[2][1] <= y_center + weight/2:
                        second_group.append(_result_json)                            
                    elif entry[2][1] >= y_center + weight/2:
                        third_group.append(_result_json)

                card_info_complete['entity_results'][key]['first_RATING'] = first_group
                card_info_complete['entity_results'][key]['second_RATING'] = second_group
                card_info_complete['entity_results'][key]['third_RATING'] = third_group
                
        return card_info_complete