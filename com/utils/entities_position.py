import numpy as np

import statistics, Levenshtein

class EntityPosition:
    def match_main_entity(self, section, ocr_results):
        result_list = []
        for line in ocr_results:
            for word_info in line:
                _result_json = {}
                distance = Levenshtein.distance(word_info[1][0], section)
                similarity = 1 - (distance / max(len(word_info[1][0]), len(section)))
                if similarity >= 0.5:
                    _result_json['entity'] = section
                else:
                    _result_json['entity'] = word_info[1][0]
                _result_json['confidence'] = round(word_info[1][1], 2)*100
                points = np.array(word_info[0])
                centroid = np.mean(points, axis=0)
                _result_json['position'] = centroid
                result_list.append(_result_json)
        return result_list
        
    def parse_positions(self, section, entity_list, blob_list, known_list):
        _entity_list = [[item['entity'], item['position']] for item in entity_list if item['entity'] != section]
        self.entity_list = entity_list[:]
        _entity_list_bool = _entity_list[:]

        for i in range(len(_entity_list)):
            if _entity_list[i][0] in known_list:
                _entity_list_bool[i].append(True)
                _entity_list_bool[i].append(known_list.index(_entity_list[i][0]))
            else:
                _entity_list_bool[i].append(False)
                _entity_list_bool[i].append(-1)

        self._entity_list_bool = _entity_list_bool[:]
        _entity_list_bool_true = [item for item in _entity_list_bool if item[2]]
        self._entity_list_bool_true = _entity_list_bool_true
        self.blob_array_list = [np.array(t) for t in blob_list]
        _entity_list_bool_sorted = sorted(_entity_list_bool_true, key=lambda x: x[-1])
        
        if len(_entity_list_bool_sorted) == len(known_list):
            return 'Not determined'
        elif section in ['GENDER', 'HEARD OF', 'TRIED']:
            if len(_entity_list_bool_sorted) == 1 and len(blob_list) == 0:
                known_value = _entity_list_bool_sorted[0][0]
                self.selected_item = next(item for item in known_list if item != known_value)
            elif len(_entity_list_bool_sorted) == 0 and len(blob_list) == 1:
                if len(blob_list) == 1:
                    if section in ['HEARD OF', 'TRIED']:
                        x_distance = blob_list[0][0] - entity_list[0]['position'][0]
                        if x_distance > 1.5*40:
                            self.selected_item = known_list[1]
                        else:
                            self.selected_item = known_list[0]
            elif len(_entity_list_bool_sorted) == 1 and  len(blob_list) == 1:
                _x_coord = _entity_list_bool_sorted[0][1][0]
                _x_blob = blob_list[0][0]
                if _x_coord >= _x_blob:
                    self.selected_item = known_list[0]
                else:
                    self.selected_item = known_list[1]
            else:
                self.selected_item = 'Not determined'

            return self.selected_item
        else:
            if len(blob_list) == 0:
                return None
            if len(_entity_list_bool_sorted) == 0 or len(_entity_list_bool_sorted) == 1:
                return 'Not determined'
            # Loop through each pair of coordinates without repetitions
            else:
                x_distances = []
                for i in range(len(_entity_list_bool_sorted)):
                    for j in range(i + 1, len(_entity_list_bool_sorted)):
                        # Calculate the Euclidean distance between the coordinates
                        coord_x1 = np.array(_entity_list_bool_sorted[i][1][0])
                        coord_x2 = np.array(_entity_list_bool_sorted[j][1][0])
                        distance = abs(coord_x1 - coord_x2)
                        # Append the distance and the pair of coordinates to the distances list
                        x_distances.append((int(distance),(_entity_list_bool_sorted[i][3],_entity_list_bool_sorted[j][3])))
                _x = []
                for x in x_distances:
                    value = x[0]/(abs(x[1][0]-x[1][1]))
                    _x.append(value)
                x_average = int(np.mean(_x))

                reference_position = _entity_list_bool_sorted[0][3]
                reference_coordinates = _entity_list_bool_sorted[0][1]
                total_positions = len(known_list)
                total_right = len(known_list) - reference_position - 1
                total_left = reference_position
                blob_x = blob_list[0][0]
                relatives = []
                candidate_position_list = []
                for el in _entity_list_bool_sorted:
                    pos_estimate = round((blob_x-el[1][0])/x_average)
                    relatives.append(pos_estimate)
                    candidate_position_list.append(el[3] + pos_estimate)            
                most_common_element = most_common_element = statistics.mode(candidate_position_list)
                
                self.selected_item = known_list[most_common_element]
                return self.selected_item
        
    def build_response(self, section:str, selected:str, known_list:list, detected_entites = []):
        if selected == 'Not determined':
            result_list = []
            for element in detected_entites:
                result_list.append({'entity': element['entity']})
            return result_list
        else:
            result_list = [{'entity' : entry['entity']} for entry in self.entity_list if entry['entity'] == section]
            for el in self._entity_list_bool_true:
                res_dict = {} 
                res_dict['entity'] = el[0]
                result_list.append(res_dict)

            _detected = [e[0] for e in self._entity_list_bool_true]
            difference = [item for item in known_list if item not in _detected]

            for el in difference:
                res_dict = {} 
                if el != selected or selected == None:
                    res_dict['entity'] = el
                    result_list.append(res_dict)
            return result_list
        



            



        """
        [['H', array([134.5,  88. ]), True, 1], ['A', array([376. ,  88.5]), True, 4]]

        [{'entity': 'ETHNICITY', 'confidence': 89.0, 'position': array([214. ,  32.5])}, {'entity': 'M', 'confidence': 86.0, 'position': array([53., 87.])}, {'entity': 'H', 'confidence': 95.0, 'position': array([134.5,  88. ])}, {'entity': 'A', 'confidence': 100.0, 'position': array([376. ,  88.5])}]
        """

        return self.entity_list

        
        
