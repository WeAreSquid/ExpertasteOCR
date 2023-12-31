import numpy as np
import cv2

class workPoints():   
    def order_points(self, samplers_points):
        # Sort points based on their x and y coordinates
        sorted_points = sorted(samplers_points[1], key=lambda p: (p[0], p[1]))
        
        # Split the sorted points into two halves (top and bottom)
        top_half = sorted_points[:len(sorted_points) // 2]
        bottom_half = sorted_points[len(sorted_points) // 2:]
        
        # Sort the top half in ascending order of y-coordinate
        top_half.sort(key=lambda p: p[1])
        
        # Sort the bottom half in descending order of y-coordinate
        bottom_half.sort(key=lambda p: p[1], reverse=True)
        
        # Arrange the points in clockwise order
        clockwise_points = top_half + bottom_half
        
        center = [sum([x[0] for x in clockwise_points])//4, sum([y[1] for y in clockwise_points])//4]
        _points = {'name' : samplers_points[0], 'points': clockwise_points, 'center': center}
        return _points
    
    def crop_cards(self, sampler_points, img):
        l = 0
        # Create a mask (black image) of the same size as the original image
        mask = np.zeros_like(img)
        height, width, channels = img.shape
        _center = sampler_points['center']
        _points = sampler_points['points']
        
        points = np.array(_points)
        _centroid = np.mean(points, axis=0)
        _entity_points = sorted(points, key=lambda point: (-np.arctan2(point[1] - _centroid[1], point[0] - _centroid[0])))
        max_per_coordinate = np.max(_entity_points, axis=0)
        min_per_coordinate = np.min(_entity_points, axis=0)
                    
        _width = int(max_per_coordinate[0] - min_per_coordinate[0])
        _height = int(max_per_coordinate[1] - min_per_coordinate[1])
        
        #cropped_image = img[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        cropped_image = img[int(min_per_coordinate[1]) - int(_width/3):int(max_per_coordinate[1]) + int(_width/2.5), int(_entity_points[0][0]):int(width)]
        
        sampler_points['cropped_card'] = cropped_image
        return sampler_points
        
        