import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


class BlobDetector:
    def detect_blobs(self, image):
        # Scan the image for black regions
        otsu_thresh = self.gray_image(image)
        threshold=0.8
        window_size=(40, 40)
        clusters_center = self.scan_image_for_black_regions(otsu_thresh, window_size, threshold)

        # Draw the detected regions on the image
        output_image = image.copy()
        color = (0, 255, 0)
        radius = 5
        thickness = -1
        for center in clusters_center:
            cv2.circle(output_image, center, radius, color, thickness)
            #cv2.imwrite('black_regions_detected.jpg', output_image)
        return clusters_center

    def gray_image(self, image):
        # Read image
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray_image, (5, 5), 1)
        # Apply Otsu's binarization
        _, otsu_thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return otsu_thresh
    
    def is_black_region(self, region, threshold):
        """Check if the region has more than the threshold percentage of black pixels."""
        total_pixels = region.size
        black_pixels = np.sum(region == 0)
        return (black_pixels / total_pixels) > threshold
        
    def cluster_regions(self, box_regions):
        center_list = []

        for (x1, y1, x2, y2) in box_regions:
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            center_list.append((center_x,center_y))
        # Define the maximum distance (epsilon) between two points for them to be considered in the same neighborhood
        epsilon = 40
        # Define the minimum number of points to form a dense region (cluster)
        min_samples = 1

        # Apply DBSCAN
        coordinates_array = np.array(center_list)
        db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(coordinates_array)
        labels = db.labels_
        
        # Create a list to hold the clusters
        unique_labels = set(labels)
        clusters = []
        for k in unique_labels:
            if k != -1:
                class_member_mask = (labels == k)
                xy = coordinates_array[class_member_mask]
                clusters.append(xy.tolist())

        center_clusters = []
        for i, cluster in enumerate(clusters):
            one=[]
            two=[]
            for tup in cluster:
                one.append(tup[0])
                two.append(tup[1])

            cluster_center = (int(np.mean(one)),  int(np.mean(two)))
            center_clusters.append(cluster_center)

        return center_clusters

    def scan_image_for_black_regions(self, image, window_size, threshold):
        """Scan the image and detect regions with more than threshold percentage of black pixels."""
        regions = []
        h, w = image.shape
        window_h, window_w = window_size

        for y in range(0, h, int(window_h/3)):
            for x in range(0, w, int(window_w/3)):
                window = image[y:y + window_h, x:x + window_w]
                if window.shape[0] != window_h or window.shape[1] != window_w:
                    continue
                if self.is_black_region(window, threshold):
                    regions.append((x, y, x + window_w, y + window_h))
        
        if len(regions) != 0:
            num_coordinates = len(regions[0])
            sums = [0] * num_coordinates
            count = 0
            cluster_centers = self.cluster_regions(regions)
        else:
            cluster_centers = []
        return cluster_centers
  
