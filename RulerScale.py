import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Get the directory of the current file
current_file_directory = os.path.dirname(os.path.realpath(__file__))

# Change current working directory
os.chdir(current_file_directory)
imgPath="images/rule1.png"
img = cv2.imread(imgPath)  # vd.jpg,rice1dilate.jpg "D:\\blood_bao1s.jpg"
mean_img=cv2.pyrMeanShiftFiltering(img,20,30)#The larger the value is, the slower it will be. 20 is the radius and 30 is the color range.
gray_img = cv2.cvtColor(mean_img, cv2.COLOR_BGR2GRAY)

th2=cv2.adaptiveThreshold(gray_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,2)#11 is better than 5, mean is better than Gaussian, and mean is better closed. If it is a white ruler with black scale, use Thresh_Binary_inv
from scipy.signal import argrelextrema

# Project the white point onto the x-axis
projection = np.sum(th2, axis=0)
# Calculate the first derivative

gradient = np.gradient(projection)

# Find the local maximum in the projection result, i.e. the location of the peak
peaks = argrelextrema(projection, np.greater)

# Calculate distance between peaks
distances = np.diff(peaks)

# Print the distribution of the number of white points along the x-axis and the distance between the peaks
print("The distribution of the number of white points along the x-axis:", projection)
print("peak position:", peaks)
print("distance between peaks:", distances)
from scipy.signal import find_peaks
from sklearn.cluster import KMeans

# find all peaks
peaks, _ = find_peaks(projection)

# Get the height of the peak
peak_heights = projection[peaks]

# Group peaks using K-Means clustering algorithm
kmeans = KMeans(n_clusters=2, random_state=0).fit(peak_heights.reshape(-1, 1))

# Find the high and low peaks
high_peaks = peaks[kmeans.labels_ == 0]  # Assume that the high peak is in the 0th cluster
low_peaks = peaks[kmeans.labels_ == 1]  # Assume that the low peak is in cluster 1

print("high peak:", high_peaks)
print("low peak:", low_peaks)

# Use matplotlib to draw the distribution of white points along the x-axis and the first derivative curve
plt.figure()
plt.plot(projection, label='Projection')
plt.scatter(high_peaks, projection[high_peaks], color='blue', label='High peaks')  # Draw high peaks
plt.scatter(low_peaks, projection[low_peaks], color='red', label='Low peaks')  # plot low peaks
plt.title('Distribution of white points along x-axis')
plt.xlabel('x')
plt.ylabel('Number of white points')
plt.legend()
plt.show()
cv2.imshow('Detected Lines', th2)
cv2.waitKey()


