import cv2; 
[print(f'Device {i}: {"OK" if cv2.VideoCapture(i).isOpened() else "N/A"}') for i in range(5)]