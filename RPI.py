from __future__ import print_function
import cv2 as cv
import argparse
import pytesseract as ts
import numpy as np
from PIL import Image
from pytesseract import image_to_string
import time
import math
import math as m
import serial as sr
import threading

###
ser = sr.Serial('/dev/ttyUSB0',9600)
###

global fpsc

ARDUINO_OK = 10
RASPBERRY_OK = 11
READ_SIGN = 19

threshold_arrow = 0.15
arrow = cv.imread('ok.png', cv.IMREAD_GRAYSCALE)
arrow = cv.inRange(arrow, 0, 100, dst = None)

###  FUNCTION DEFINITIONS

def is_arrow(thresholded_image):
    distance = cv.matchShapes(thresholded_image,arrow,cv.CONTOURS_MATCH_I2,0)
    if(distance<threshold_arrow):
        return True
    else:
        return False
        
def max_distance(arr,cX,cY):
    i=0
    max=0
    j=0
    for x,y in arr:
        distance=m.sqrt((x-cX)*(x-cX)+(y-cY)*(y-cY))
        if distance>max:
            max=distance
            i=j
        j += 1
    return arr[i][:]
fpsc = 0

def fps():
	global fpsc
	while True:
		#print(fpsc)
		fpsc = 0
		time.sleep(1)

def read_sign():
	a = threading.Thread(target= fps)
	a.start()
	while True:
		try:
			
			_, frame = cap.read()
			# time.sleep(0.5)
			# color convertion bgr to gray
			frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
			# gaussian blur
			frame = cv.GaussianBlur(frame, (3, 3), 2)

			# find circles
			circles = cv.HoughCircles(frame, cv.HOUGH_GRADIENT, 1, frame.shape[0] // 8, param1=150, param2=80,
									  minRadius=20, maxRadius=300)

			if circles is not None:

				largestCircle = circles[0][0]

				for i in circles[0, :]:
					#cv.circle(frame, (i[0], i[1]), i[2], (255), 5)

					if i[2] > largestCircle[2]:
						largestCircle = i

				# debug
				# print(largestCircle)

				insideOfCircle = cv.getRectSubPix(frame, (
				math.ceil(2 * largestCircle[2]), math.ceil(2 * largestCircle[2])),
												  (largestCircle[0], largestCircle[1]))
				result = cv.inRange(insideOfCircle, 0, 125, dst=None)
				tempimg = np.zeros((insideOfCircle.shape[0] + 100, insideOfCircle.shape[1] + 100), dtype=np.uint8)
				tempimg[50:-50, 50:-50] = result
				result = tempimg

				result = cv.medianBlur(result, 3, dst=None)
				kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
				result = cv.filter2D(result, -1, kernel)
				# print(is_arrow(result))
				if is_arrow(result):
					contours, _ = cv.findContours(result.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
					# finds corners

					# cv.imshow('grayscale',image)
					# cv.imshow('binary',binarized)
					# corners = cv.goodFeaturesToTrack(result, 20, 0.01, 20)
					dst = cv.cornerHarris(result, 2, 3, 0.04)
					dst = cv.dilate(dst, None)

					# Threshold for an optimal value, it may vary depending on the image.
					tf = dst > 0.1 * dst.max()
					result[dst > 0.1 * dst.max()] = 128
					y, x = np.where(tf == True)
					corners = []
					i = 0
					for x1 in x:
						corners.append([x1, y[i]])
						i = i + 1
					# print(corners)
					# for asd in corners:
					#   cv.circle(result, (asd[0][0],asd[0][1]), 1, [100, 0, 0], 4)
					# plt.imshow(img),plt.show()
					cX = 0
					cY = 0
					area = 0
					contours = sorted(contours, key=cv.contourArea)
					# for c in contours[0]:
					c = contours[-1]
					if (1):
						M = cv.moments(c)
						cX += M["m10"]
						cY += M["m01"]
						area += M["m00"]
					cX = int(cX / area)
					cY = int(cY / area)
					cv.drawContours(result, [c], -1, (0, 255, 0), 2)
					cv.circle(result, (cX, cY), 7, (0, 0, 0), -1)
					cv.circle(result, (int(result.shape[1] / 2), int(result.shape[0] / 2)), 7, (128, 0, 0), -1)
					cv.putText(result, "center", (cX - 20, cY - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
					pointy = []
					for i in corners:
						x, y = i

						if abs(cX - x) < 5 or abs(cY - y) < 5:
							pointy.append([x, y])
					#
					pointy = max_distance(pointy, cX, cY)
					#print(pointy)
					cv.circle(result, (pointy[0], pointy[1]), 4, [0, 0, 0], -1)
					cv.putText(result, "pointy", (pointy[0], pointy[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
							   2)
					pointy_X = pointy[0]
					pointy_Y = pointy[1]

					# print(cX, pointy_X, cY, pointy_Y)
					if abs(cX - pointy_X) < 5 and cY - pointy_Y < 0:
						cv.putText(result, "Down Arrow", (cX + 20, cY + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
						outputData = ''
						#print("Down arrow")
					elif abs(cX - pointy_X) < 5 and cY - pointy_Y > 0:
						cv.putText(result, "UP Arrow", (cX + 20, cY + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
						outputData = b's'
						#print("Up arrow")
					elif abs(cY - pointy_Y) < 5 and cX - pointy_X > 0:
						cv.putText(result, "left Arrow", (cX + 20, cY + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
						outputData = b'l'
						#print("Left arrow")
					elif abs(cY - pointy_Y) < 5 and cX - pointy_X < 0:
						cv.putText(result, "Right Arrow", (cX + 20, cY + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),
								   2)
						outputData = b'r'
						#print("Right arrow")
					else:
						print('none')

				else:
					#ts.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
					text = ts.image_to_string(result,
											  config="-c tessedit_char_whitelist=123AB tessedit_char_blacklist=il74567890zZ| --psm 6")
					print(text)
					
					if text == 'A1':
						outputData= b'a'
						break

					if text == 'A2':
						outputData= b'b'
						break

					if text == 'A3':
						outputData= b'c'
						break

					if text == 'B1':
						outputData= b'd'
						break

					if text == 'B2':
						outputData= b'e'
						break

					if text == 'B3':
						outputData= b'f'	
						break		

		except: pass			
				#cv.imshow('Result', result)

			#cv.imshow('Big Picture', frame)
	
	
	#outputData = A   ##write a type from  '### OUTPUT TYPES'
	
	ser.write(outputData)
	
	return outputData

### OUTPUT TYPES

#LEFT = b'L'
#RIGHT = b'R'
#FORWARD = b'F'
#A1 = b'1'
#A2 = b'2'
#A3 = b'3'
#B1 = b'4'
#B2 = b'5'
#B3 = b'6'

### Initialize 



	
while int.from_bytes(ser.read(1), byteorder = 'little') != ARDUINO_OK :
	print("Arduino is NOT ready!")

ser.write(b'r')

ser.reset_input_buffer()
ser.reset_output_buffer()

print("Initialized !")



while 1:
	
	# check if the READ_SIGN signal is here
	if( int.from_bytes(ser.read(1), byteorder = 'little') == READ_SIGN ):

		cap = cv.VideoCapture(0)

		#print(" Reading sign...")
        signRead = read_sign()		
	    #out = signRead.decode("utf-8")		
		#print("Serial Sent:" +  out)
		
	


