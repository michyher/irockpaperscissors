import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0)

# Create a window
cv2.namedWindow('Hand Gesture Recognition')

selected_choice = None  # Variable to store the selected choice

while True:
    try:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        kernel = np.ones((3, 3), np.uint8)
        
        roi = frame[200:800, 200:800]
        cv2.rectangle(frame, (200, 200), (800, 800), (0, 255, 0), 0)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        mask = cv2.dilate(mask, kernel, iterations=4)
        mask = cv2.GaussianBlur(mask, (5, 5), 100)
        
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            cnt = max(contours, key=lambda x: cv2.contourArea(x))
            epsilon = 0.0005 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            hull = cv2.convexHull(cnt)
            areahull = cv2.contourArea(hull)
            areacnt = cv2.contourArea(cnt)
            arearatio = ((areahull - areacnt) / areacnt) * 100
            
            hull = cv2.convexHull(approx, returnPoints=False)
            defects = cv2.convexityDefects(approx, hull)
            
            l = 0
            
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(approx[s][0])
                end = tuple(approx[e][0])
                far = tuple(approx[f][0])
                pt = (100, 180)
                
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                s = (a + b + c) / 2
                ar = math.sqrt(s * (s - a) * (s - b) * (s - c))
                
                d = (2 * ar) / a
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57
                
                if angle <= 90 and d > 30:
                    l += 1
                    cv2.circle(roi, far, 3, [255, 0, 0], -1)
                
                cv2.line(roi, start, end, [0, 255, 0], 2)
            
            l += 1
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            if l == 1:
                if areacnt < 2000:
                    cv2.putText(frame, 'Put hand in the box', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                    selected_choice = None  # Reset selected_choice if hand is not in the box
                else:
                    if cv2.waitKey(1) & 0xFF == ord(' '):  # Check if spacebar is pressed
                        selected_choice = 'rock'
            elif l == 2:
                cv2.putText(frame, '2 paper', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                if cv2.waitKey(1) & 0xFF == ord(' '):  # Check if spacebar is pressed
                    selected_choice = 'paper'
            elif l == 3:
                cv2.putText(frame, '3 scissors', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                if cv2.waitKey(1) & 0xFF == ord(' '):  # Check if spacebar is pressed
                    selected_choice = 'scissors'
            elif l == 4:
                cv2.putText(frame, '4', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                if cv2.waitKey(1) & 0xFF == ord(' '):  # Check if spacebar is pressed
                    selected_choice = '4'
            elif l == 5:
                cv2.putText(frame, '5', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                if cv2.waitKey(1) & 0xFF == ord(' '):  # Check if spacebar is pressed
                    selected_choice = '5'
        
        # Display the selected choice in the top right of the screen
        if selected_choice is not None:
            cv2.putText(frame, selected_choice, (frame.shape[1] - 250, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
        if cv2.waitKey(1) & 0xFF == ord('m'):  # Check if spacebar is pressed
            print('do logic to play agaistnt a robot')
        # Display frames in the window
        cv2.imshow('Hand Gesture Recognition', frame)
        
    except Exception as e:
        print("An error occurred:", str(e))
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

# Destroy the window and release the video capture
cv2.destroyAllWindows()
cap.release()
