import cv2 as cv
import numpy as np
import handtrackingmodule as htm

cap = cv.VideoCapture(0)
cap.set(3,1280)
cap.set(4, 720)
blank = np.zeros([720,1280], dtype='uint8')
blank = cv.putText(blank, 'Captured!', (150,430), cv.FONT_HERSHEY_TRIPLEX, 5.5, (255,255,255), thickness=50)

detector = htm.handDetector(maxHands = 1, detectionCon=0.88)
# xp,yp = 0,0

while True:
    
     # 1.import the image
    success, img = cap.read()
    img1 = cv.flip(img, 1)
    # img1 = img
    img = cv.flip(img, 1)
    
    
    # 2. find hand landmarks
    
    img1 = detector.findHands(img1)
    lmList = detector.findPosition(img1 , draw= False)
    
    if len(lmList)!=0:
        
        #print(lmList)
        
     # tip of index and middle finger   
        x1,y1= lmList[8][1:]
        x2,y2= lmList[12][1:]
        rx1, ry1 = lmList[7][1:]
        rx2, ry2 = lmList[11][1:]

        
    #3. check which fingers are up
    
        fingers = detector.fingersUp()
        
        
        if fingers[1] and fingers[2]:
            # xp,yp = 0,0
            # dx, img, _ = detector.findDistance(8, 7, img) 
            # dy, img, _ = detector.findDistance(12, 11, img)
            dx = np.sqrt((rx1-x1)*(rx1-x1) + (ry1-y1)*(ry1-y1))
            dy = np.sqrt((rx2-x2)*(rx2-x2) + (ry2-y2)*(ry2-y2))
            if y2 - y1 < dy and x2 - x1 < dx:
                
                Captured = cv.imwrite('D:/Rakshiiii/Pictures/NewP/junkcaptures/capture.png',img)

                if Captured:
                    print("Captured!")
                

                img1 =  cv.bitwise_and(img1,img1,mask=blank)
                cv.imshow("Image", img1)
                cv.waitKey(1000)

    cv.imshow("Image", img1)
    
    if cv.waitKey(2)  & 0xFF == ord('d'):
        break
cap.release()
cv.destroyAllWindows()