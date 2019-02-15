#!python3
import numpy as np
import cv2

class AR:
    def __init__(self, src=0, scale=0.9):
        self.src = src
        self.cap = cv2.VideoCapture(src)
        self.scaleFactor = 0.9
        self.blur = 10

    def ImgPreprocess(self, image):
        frame = cv2.resize(image, (0,0), fx=self.scaleFactor, fy=self.scaleFactor) 
        if self.src == 0:
            frame = cv2.flip(image, 1)

        return frame

    def quit(self):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return True
        else:
            return False
    
    def show(self, imgs):
        if len(imgs) == 1:
            cv2.imshow('frame', imgs[0])
        else:
            final = imgs[0]
            for i in np.arange(1,len(imgs)):
                if len(imgs[i].shape) == 2: # 1 Channel image
                    temp = cv2.merge((imgs[i], imgs[i], imgs[i] ))
                    final = np.hstack((final,temp))
                else:
                    final = np.hstack((final,imgs[i]))

            cv2.imshow('frame',final)
    
    def calibrate(self):
        print("Starting calibration. Press A,S to change the size of the window. Press D to proceed.")
        i = 0
        while True:
            ret, frame = self.cap.read()
            frame = self.ImgPreprocess(frame)
            drawImage = frame.copy()

            if i == 0:
                #Co-ordinates
                R,C,Ch = frame.shape
                xC = C/2 ; yC = R/2 # Centre of image
                H = 100 ; W = 100
                i = 1
            
            x1 = int(xC - W/2) ; x2 = int(xC + W/2)
            y1 = int(yC - H/2) ; y2 = int(yC + H/2)
            c = x1 ; r = y1


            cv2.rectangle(drawImage, (x1,y1), (x2,y2), (255,0,0), 2)
            self.show([ drawImage ])

            k = cv2.waitKey(10)
            if k == ord('a'):
                H += 10 ; W += 10 
            elif k == ord('s') and H > 30 and W > 30 :
                H -= 10 ; W -= 10
            elif k == ord('d'):
                frame = cv2.blur(frame,(self.blur,self.blur)) 
                roi = frame[r:r+H, c:c+W]
                hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
                roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
                cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
                break
        
        # Wrap up
        cv2.destroyAllWindows()
        return (x1, y1, W, H), roi_hist # (c,r,w,h)

    def startDetection(self):
        term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
        track_window, roi_hist = self.calibrate()

        while True:
            ret, frame = self.cap.read()
            frame = self.ImgPreprocess(frame)
            fblur = cv2.blur(frame,(self.blur,self.blur)) 
            
            # Grayscale thresh
            gray = cv2.cvtColor(fblur, cv2.COLOR_BGR2GRAY)
            ret,dst = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
            dst = cv2.bitwise_not(dst)

            # Camshift tracking
            ret, track_window = cv2.CamShift(dst, track_window, term_crit)
            # Draw tracking window
            pts = cv2.boxPoints(ret) ; pts = np.int0(pts)
            img2 = frame.copy()
            # img2 = cv2.polylines(frame,[pts],True, 255,2)
            # Draw bounding box in green
            bb = self.boundingBox(pts)
            cv2.rectangle(img2, (bb[0],bb[1]), (bb[2],bb[3]), (0,255,0), 2)
            # Largest contour inside bounding box
            success, fourPt, box = self.sqCoords( dst[bb[1]:bb[3], bb[0]:bb[2] ] )

            if success and fourPt.shape[0] == 4:
                box[:, 0] += bb[0] ; box[:, 1] += bb[1] 
                cv2.drawContours(img2,[box],0,(255,0,0),2) # Min Area rect
                fourPt[:,0,0] += bb[0] ; fourPt[:,0,1] += bb[1]
                cv2.drawContours(img2 , [fourPt], -1, (0, 0, 255), 2) # Approximated contour with 4 vertices

            self.show([img2,dst])

            if self.quit(): break
        
    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def boundingBox(self, pts):
        x1 = np.min(pts[:,0]) ; x2 = np.max(pts[:,0])
        y1 = np.min(pts[:,1]) ; y2 = np.max(pts[:,1]) 

        return x1,y1,x2,y2

    def sqCoords(self,subImage):
        _, contours, _ = cv2.findContours(subImage, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        try:
            c = max(contours, key=cv2.contourArea)
            box = cv2.boxPoints( cv2.minAreaRect(c) )
            box = np.int0(box)

            epsilon = 0.1*cv2.arcLength(c,True)
            approx = cv2.approxPolyDP(c,epsilon,True)

            return 1, approx, box
        except:
            return 0, None, None


if __name__ == '__main__':
    a = AR(src="http://192.168.1.6:8080/video", scale=1.0)
    # a = AR()
    a.startDetection()
    a.close()

