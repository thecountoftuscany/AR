#!python3
import numpy as np
import cv2

class AR:
    def __init__(self, src=0, scale=0.9):
        self.src = src
        self.cap = cv2.VideoCapture(src)
        self.scaleFactor = 0.9
        self.blur = 10
        self.blackThresh = 150

    def ImgPreprocess(self, image):
        frame = cv2.resize(image, (0,0), fx=self.scaleFactor, fy=self.scaleFactor) 
        if self.src == 0:
            frame = cv2.flip(image, 1)

        return frame

    def reverseThreshold(self, fblur):
        gray = cv2.cvtColor(fblur, cv2.COLOR_BGR2GRAY)
        ret,dst = cv2.threshold(gray, self.blackThresh ,255,cv2.THRESH_BINARY)
        dst = cv2.bitwise_not(dst)
        return dst

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
            frame = self.ImgPreprocess(frame) ; img2 = frame.copy()
            fblur = cv2.blur(frame,(self.blur,self.blur)) 
            
            # ****** Threshold based on brightness
            dst = self.reverseThreshold(fblur)

            # ****** Track using Camshift algo
            ret, track_window = cv2.CamShift(dst, track_window, term_crit)

            # ****** Draw bounding box from Camshift output
            pts = np.int0(cv2.boxPoints(ret))
            bb = self.boundingBox(pts)
            cv2.rectangle(img2, (bb[0],bb[1]), (bb[2],bb[3]), (0,255,0), 2)

            # ****** Detect largest contour inside bounding box, return coords of 4 vertices
            success, fourPt, _ = self.sqCoords( dst[bb[1]:bb[3], bb[0]:bb[2] ] )

            if success and fourPt.shape[0] == 4: # If contour with 4 pts is detected
                fourPt[:,0,0] += bb[0] ; fourPt[:,0,1] += bb[1]
                cv2.drawContours(img2 , [fourPt], -1, (0, 0, 255), 2) # draw approx contour with 4 vertices
                # Write text (labels) on vertices
                for i in range(4): cv2.putText(img2,str(i),(fourPt[i,0,0],fourPt[i,0,1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,255),1,cv2.LINE_AA)
                # Get centroid co-ords and label it
                origin = self.getOrigin( fourPt )
                if origin[0] != -1: cv2.circle(img2,origin, 2, (0,0,255), -1)
                # Get X,Y orientation and gridlines (Inaccurate)
                X,Y = self.getXY(fourPt) ; cv2.line(img2, origin, X, (255,255,0), 1) ; cv2.line(img2, origin, Y, (255,255,0), 1)

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
    
    def getOrigin(self, f):
        # Intersection point of diagonals
        try:
            tx = f[:,0,0].tolist() ; ty = f[:,0,1].tolist() 
            x = [ tx[0], tx[2],tx[1],tx[3] ] ; y = [ ty[0], ty[2],ty[1],ty[3] ] 
            m0 = ( y[1] - y[0] )/( x[1] - x[0] ) ; m2 = ( y[3] - y[2] )/( x[3] - x[2] )
            X = ( (y[2] - y[0]) - ( m2*x[2] - m0*x[0] ) ) / ( m0 - m2 ) 
            Y = m0*( X - x[0] ) + y[0]
            return ( int(X), int(Y) ) 
        except:
            return (-1, -1)

    def getXY(self,f):
        # This method is not correct actually as perspective not taken into account
        tx = f[:,0,0].tolist() ; ty = f[:,0,1].tolist()
        a = np.argsort(tx) 
        m,n = a[-1] , a[-2] # m is index of p1, n of p2 and k of p3
        p1 = (tx[m], ty[m] ) ; p2 = (tx[n], ty[n] ) 

        if n == 0 and m == 1: k = 3
        elif n == 0 and m == 3: k = 1
        elif n == 1 and m == 2: k = 0
        elif n == 1 and m == 0: k = 2
        elif n == 2 and m == 3: k = 1
        elif n == 2 and m == 1: k = 3
        elif n == 3 and m == 0: k = 2
        elif n == 3 and m == 2: k = 0
        
        p3 = (tx[k], ty[k] ) 

        X = ( int((p1[0] + p2[0])/2) , int((p1[1] + p2[1])/2) )
        Y = ( int((p2[0] + p3[0])/2) , int((p2[1] + p3[1])/2) )
        
        return X,Y


if __name__ == '__main__':
    a = AR(src="http://192.168.43.1:8080/video", scale=1.0)
    # a = AR()
    a.startDetection()
    a.close()

