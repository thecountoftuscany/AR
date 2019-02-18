#!python3
import numpy as np
import cv2

class AR:
    def __init__(self, src=0, scale=0.9):
        self.src = src
        self.cap = cv2.VideoCapture(src)
        self.scaleFactor = scale # Only used while drawing image on screen
        self.blur = 10
        self.blackThresh = 150

    def reverseThresholdImage(self, image):
        grayscaleImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, maskedImage = cv2.threshold(grayscaleImage, self.blackThresh ,255,cv2.THRESH_BINARY)
        maskedImage = cv2.bitwise_not(maskedImage)
        return maskedImage

    def quit(self):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return True
        else:
            return False

    def show(self, imgs, windowTitle):
        # Scale while drawing so that image fits on screen
        scaledImgs = imgs
        for i in range(0,len(imgs)):
            scaledImgs[i] = cv2.resize(imgs[i], (0,0), fx=self.scaleFactor, fy=self.scaleFactor)
            # Flip if webcam
            if self.src == 0:
                scaledImgs[i] = cv2.flip(imgs[i], 1)

        # If single image received, draw it
        if len(scaledImgs) == 1:
            cv2.imshow('AR', scaledImgs[0])
        # If multiple images received, draw them side-by-side
        else:
            compositeImage = scaledImgs[0]
            for i in np.arange(1,len(scaledImgs)):
                if len(scaledImgs[i].shape) == 2: # 1 Channel image
                    threeChannelImg = cv2.merge((scaledImgs[i], scaledImgs[i], scaledImgs[i] ))
                    compositeImage = np.hstack((compositeImage,threeChannelImg))
                else:
                    compositeImage = np.hstack((compositeImage,scaledImgs[i]))
            cv2.imshow(windowTitle,compositeImage)

    def getTrackingBlob(self):
        print("Starting calibration. Press A,S to change the size of the window. Press D to proceed.")
        flag = 0
        while True:
            _, image = self.cap.read()

            # Drawing and modifications done on a copy, original frame available if required later
            drawnImage = image.copy()

            # Detection box (blue) dimensions
            if flag == 0:
                imageRows, imageCols, imageCh = image.shape
                detectionCenter_x = imageCols/2 ; detectionCenter_y = imageRows/2
                detectionH = 30 ; detectionW = 30
                flag = 1

            # Detection box top left corner
            detectionCorner1_x = int(detectionCenter_x - detectionW/2)
            detectionCorner1_y = int(detectionCenter_y - detectionH/2)
            # Detection box bottom right corner
            detectionCorner2_x = int(detectionCenter_x + detectionW/2)
            detectionCorner2_y = int(detectionCenter_y + detectionH/2)
            # Detection box top left row, column
            detectionCol = detectionCorner1_x ; detectionRow = detectionCorner1_y
            # Draw the box
            cv2.rectangle(drawnImage, (detectionCorner1_x, detectionCorner1_y), (detectionCorner2_x,detectionCorner2_y), (255,0,0), 5)
            self.show([ drawnImage ], "Set square within tracking box")

            # Resize detection box or start detection
            key = cv2.waitKey(10)
            if key == ord('s'):
                detectionH += 50 ; detectionW += 50
            elif key == ord('a') and detectionH > 70 and detectionW > 70 :
                detectionH -= 50 ; detectionW -= 50
            elif key == ord('d'):
                break

        # Wrap up
        cv2.destroyAllWindows()
        return (detectionCorner1_x, detectionCorner1_y, detectionW, detectionH) # (c,r,w,h)

    def startDetection(self):
        trackWindow = self.getTrackingBlob() # Track the blob in this window

        while True:
            _, image = self.cap.read()
            drawnImage = image.copy()
            blurredImage = cv2.blur(image,(self.blur,self.blur))

            # ****** Threshold based on brightness
            maskedImage = self.reverseThresholdImage(blurredImage)

            # ****** Track using Camshift algo
            term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
            trackBox, trackWindow = cv2.CamShift(maskedImage, trackWindow, term_crit)

            # ****** Find bounding box from Camshift output
            boundingBoxPoints = np.int0(cv2.boxPoints(trackBox))
            bb = self.boundingBox(boundingBoxPoints)
            # cv2.rectangle(img2, (bb[0],bb[1]), (bb[2],bb[3]), (0,255,0), 2)

            # ****** Detect largest contour inside bounding box, return coords of 4 vertices
            success, detectedVertices = self.sqCoords( maskedImage[ bb[1]:bb[3], bb[0]:bb[2] ] )
            if success and len(detectedVertices) == 4:
                squareX=detectedVertices[:,0,0]; squareY=detectedVertices[:,0,1]
            else:
                squareX=[-1,-1,-1,-1]; squareY=[-1,-1,-1,-1]

            # Print coordinates of detected square
            print([squareX[0],squareY[0]], [squareX[1],squareY[1]], [squareX[2],squareY[2]], [squareX[3],squareY[3]])

            # If contour with 4 pts is detected
            if success and detectedVertices.shape[0] == 4:
                detectedVertices[:,0,0] += bb[0] ; detectedVertices[:,0,1] += bb[1]
                # Draw approx contour with 4 vertices
                cv2.drawContours(drawnImage , [detectedVertices], -1, (0, 0, 255), 5)
                # Get centroid co-ords and label it
                centroid = self.getCentroid( detectedVertices )
                if centroid[0] != -1: cv2.circle(drawnImage ,centroid, 5, (0,0,255), -1)
                # Get ordered vertices starting from Ref Vertex and clockwise
                orderedVertices = self.correctVertexOrder(detectedVertices, centroid, maskedImage)
                cv2.circle(drawnImage ,orderedVertices[0], 10, (0,255,0), -1)


            self.show([drawnImage ,maskedImage], "Square detection")

            if self.quit(): break

    def correctVertexOrder(self, detectedVertices, centroid, maskedImage):
        # First detect the reference vertex
        # Divide sq in 4 quadrants. Find the one which has least no of white pixels. Return corresponding vertex
        vertices = detectedVertices[:,0,:]
        # Divide square in 4 parts
        imgArray = []
        for i in range(4):
            pPrev, pCurrent, pNext = vertices[i-2,:], vertices[i-1, :], vertices[i, :] 
            midpointPC = [ (pPrev[0]+pCurrent[0])/2, (pPrev[1]+pCurrent[1])/2  ]
            midpointCN = [ (pNext[0]+pCurrent[0])/2, (pNext[1]+pCurrent[1])/2  ]
            ptsOriginal = np.float32([ [centroid[0], centroid[1]], midpointPC, pCurrent, midpointCN ])
            ptsNew = np.float32([[0,0],[300,0],[300,300],[0,300]])
            M = cv2.getPerspectiveTransform(ptsOriginal,ptsNew)
            straightImage = cv2.warpPerspective(maskedImage,M,(300,300))
            imgArray.append(straightImage)
        
        indexOfDotImage = np.argmin([ np.count_nonzero(straightImage) for straightImage in imgArray])
        i = indexOfDotImage - 1 # Index of Reference vertex
        
        # Ordering should be clockwise
        if i == 0:
            if self.isClockWise(centroid, vertices[0,:], vertices[1,:] ):
                order = [0,1,2,3]
            else:
                order = [0,3,2,1]
        elif i == 1:
            if self.isClockWise(centroid, vertices[1,:], vertices[2,:] ):
                order = [1,2,3,0]
            else:
                order = [1,0,3,2]
        elif i == 2:
            if self.isClockWise(centroid, vertices[2,:], vertices[3,:] ):
                order = [2,3,0,1]
            else:
                order = [2,1,0,3]
        elif i == 3 or i == -1:
            if self.isClockWise(centroid, vertices[3,:], vertices[0,:] ):
                order = [3,0,1,2]
            else:
                order = [3,2,1,0]

        # Vertices in clockwise order starting from Reference Vertex
        orderedVertices = [ (vertices[i,0], vertices[i,1] ) for i in order ]
        
        return orderedVertices

    def isClockWise(self,o,m,n):
        OA = [m[0] - o[0], m[1] - o[1], 0 ]
        OB = [n[0] - o[0], n[1] - o[1], 0 ]

        Z = np.cross(OA,OB)
        return Z[2] < 0

    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def boundingBox(self, pts):
        x1 = np.min(pts[:,0]) ; x2 = np.max(pts[:,0])
        y1 = np.min(pts[:,1]) ; y2 = np.max(pts[:,1])
        return x1,y1,x2,y2

    def sqCoords(self,subImage):
        # findContours() has 3 return values in openCV 2.x and 2 values after 3.x
        if int(cv2.__version__[0]) > 3:
            contours, _ = cv2.findContours(subImage, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        else:
            _, contours, _ = cv2.findContours(subImage, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        try:
            c = max(contours, key=cv2.contourArea)
            epsilon = 0.1*cv2.arcLength(c,True)
            approx = cv2.approxPolyDP(c,epsilon,True)
            return 1, approx
        except:
            return 0, [None]

    def getCentroid(self, f):
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

if __name__ == '__main__':
    with open(".camIP","r") as camIPFile:
        cameraIP = camIPFile.read()
    a = AR(src="http://" + str(cameraIP) + ":8080/video", scale=0.9)
    a.startDetection()
    a.close()
