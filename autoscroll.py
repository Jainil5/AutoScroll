import cv2
import mediapipe as mp
import time
import mouse


cap = cv2.VideoCapture(0)
pTime = 0
l = []
m=[]


mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)
while True:
    
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms)
    
            for id, lm in enumerate(faceLms.landmark):
                ih, iw, ic = img.shape
                x, y = int(lm.x * iw), int(lm.y * ih)
                if id == 13:
                    y1 = y
                    x1 = x
                    l.append(y1)
                    m.append(x1)
                    time.sleep(0.2)
                    if len(l) >= 2:
                        a = l[-2]
                        b = l[-1]
                        c = m[-2]
                        d = m[-1]
                        if (a-b) >= 20:     
                            mouse.wheel(-3)
                            time.sleep(0.1)
                    
                        if(c-d)<=-20:
                            mouse.wheel(2)
                
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    if cv2.waitKey(5) & 0xFF == 27:
        break
