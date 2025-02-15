import cv2 as cv 
model = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_alt.xml')
if model.empty():
    print('Tidak bisa membuka file xml')
    
video = cv.VideoCapture(0)
while True:
    ret, frame = video.read()
    frame = cv.resize(frame, (500, 500))
    frame_grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    hasil = model.detectMultiScale(frame_grey, scaleFactor=1.1, minNeighbors=5)
    for x,y,w,h in hasil:
        cv.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), thickness=5)
    for x,y,w,h in hasil:
        cv.putText(frame, 'Face', (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
    cv.imshow("hasil video",frame)
    tombol = cv.waitKey(1)
    if tombol & 0xFF == ord('q'):
        break
video.release()
cv.destroyAllWindows()