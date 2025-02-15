import cv2 as cv

img = cv.imread('open cv\any_Image.jpg')
img_grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


cc_wajah = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_alt.xml')
if cc_wajah.empty():
    print('Tidak bisa membuka file xml')

hasil = cc_wajah.detectMultiScale(img_grey, scaleFactor=1.1, minNeighbors=5)

print(f'jumlah wajah: {len(hasil)} list wajah: {hasil}')
x,y,w,h = hasil[0][0],hasil[0][1],hasil[0][2],hasil[0][3]
cv.rectangle(img, (x,y), (x+w,y+h), (255,125,255), thickness=5)
cv.imshow("foto",img)
cv.waitKey(0)
cv.destroyAllWindows()
