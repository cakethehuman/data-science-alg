import cv2 as cv

foto = cv.imread('open cv\souce.jpeg')
foto_grey = cv.cvtColor(foto, cv.COLOR_BGR2GRAY)
cc_wajah = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_alt.xml')
if cc_wajah.empty():
    print('Tidak bisa membuka file xml')


hasil = cc_wajah.detectMultiScale(foto_grey, scaleFactor=1.1, minNeighbors=5)

for x,y,w,h in hasil:
    cv.rectangle(foto, (x,y), (x+w,y+h), (255,125,255), thickness=5)

print(f'jumlah wajah: {len(hasil)} list wajah: {hasil}')
cv.imshow("hasil foto",foto)
cv.waitKey(0)
cv.destroyAllWindows()