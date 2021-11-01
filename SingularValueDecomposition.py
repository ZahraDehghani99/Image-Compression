#Zahra Dehghanitafti(96222037)
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


path = input("please write your picture name :)") #input image
img = Image.open(path)
s = float(os.path.getsize(path))/1000 #calculate the size of the image
print("Size(dimension): ",img.size) #print size of the image
plt.title("Original Image (%0.2f Kb):" %s) #title of the image
plt.imshow(img)

imggray = img.convert('LA') #convert the RGB image into grayscale
imgmat = np.array( list(imggray.getdata(band = 0)), float) #convert the imggray into matrix
imgmat.shape = (imggray.size[1], imggray.size[0])
imgmat = np.matrix(imgmat)
print(imgmat)
plt.figure()
plt.imshow(imgmat, cmap = 'gray') #grayscale
plt.title("Image after converting it into the Grayscale pattern")
plt.show()

print("After compression: ")
U, S, Vt = np.linalg.svd(imgmat) #single value decomposition
for i in range(5, 70, 20):
    cmpimg = np.matrix(U[:, :i]) * np.diag(S[:i]) * np.matrix(Vt[:i,:]) #sum of the first i of Ui*Si*Vti(product of the matrices)
    plt.imshow(cmpimg, cmap = 'gray')
    title = " Image after =  {0} \n picture of sum of the first {1} of Ui*Si*Vti" .format(i,i)
    plt.title(title)
    plt.show()
    result = Image.fromarray((cmpimg ).astype(np.uint8))
result.save('compressed.jpg')