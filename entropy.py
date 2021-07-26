from skimage import measure    
import cv2

img=cv2.imread('decrypted.png')
entropy = measure.shannon_entropy(img)

print(entropy)