import imageio
from PIL import Image,ImageOps 
import numpy as np
import pickle
import cv2
import math
n = 50
x1=[]

def permutation_single1(img,S):
    Mod=256
    img1=np.zeros([50,50,3])
    for j in range(int(400/n)):
        for i in range(int(400/n)):
            Dec1 = (np.matmul(S % Mod,img[i * n:(i + 1) * n, j * n:(j + 1) * n,0] % Mod)) % Mod
            Dec1 = np.matmul(S % Mod, np.transpose(Dec1)) % Mod
            Dec2 = (np.matmul(S % Mod,img[i * n:(i + 1) * n, j * n:(j + 1) * n,1] % Mod)) % Mod
            Dec2 = np.matmul(S % Mod, np.transpose(Dec2)) % Mod
            Dec3 = (np.matmul(S % Mod,img[i * n:(i + 1) * n, j * n:(j + 1) * n,2] % Mod)) % Mod
            Dec3 = np.matmul(S % Mod, np.transpose(Dec3)) % Mod
            
            Dec1 = np.resize(Dec1,(Dec1.shape[0],Dec1.shape[1],1))
            Dec2 = np.resize(Dec2,(Dec2.shape[0],Dec2.shape[1],1))
            Dec3 = np.resize(Dec3,(Dec3.shape[0],Dec3.shape[1],1))
            Dec = np.concatenate((Dec1,Dec2,Dec3), axis = 2)                #Dec = A * Enc
            img1[i * n:(i + 1) * n, j * n:(j + 1) * n] += Dec
            return img1
  
  
def XOR(x):
    w=x.shape[0]
    h=x.shape[1]
    for i in range(w):
        for j in range(h):
            
            y=x[i][j][0]
            # print(y)
            y=int(y)
            
            y1=y^24
            x[i][j][0]=y1
    return x   


      
def decrypt(x):
    global x1
    x1=x
    f=open('S.pkl','rb')
    S=pickle.load(f)
    f.close()
    w=x.shape[0]
    h=x.shape[1]
    #Parallel(n_jobs=4)(delayed(parallel2)(i, j,S,x) for j in range(h) for i in range(w))
    for i in range(w):
        for j in range(h):
            y=x[i][j]
            y1=XOR(y)
            y1=permutation_single1(y1,S)
            
            x[i][j]=y1
    return x


def unblockshaped(arr, h, w):
    #print(arr[0][0].shape)
    if arr.size != 0:
        img=np.zeros([h,w,3])
        l=0
    #print(img[0:50,0:50].shape)
        for i in range(0,h,50):
            m=0
            for j in range(0,w,50):
                img[i:i+50,j:j+50]=arr[l][m]
                m+=1
            l+=1  

        return img 

def decrypt1(img) :
    chopsize = 50

    #img = Image.open(infile).convert('RGB') 
    img = Image.fromarray(img).convert('RGB') 
    img=img.resize((400,400))
    
    #img=ImageOps.grayscale(img)
    width, height = img.size

    print(width,height)

    matrix_om=[]
    # Save Chops of original image
    r=0
    col=0
    for y0 in range(0, height, chopsize):
        r+=1
        col=0
        r_om=[]
        for x0 in range(0, width, chopsize):
            col+=1 
            box = (x0, y0,
                    x0+chopsize if x0+chopsize <  width else  width,
                    y0+chopsize if y0+chopsize < height else height)
            c1=img.crop(box)
            
            c2=np.array(c1)
            r_om.append(c2)
            # c1.save('chunks-1/%d-%d.jpg' % (r, col))
        matrix_om.append(r_om)

    matrix_om=np.array(matrix_om)
    #print(matrix_om.shape)
    x=decrypt(matrix_om)
    matrix_om=x

   # print('After chunking',str(matrix_om[0][0][0][0])) 
    print(matrix_om.shape)

    img=unblockshaped(matrix_om,400,400)
   # print(type(img))
    #img= cv2.cvtColor(img.astype('float32'), cv2.COLOR_RGB2BGR) 
    
    cv2.imwrite('r2.png',img)
    return img   


import glob
import cv2 as cv

import time




    

#start_time = time.time()


import os

#path = "C:/Users/lysias/Desktop/changed/frames/*.jpg"
path = "C:/Users/lysias/Desktop/Hill-Cipher---Image-Encryption-master/encrypted/*.png"
#img = cv2.imread('0.jpg') 
# frame = cv2.imread('0.jpg')
# frame=encrypt1(frame)
# cv2.write('1.jpg',frame)
if __name__=="__main__":
    start_time = time.time()
for index, filename in enumerate(glob.glob(path)):
    frame = cv2.imread(filename)
    #start_time = time.time()
    frame=decrypt1(frame)
#    blur = 'lis'
    basename = os.path.splitext(os.path.basename(filename))[0]
    #cv2.imwrite(f'{basename}_Gaussian{index}.png',frame)
    par = 'C:/Users/lysias/Desktop/Hill-Cipher---Image-Encryption-master/decrypted'
    cv2.imwrite(os.path.join(par , f'{basename}{index}.png'), frame)
    t=time.time() - start_time
    print("--- %s seconds ---" % (time.time() - start_time))