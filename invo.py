import imageio
from PIL import Image,ImageOps 
import numpy as np
import pickle
import cv2
import math
n = 50
x1=[]
def pr_matrix(n):
    
    #-------------Generating Encryption Key-------------
    Mod = 256
    k = 23                                                      #Key for Encryption

    d = np.random.randint(256, size = (int(n/2),int(n/2)))          #Arbitrary Matrix, should be saved as Key also
    I = np.identity(int(n/2))
    a = np.mod(-d,Mod)

    b = np.mod((k * np.mod(I - a,Mod)),Mod)
    k = np.mod(np.power(k,127),Mod)
    c = np.mod((I + a),Mod)
    c = np.mod(c * k, Mod)

    A1 = np.concatenate((a,b), axis = 1)
    A2 = np.concatenate((c,d), axis = 1)
    A = np.concatenate((A1,A2), axis = 0)
    Test = np.mod(np.matmul(np.mod(A,Mod),np.mod(A,Mod)),Mod) 
    print(A)
    return A

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

def permutation_single(img,S):
    Mod=256
    # print('called')
    img1=np.zeros([50,50,3])
    for j in range(int(400/n)):
        for i in range(int(400/n)):
            Enc1 = (np.matmul(S % Mod,img[i * n:(i + 1) * n, j * n:(j + 1) * n,0] % Mod)) % Mod
            Enc1 = np.matmul(S % Mod, np.transpose(Enc1)) % Mod
            Enc2 = (np.matmul(S % Mod,img[i * n:(i + 1) * n, j * n:(j + 1) * n,1] % Mod)) % Mod
            Enc2 = np.matmul(S % Mod, np.transpose(Enc2)) % Mod
            Enc3 = (np.matmul(S % Mod,img[i * n:(i + 1) * n, j * n:(j + 1) * n,2] % Mod)) % Mod
            Enc3 = np.matmul(S % Mod, np.transpose(Enc3)) % Mod        
            
            Enc1 = np.resize(Enc1,(Enc1.shape[0],Enc1.shape[1],1))
            Enc2 = np.resize(Enc2,(Enc2.shape[0],Enc2.shape[1],1))
            Enc3 = np.resize(Enc3,(Enc3.shape[0],Enc3.shape[1],1))
            img1[i * n:(i + 1) * n, j * n:(j + 1) * n] += np.concatenate((Enc1,Enc2,Enc3), axis = 2) 
    # print(img[49][49],'####')
            return img1 

def encrypt(x):
    global x1
    x1=x
    
    f=open('S.pkl','rb')
    S=pickle.load(f)
    f.close()
    w=x.shape[0]
    h=x.shape[1]
    # Parallel(n_jobs=4)(delayed(parallel1)(i, j,S) for j in range(h) for i in range(w))
   
    for i in range(w):
        for j in range(h):
            y=x[i][j]
            y1=permutation_single(y,S)
            y1=XOR(y1)
            x[i][j]=y1
    # print(x[0][0][0])
    # print(x1[0][0][0])
    # n/0
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
    
    
def encrypt1(img) :
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
    #print(matrix_om)
    x=encrypt(matrix_om)
    matrix_om=x

    #print('After chunking',str(matrix_om[0][0][0][0])) 
    print(matrix_om.shape)

    img=unblockshaped(matrix_om,400,400)
   # print(type(img))
    #img= cv2.cvtColor(img.astype('float32'), cv2.COLOR_RGB2BGR) 
    
    #cv2.imwrite('r1.png',img)
    return img


import glob
import cv2 as cv

import time


global S

S=pr_matrix(n)
f=open('S.pkl','wb')
pickle.dump(S,f)
f.close()
    

#start_time = time.time()


import os

path = "C:/Users/lysias/Desktop/Hill-Cipher---Image-Encryption-master/frames/*.jpg"
#img = cv2.imread('0.jpg') 
# frame = cv2.imread('0.jpg')
# frame=encrypt1(frame)
# cv2.write('1.jpg',frame)
if __name__=="__main__":
    start_time = time.time()
for index, filename in enumerate(glob.glob(path)):
    frame = cv2.imread(filename)
    #start_time = time.time()
    print("hello")
    frame=encrypt1(frame)
#    blur = 'lis'
    basename = os.path.splitext(os.path.basename(filename))[0]
    #cv2.imwrite(f'{basename}_Gaussian{index}.png',frame)
    par = 'C:/Users/lysias/Desktop/Hill-Cipher---Image-Encryption-master/encrypted'
    cv2.imwrite(os.path.join(par , f'{basename}_Gaussian{index}.png'), frame)
    t=time.time() - start_time
    print("--- %s seconds ---" % (time.time() - start_time))


