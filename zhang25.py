from PIL import Image
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from sewar.full_ref import uqi


def encrypte(img, encryption_key):
    encrypted_img = []
    for i in range(0,img.shape[0]):
        encrypted_row = []
        for j in range(0,img.shape[1]):
            encrypted_pixel = img[i][j]^encryption_key[i][j]
            encrypted_row.append(encrypted_pixel)
        encrypted_img.append(encrypted_row)
    
    return encrypted_img

def change_pixel2bin(image):
    bin_image = []
    for i in range(0,len(image)):
        bin_row = []
        for j in range(0,len(image[0])):
            dec_pixel = image[i][j]
            bin_str = str(bin(dec_pixel).lstrip('0b')).zfill(8)
            bin_pixel = list(map(int,bin_str))
            bin_row.append(bin_pixel)
        bin_image.append(bin_row)
    return bin_image

def change_pixel2dec(image):
    dec_image = []
    for i in range(0,len(image)):
        dec_row = []
        for j in range(0,len(image[0])):
            bin_pixel = image[i][j]
            dec_pixel = int(','.join(str(i) for i in bin_pixel))
            dec_row.append(dec_pixel)
        dec_image.append(dec_row)
    return dec_image

def select_Np(encrypted_img, dh_key, L):
    random.seed(dh_key)
    length = len(encrypted_img)
    Np = np.random.choice(length-1,20)  
    pick_pixel = encrypted_img
    pick_pixel = np.delete(pick_pixel,Np)
    print(pick_pixel[2])
    permuted_pixel = np.zeros((1,len(pick_pixel)),dtype=int)
    permuted_order = list(range(0,pick_pixel.shape[0]))
    np.random.shuffle(permuted_order)
    permuted_pixel.put(permuted_order, pick_pixel)
    print(permuted_pixel[0][permuted_order[2]])
    
    divided_groups = []
    least_num = math.floor(permuted_pixel.shape[1]/L)
    for i in range(0,least_num):
        divided_groups.append(permuted_pixel[0][i:i+L])
    divided_groups.append(permuted_pixel[0][least_num*L:permuted_pixel.shape[1]])
    return Np, divided_groups, permuted_order


def find_LSB(divided_group, M):
    LSB = []
    for p in divided_group:
        p_LSB = p[8 - M : 8]
        LSB.extend(p_LSB)
    return LSB

def generate_G(M, L, S, dh_key):
    I = np.mat(eye(M*L-S,M*L-S,dtype=int))
    random.seed(dh_key)
    Q = np.random.randint(1,size=(M*L-S,S))
    G = np.append(I,Q,axis=1)
    return G

def make_data(Np, encrpted_img, data):
    processed_data = []
    for p in Np:
        p = str(bin(p).lstrip('0b'))
        processed_data.append(p[7])
    data_list = data.tolist()
    for d in data_list:
        processed_data.append(d)
    return processed_data


def compress_img(LSB, G, data, divided_group, M):
    LSB_arr = np.array([LSB])
    compressed_img = (G * LSB_arr.T).tolist()
    compressed_img.append(data)
    for i in range(0, len(divided_group)):
        divided_group[i][0:8-M].extend(compressed_img[i*M:i*M+M])  
        divided_group[i] = int(''.join(str(j) for j in divided_group[i]),2)
    return divided_group

def make_para_info(L, S, M):
    L = bin(L).lstrip('0b')
    S = bin(S).lstrip('0b')
    M = bin(M).lstrip('0b')
    info = str(L)+str(S)+str(M)
    return info.zfill(20)

def embed_data(encrypted_img, dh_key, data, L, S, M):
    encrypted_arr = np.array(encrypted_img)
    processed_img = encrypted_arr.flatten()
    print(processed_img[0:10])
    compressed_img = []
    Np,divided_groups, permute_order = select_Np(processed_img, dh_key ,L)
    Np.sort()
    em_data = make_data(Np, processed_img, data)
    processed_groups = change_pixel2bin(divided_groups[0:len(divided_groups)-1])
    
    for i in range(0,len(processed_groups)):
        LSB = find_LSB(processed_groups[i], M)
        G = generate_G(M, L, S, dh_key)
        img_peice = compress_img(LSB, G, em_data[i*S:(i+1)*S], processed_groups[i], M)
        compressed_img.append(img_peice)
    compressed_arr = np.array(compressed_img).flatten().tolist()
    compressed_arr.extend(divided_groups[len(divided_groups)-1])
    inversed_img = np.zeros((1,len(compressed_arr)),dtype=int)
    
    for i in range(0,len(compressed_arr)):
        
        inversed_img[0][i] = compressed_arr[permute_order[i]]
    parameter_info = make_para_info(L,S,M)
    print(inversed_img[0][0:10])
    embeded_img = inversed_img
    for i in range(0,len(Np)):
        temp = bin(processed_img[Np[i]]).lstrip('0b')
        embed = str(temp)[0:6] + str(parameter_info[i])
        dec_embed = int(embed)
        embeded_img = np.insert(embeded_img, Np[i], dec_embed)
    embeded_arr = np.array(embeded_img)
    print(embeded_arr.shape)
    embeded_img = embeded_arr.reshape(encrypted_arr.shape[0],encrypted_arr.shape[1]).tolist()
    return embeded_img

def get_PSNR(origin_img, encrypted_img):
    origin_data = np.array(origin_img)
    encrypted_data = np.array(encrypted_img)
    diff = origin_data - encrypted_data
    #diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff**2.))
    return 20*math.log10(1.0/rmse)
    
def get_Q(origin_img, encrypted_img):
    return uqi(origin_img, encrypted_img)


def decipher_img(encryption_key, encrypted_img):
    deciphered_img = []
    for i in range(0,encrypted_img.shape[0]):
        decipher_row = []
        for j in range(0,encrypted_img.shape[1]):
            deciphered_pixel = encrypted_img[i][j]^encryption_key[i][j]
            decipher_row.append(deciphered_pixel)
        deciphered_img.append(decipher_row)
    return deciphered_img

img = Image.open('D:/temp.bmp').convert('L')
img_arr=np.array(img)
encryption_key = np.random.randint(256,size=(img_arr.shape[0],img_arr.shape[1]))
encrypted_img = encrypte(img_arr,encryption_key)
encrypted_arr = np.array(encrypted_img)
'''
en_img = Image.fromarray(encrypted_arr)
en_img.show()
'''
dh_key = random.randint(2000,7890)

L = 2000
S =4
data = np.random.randint(1,size=(math.floor((img_arr.shape[0]*img_arr.shape[1]-20)*S/L-20),1))

embeded_img = embed_data(encrypted_img, dh_key,data, L, S, 3)
embeded_arr = np.array(embeded_img)
embed_img = Image.fromarray(embeded_arr)
embed_img.show()
deciphered_img = decipher_img(embeded_img, encryption_key)

deciphered_arr = np.array(deciphered_img)
new_img = Image.fromarray(deciphered_arr)
new_img.show()

'''

psnr_list = []
for S in range(1,7):
    data = np.mat(random.randint(1,size=(1,(img.shape[0]*img.shape[1]-20)*S/L-20)))
    embeded_img = embed_data(encrypted_img, dh_key,data, L, S, 3)
    deciphered_img = decipher_img(embeded_img)
    dec_decipher = 
    psnr = get_PSNR(img,deciphered_img)
    psnr_list.append(psnr)
'''

    
    