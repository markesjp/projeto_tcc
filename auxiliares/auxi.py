import cv2
import numpy as np

def converte_em_cinza(imagem):
    b, g, r = cv2.split(imagem)

    if np.count_nonzero(b) > 0:
        gray_image = b
    elif np.count_nonzero(g) > 0:
        gray_image = g
    elif np.count_nonzero(r) > 0:
        gray_image = r
    else:
        gray_image = np.zeros_like(b)
    
    return gray_image

