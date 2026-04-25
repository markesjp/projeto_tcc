import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_sauvola

# Carregar a imagem em escala de cinza
# Substitua 'sua_imagem.png' pelo caminho para a sua imagem
imagem = cv2.imread(r'C:\Users\jhon_\Documents\GitHub\Projeto-imagens\Resultados\3\1.jpg', cv2.IMREAD_GRAYSCALE)

# Verificar se a imagem foi carregada corretamente
if imagem is None:
    print("Erro ao carregar a imagem. Verifique o caminho e o nome do arquivo.")
    exit()

# Função para ajustar o contraste
def ajustar_contraste(imagem, kernel_size=3, fator_contraste=10):
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
    media = cv2.filter2D(imagem, -1, kernel)
    imagem_contraste = cv2.addWeighted(imagem, fator_contraste, media, -fator_contraste + 1, 0)
    return imagem_contraste

# Função para remover objetos pequenos
def remover_objetos(imagem_binaria, tamanho_min=0, tamanho_max=250):
    # Encontrar todos os componentes conectados (objetos brancos na imagem binária)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(imagem_binaria, connectivity=8)
    # Criar uma imagem de saída para armazenar os resultados
    imagem_saida = np.zeros_like(imagem_binaria)
    # Percorrer cada objeto detectado
    for i in range(1, num_labels):  # Começa em 1 para ignorar o fundo
        area = stats[i, cv2.CC_STAT_AREA]
        if tamanho_min <= area <= tamanho_max:
            # Ignora objetos dentro do intervalo especificado
            continue
        else:
            # Mantém objetos fora do intervalo (maiores que tamanho_max)
            imagem_saida[labels == i] = 255
    return imagem_saida

# Função para conectar falhas (morphological closing)
def conectar_falhas(imagem, kernel_size=15):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    imagem_conectada = cv2.morphologyEx(imagem, cv2.MORPH_CLOSE, kernel)
    return imagem_conectada

# --- Início do Pré-Processamento ---

# Lista para armazenar imagens intermediárias para plotagem
imagens_para_plotar = []

# Etapa 1: Ajuste de Contraste
#imagem_contraste = ajustar_contraste(imagem, kernel_size=3, fator_contraste=10)

# Etapa 2: Filtro Gaussiano
imagem_gaussiana = cv2.GaussianBlur(imagem, (19, 19), sigmaX=5, sigmaY=5)
imagens_para_plotar.append(('Após Filtro Gaussiano', imagem_gaussiana))

# Etapa 3: Binarização Sauvola
window_size = 41
k = 0.30
thresh_sauvola = threshold_sauvola(imagem_gaussiana, window_size=window_size, k=k)
imagem_binaria = (imagem_gaussiana > thresh_sauvola).astype(np.uint8) * 255

# Etapa 4: Remoção de Objetos Pequenos (0 a 250 pixels)
imagem_removida = remover_objetos(imagem_binaria, tamanho_min=0, tamanho_max=250)
imagens_para_plotar.append(('Após Remoção de Objetos (0-250 px)', imagem_removida))

# Etapa 5: Conectar Falhas
imagem_conectada = conectar_falhas(imagem_removida, kernel_size=15)

# Etapa 6: Bitwise NOT
imagem_invertida = cv2.bitwise_not(imagem_conectada)

# Etapa 7: Remoção de Objetos Pequenos (0 a 250 pixels)
imagem_removida_2 = remover_objetos(imagem_invertida, tamanho_min=0, tamanho_max=250)

# Etapa 8: Remoção de Objetos Pequenos (0 a 400 pixels)
imagem_removida_3 = remover_objetos(imagem_removida_2, tamanho_min=0, tamanho_max=400)

# Etapa 9: Bitwise NOT
imagem_invertida_2 = cv2.bitwise_not(imagem_removida_3)

# Etapa 10: Remoção de Objetos Pequenos (0 a 400 pixels)
imagem_removida_4 = remover_objetos(imagem_invertida_2, tamanho_min=0, tamanho_max=400)

# Etapa 11: Remoção de Objetos Pequenos (0 a 1000 pixels)
imagem_removida_5 = remover_objetos(imagem_removida_4, tamanho_min=0, tamanho_max=1000)

# Etapa 12: Bitwise NOT
imagem_invertida_3 = cv2.bitwise_not(imagem_removida_5)

# Etapa 13: Remoção de Objetos Pequenos (0 a 1000 pixels)
imagem_removida_6 = remover_objetos(imagem_invertida_3, tamanho_min=0, tamanho_max=1000)

# Etapa 14: Bitwise NOT (duas vezes)
imagem_invertida_4 = cv2.bitwise_not(imagem_removida_6)
imagem_invertida_5 = cv2.bitwise_not(imagem_invertida_4)

# Etapa 15: Remoção de Objetos Grandes (100.000 a 1.000.000 pixels)
imagem_removida_grandes = remover_objetos(imagem_invertida_5, tamanho_min=100000, tamanho_max=1000000)

# Etapa 16: Dilatação - Kernel: 3, Iterações: 3
kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
imagem_dilatada = cv2.dilate(imagem_removida_grandes, kernel_dilate, iterations=3)

# Etapa 17: Bitwise XOR
imagem_xor = cv2.bitwise_xor(imagem_removida_grandes, imagem_dilatada)

# Etapa 18: Conectar Falhas - Kernel: 3
imagem_conectada_2 = conectar_falhas(imagem_xor, kernel_size=3)

# Etapa 19: Ajuste de Contraste - Kernel: 13, Fator contraste: 4
#imagem_contraste_2 = ajustar_contraste(imagem_conectada_2, kernel_size=13, fator_contraste=4)

# Etapa 20: Filtro Gaussiano - Kernel: 31, Sigma: 4
imagem_gaussiana_2 = cv2.GaussianBlur(imagem_conectada_2, (31, 31), sigmaX=4, sigmaY=4)
imagens_para_plotar.append(('Após Filtro Gaussiano 2', imagem_gaussiana_2))

# Etapa 21: Binarização Sauvola - Kernel: 131, Threshold: 0
window_size = 131
k = 0.0
thresh_sauvola_2 = threshold_sauvola(imagem_gaussiana_2, window_size=window_size, k=k)
imagem_binaria_2 = (imagem_gaussiana_2 > thresh_sauvola_2).astype(np.uint8) * 255

# Etapa 22: Remoção de Objetos Pequenos (0 a 20.000 pixels)
imagem_removida_7 = remover_objetos(imagem_binaria_2, tamanho_min=0, tamanho_max=20000)
imagens_para_plotar.append(('Após Remoção de Objetos (0-20.000 px)', imagem_removida_7))

# Etapa 23: Bitwise NOT
imagem_invertida_6 = cv2.bitwise_not(imagem_removida_7)

# Etapa 24: Remoção de Objetos Pequenos (0 a 20.000 pixels)
imagem_removida_8 = remover_objetos(imagem_invertida_6, tamanho_min=0, tamanho_max=20000)

# Etapa 25: Bitwise NOT
imagem_invertida_7 = cv2.bitwise_not(imagem_removida_8)

# Etapa 26: Erosão - Kernel: 3, Iterações: 5
kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
imagem_erodida = cv2.erode(imagem_invertida_7, kernel_erode, iterations=5)
imagens_para_plotar.append(('Após Erosão', imagem_erodida))

# Etapa 27: Remoção de Objetos Pequenos (0 a 40.000 pixels)
imagem_removida_9 = remover_objetos(imagem_erodida, tamanho_min=0, tamanho_max=40000)

# Etapa 28: Dilatação - Kernel: 3, Iterações: 5
imagem_final = cv2.dilate(imagem_removida_9, kernel_erode, iterations=5)
imagens_para_plotar.append(('Imagem Final', imagem_final))

# --- Plotagem das Etapas Selecionadas ---

# Número de imagens para plotar
num_imagens = len(imagens_para_plotar)

# Configurar a figura para exibir todas as imagens selecionadas
fig, axs = plt.subplots(1, num_imagens + 1, figsize=(5 * (num_imagens + 1), 5))

# Imagem Original
axs[0].imshow(imagem, cmap='gray')
axs[0].set_title('Imagem Original')
axs[0].axis('off')

# Plotar as imagens intermediárias selecionadas
for idx, (titulo, img) in enumerate(imagens_para_plotar):
    axs[idx + 1].imshow(img, cmap='gray')
    axs[idx + 1].set_title(titulo)
    axs[idx + 1].axis('off')

plt.show()
