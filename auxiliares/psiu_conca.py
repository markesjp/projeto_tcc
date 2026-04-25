import cv2
import numpy as np
import math
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def aplica_blur_mediana(imagem, tamanho_kernel=7):
    return cv2.medianBlur(imagem, tamanho_kernel)

def binariza_imagem(imagem, limiar=127):
    _, imagem_binaria = cv2.threshold(imagem, limiar, 255, cv2.THRESH_BINARY)
    return imagem_binaria

def aplica_morfologia(imagem, tamanho_kernel=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tamanho_kernel, tamanho_kernel))
    return cv2.morphologyEx(imagem, cv2.MORPH_CLOSE, kernel)

def detecta_concavidades(contorno, profundidade_minima=200):
    if len(contorno) < 4:
        return []
    casco = cv2.convexHull(contorno, returnPoints=False)
    if casco is None or len(casco) < 4:
        return []
    pontos_concavos = []
    try:
        defeitos = cv2.convexityDefects(contorno, casco)
        if defeitos is not None:
            for i in range(defeitos.shape[0]):
                _, _, ponto_afastado_idx, profundidade = defeitos[i, 0]
                if profundidade > profundidade_minima:
                    ponto_afastado = tuple(contorno[ponto_afastado_idx][0])
                    pontos_concavos.append(ponto_afastado)
    except cv2.error as e:
        print(f"Erro ao calcular convexityDefects: {e}")
    return pontos_concavos

def encontra_contornos(imagem):
    contornos, _ = cv2.findContours(imagem, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contornos

def agrupa_pontos_concavidades(pontos_concavidades, distancia_cluster=15):
    if not pontos_concavidades:
        return []
    pontos_array = np.array(pontos_concavidades)
    clustering = DBSCAN(eps=distancia_cluster, min_samples=1).fit(pontos_array)
    labels = clustering.labels_
    pontos_consolidados = []
    for label in set(labels):
        indices = np.where(labels == label)[0]
        grupo_pontos = pontos_array[indices]
        centroide = np.mean(grupo_pontos, axis=0)
        pontos_consolidados.append(tuple(centroide.astype(int)))
    return pontos_consolidados

def calcula_focos_elipse(elipse):
    (x_centro, y_centro), (eixo_maior, eixo_menor), angulo = elipse
    a = eixo_maior / 2.0
    b = eixo_menor / 2.0
    if a < b:
        a, b = b, a
        angulo += 90
    c = math.sqrt(a**2 - b**2)
    angulo_rad = np.deg2rad(angulo)
    dx = c * np.cos(angulo_rad)
    dy = c * np.sin(angulo_rad)
    foco1 = (int(x_centro + dx), int(y_centro + dy))
    foco2 = (int(x_centro - dx), int(y_centro - dy))
    return foco1, foco2, c

def calcula_pontos_modificados(elipse):
    (x_centro, y_centro), (eixo_maior, eixo_menor), angulo = elipse
    a = eixo_maior / 2.0
    b = eixo_menor / 2.0
    if a < b:
        a, b = b, a
        angulo += 90
    c = math.sqrt(a**2 - b**2)
    angulo_rad = np.deg2rad(angulo)
    dx = c * np.cos(angulo_rad)
    dy = c * np.sin(angulo_rad)
    foco1 = (int(x_centro + dx), int(y_centro + dy))
    foco2 = (int(x_centro - dx), int(y_centro - dy))
    ponto_modificado_1 = (int((x_centro + foco1[0]) / 2), int((y_centro + foco1[1]) / 2))
    ponto_modificado_2 = (int((x_centro + foco2[0]) / 2), int((y_centro + foco2[1]) / 2))
    return ponto_modificado_1, ponto_modificado_2, foco1, foco2

def ponto_mais_proximo_no_segmento(ponto, linha_p1, linha_p2):
    linha = np.array(linha_p2) - np.array(linha_p1)
    comprimento_linha = np.linalg.norm(linha)
    linha_norm = linha / comprimento_linha if comprimento_linha != 0 else linha
    vetor_ponto = np.array(ponto) - np.array(linha_p1)
    projecao = np.dot(vetor_ponto, linha_norm)
    projecao_limitada = max(0, min(projecao, comprimento_linha))
    ponto_projetado = np.array(linha_p1) + projecao_limitada * linha_norm
    return int(ponto_projetado[0]), int(ponto_projetado[1])

def divide_imagem_em_subimagens(imagem, tamanho_subimagem, sobreposicao=0.5):
    h, w = imagem.shape[:2]
    passo = int(tamanho_subimagem * (1 - sobreposicao))
    subimagens = []
    for y in range(0, h - tamanho_subimagem + 1, passo):
        for x in range(0, w - tamanho_subimagem + 1, passo):
            subimg = imagem[y:y + tamanho_subimagem, x:x + tamanho_subimagem]
            subimagens.append((subimg, x, y))
    return subimagens

def salva_imagem(imagem, caminho_saida):
    cv2.imwrite(caminho_saida, imagem)

def exibe_imagem(imagem, titulo="Imagem"):
    plt.figure(figsize=(10, 10))
    if len(imagem.shape) == 2:  # Imagem em escala de cinza
        plt.imshow(imagem, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
    plt.title(titulo)
    plt.axis('off')
    plt.show()

def dilata_imagem(imagem, tamanho_kernel=5, iteracoes=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tamanho_kernel, tamanho_kernel))
    imagem_dilatada = cv2.dilate(imagem, kernel, iterations=iteracoes)
    return imagem_dilatada

def estende_linha(ponto_inicio, ponto_fim, comprimento_extensao=50):
    """
    Estende uma linha a partir do ponto_fim na mesma direção por um comprimento especificado.

    Parâmetros:
    - ponto_inicio: tupla (x, y) representando o início da linha.
    - ponto_fim: tupla (x, y) representando o fim da linha.
    - comprimento_extensao: comprimento para estender a linha.

    Retorna:
    - Tupla (novo_ponto_fim_x, novo_ponto_fim_y) representando o novo ponto fim estendido.
    """
    dx = ponto_fim[0] - ponto_inicio[0]
    dy = ponto_fim[1] - ponto_inicio[1]
    distancia = math.hypot(dx, dy)
    if distancia == 0:
        return ponto_fim  # Sem extensão se os pontos são iguais
    escala = comprimento_extensao / distancia
    novo_x = int(ponto_fim[0] + dx * escala)
    novo_y = int(ponto_fim[1] + dy * escala)
    return (novo_x, novo_y)

def preenche_interior(imagem_binaria):
    """
    Preenche o interior do maior contorno encontrado na imagem binária.

    Parâmetros:
    - imagem_binaria: imagem binária (preto e branco).

    Retorna:
    - imagem_preenchida: imagem binária com o interior preenchido.
    """
    contornos, _ = cv2.findContours(imagem_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        print("Nenhum contorno encontrado para preencher.")
        return imagem_binaria
    contorno_principal = max(contornos, key=cv2.contourArea)
    imagem_preenchida = imagem_binaria.copy()
    cv2.drawContours(imagem_preenchida, [contorno_principal], -1, 255, thickness=cv2.FILLED)
    return imagem_preenchida

def processa_imagem(imagem, limiar=127, tamanho_subimagem=50, sobreposicao=0.5, 
                   profundidade_minima=1000, distancia_cluster=15, comprimento_extensao=50,
                   caminho_saida="saida.jpg",
                   caminho_saida_segunda="saida_dilatada.jpg"):
    """
    Processa a imagem para identificar concavidades e traçar linhas desde os pontos de concavidade até a linha do meio,
    estendendo-as a partir dos pontos de concavidade. Retorna duas imagens: uma com as linhas na imagem original
    e outra com as linhas na imagem dilatada.

    Parâmetros:
    - imagem: imagem de entrada (BGR ou escala de cinza).
    - limiar: valor de limiar para binarização.
    - tamanho_subimagem: tamanho das subimagens para processamento.
    - sobreposicao: sobreposição entre subimagens.
    - profundidade_minima: profundidade mínima para considerar uma concavidade.
    - distancia_cluster: distância máxima para agrupamento de concavidades.
    - comprimento_extensao: comprimento para estender as linhas a partir da concavidade.
    - caminho_saida: caminho para salvar a imagem processada original com linhas.
    - caminho_saida_segunda: caminho para salvar a imagem dilatada com linhas estendidas.

    Retorna:
    - imagem_resultado: imagem original com linhas traçadas.
    - imagem_dilatada_com_linhas: imagem dilatada com linhas estendidas.
    """
    # Converter para escala de cinza se necessário
    if len(imagem.shape) == 3:
        imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    else:
        imagem_cinza = imagem.copy()

    # Aplicar blur mediana
    imagem_blur = aplica_blur_mediana(imagem_cinza, 7)

    # Binarizar a imagem
    imagem_binaria = binariza_imagem(imagem_blur, limiar)

    # Preencher o interior do objeto na imagem binária
    imagem_preenchida = preenche_interior(imagem_binaria)

    # Aplicar morfologia na imagem preenchida
    imagem_morfologica = aplica_morfologia(imagem_preenchida, 7)

    # Encontrar contornos na imagem morfologicamente processada
    contornos = encontra_contornos(imagem_morfologica)
    if not contornos:
        print("Nenhum contorno encontrado após preenchimento e morfologia.")
        return imagem, None

    # Selecionar o contorno principal
    contorno_principal = max(contornos, key=cv2.contourArea)
    if len(contorno_principal) < 5:
        print("Não é possível ajustar uma elipse ao contorno.")
        return imagem, None

    # Ajustar uma elipse ao contorno principal
    try:
        elipse = cv2.fitEllipse(contorno_principal)
    except cv2.error as e:
        print(f"Erro ao ajustar elipse: {e}")
        return imagem, None

    # Calcular pontos modificados (focos)
    ponto_modificado_1, ponto_modificado_2, foco1, foco2 = calcula_pontos_modificados(elipse)

    # Copiar a imagem original para desenhar as linhas (sem preenchimento)
    imagem_resultado = dilata_imagem(imagem_resultado, 5, 1)

    # Dividir a imagem morfologicamente processada em subimagens
    subimagens = divide_imagem_em_subimagens(imagem_morfologica, tamanho_subimagem, sobreposicao)

    todas_concavidades = []
    for subimg, deslocamento_x, deslocamento_y in subimagens:
        contornos_subimg = encontra_contornos(subimg)
        for contorno in contornos_subimg:
            pontos_concavos = detecta_concavidades(contorno, profundidade_minima)
            pontos_ajustados = [(p[0] + deslocamento_x, p[1] + deslocamento_y) for p in pontos_concavos]
            todas_concavidades.extend(pontos_ajustados)

    if todas_concavidades:
        pontos_consolidados = agrupa_pontos_concavidades(todas_concavidades, distancia_cluster)
        linhas = []
        for ponto in pontos_consolidados:
            ponto_projetado = ponto_mais_proximo_no_segmento(ponto, ponto_modificado_1, ponto_modificado_2)
            cv2.line(imagem_resultado, ponto, ponto_projetado, (0, 0, 0), 2)
            linhas.append((ponto, ponto_projetado))
    else:
        print("Nenhuma concavidade detectada.")
        linhas = []

    salva_imagem(imagem_resultado, caminho_saida)

    imagem_dilatada = dilata_imagem(imagem_binaria, tamanho_kernel=5, iteracoes=1)

    imagem_dilatada_com_linhas = cv2.cvtColor(imagem_dilatada, cv2.COLOR_GRAY2BGR)

    # Desenhar a midline na imagem dilatada (opcional, para referência)
    cv2.line(imagem_dilatada_com_linhas, ponto_modificado_1, ponto_modificado_2, (255, 255, 0), 2)  # Midline em ciano

    # Estender e desenhar as linhas na imagem dilatada
    for linha in linhas:
        ponto_concavidade, ponto_projetado = linha
        # Desenhar linha da concavidade até a midline
        cv2.line(imagem_dilatada_com_linhas, ponto_concavidade, ponto_projetado, (0, 255, 0), 2)  # Linhas em verde

        # Calcular a direção para extensão (do ponto projetado para a concavidade)
        dx = ponto_concavidade[0] - ponto_projetado[0]
        dy = ponto_concavidade[1] - ponto_projetado[1]
        distancia = math.hypot(dx, dy)
        if distancia == 0:
            continue  # Evitar divisão por zero
        # Calcular novo ponto estendido a partir da concavidade
        escala = comprimento_extensao / distancia
        novo_x = int(ponto_concavidade[0] + dx * escala)
        novo_y = int(ponto_concavidade[1] + dy * escala)
        novo_ponto = (novo_x, novo_y)

        # Desenhar a linha estendida
        cv2.line(imagem_dilatada_com_linhas, ponto_concavidade, novo_ponto, (0, 0, 255), 2)  # Linhas estendidas em vermelho

        # Opcional: marcar os pontos
        cv2.circle(imagem_dilatada_com_linhas, ponto_concavidade, 5, (0, 255, 0), -1)  # Concavidade em verde
        cv2.circle(imagem_dilatada_com_linhas, ponto_projetado, 5, (255, 0, 0), -1)      # Ponto na midline em azul
        cv2.circle(imagem_dilatada_com_linhas, novo_ponto, 5, (255, 255, 255), -1)      # Novo ponto estendido em branco

    # Salvar a segunda imagem com linhas estendidas
    salva_imagem(imagem_dilatada_com_linhas, caminho_saida_segunda)

    return imagem_resultado, imagem_dilatada_com_linhas