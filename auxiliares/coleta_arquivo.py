import os
import shutil
import tkinter as tk
from tkinter import filedialog

def coletar_arquivos(caminho_base):
    imagens_recolhidas = []
    extensoes_imagem = [".bmp", ".png", ".tif", ".jpg", ".jpeg"]
    for i in range(1, 1000):  # Presume-se que as subpastas estejam numeradas sequencialmente.
        caminho_subpasta = os.path.join(caminho_base, str(i))
        if not os.path.exists(caminho_subpasta):
            break

        caminho_originals = os.path.join(caminho_subpasta, "originals")

        # Definindo arquivos a serem coletados
        arquivos_necessarios = ["1", "10"]

        imagens_pasta_atual = []
        for arquivo in arquivos_necessarios:
            arquivo_encontrado = False
            for extensao in extensoes_imagem:
                caminho_arquivo = os.path.join(caminho_originals, arquivo + extensao)
                if os.path.exists(caminho_arquivo):
                    print(f"Arquivo encontrado: {caminho_arquivo}")
                    imagens_pasta_atual.append(caminho_arquivo)
                    arquivo_encontrado = True
                    break

            if not arquivo_encontrado:
                # Procurando na pasta pai
                caminho_pai_originals = os.path.join(caminho_base, str(i - 1), "originals")
                for extensao in extensoes_imagem:
                    caminho_arquivo_pai = os.path.join(caminho_pai_originals, arquivo + extensao)
                    if os.path.exists(caminho_arquivo_pai):
                        print(f"Arquivo encontrado na pasta pai: {caminho_arquivo_pai}")
                        imagens_pasta_atual.append(caminho_arquivo_pai)
                        arquivo_encontrado = True
                        break

            if not arquivo_encontrado:
                print(f"Arquivo {arquivo} não encontrado na pasta {caminho_originals} ou na pasta pai.")

        if imagens_pasta_atual:
            imagens_recolhidas.append(imagens_pasta_atual)

    return imagens_recolhidas


def selecionar_caminho_base():
    root = tk.Tk()
    root.withdraw()  # Oculta a janela principal do Tkinter
    caminho_base = filedialog.askdirectory(title="Selecione o caminho base da pasta DadosMutantes_select")
    return caminho_base