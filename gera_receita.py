# gera_receita.py
from auxiliares.concavities import converte_em_cinza
from segmentacao_aneis import aplicar_segmentacao_refinada
from analisa_concavidades_anel import (
    pipeline_anel,
    analisar_dados_anel,
    exibir_grafico_interativo,
    RingConcavityAnalyzer,
)

import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import json
import re
import ast
import copy

from utils_scaling import ImageScaler
from receita_pipeline import RecipeRunner


class FiltroApp:
    ARQUIVO_CONFIG = "config_padrao_app.json"

    def __init__(self, root):
        self.root = root
        self.root.title("Processamento de Imagens - GERADOR DE RECEITAS CENTRALIZADO")
        self.root.wm_state("zoomed")

        self.scaler = ImageScaler(reference_dim=1000.0)
        self.runner = RecipeRunner(self.scaler)

        self.resize_timer = None

        self.imagem_original = None
        self.imagem_filtrada_mostrada = None
        self.imagem_contornada = None
        self.estado_atual = None
        self.imagem_resultado = None
        self.imagem_resultado2 = None
        self.imagem_resultado3 = None
        self.origin_file = None
        
        # Armazena os dados reconstruidos em poligonos
        self.nucleos_json_resultado = []

        # Histórico para Undo / Redo
        self.estados_anteriores = []
        self.lista_filtros = []
        self.lista_receita_json = []
        
        self.estados_refazer = []
        self.lista_filtros_refazer = []
        self.lista_receita_json_refazer = []
        
        self.filtro_atual_dict = {}

        self.combo_values = [
            "Abertura",
            "Anisotropic Diffusion",
            "Background Subtraction",
            "Binarizacao Adaptativa (Gaussiana)",
            "Binarizacao Adaptativa (Media)",
            "Binarizacao Normal",
            "Binarizacao Phansalkar",
            "Binarizacao Sauvola",
            "Conectar Falhas",
            "Contraste",
            "Dilatacao",
            "Erosao",
            "Fechamento",
            "Filtro Bilateral",
            "Filtro de Detalhes",
            "Filtro de Preservacao",
            "Filtro Gaussiano",
            "Filtrar por Elipse",
            "Histogram Equalization",
            "Homomorphic Filter",
            "MedianBlur",
            "Non-Local Means",
            "Realce contraste",
            "Single Scale Retinex",
            "Somar Imagem",
            "Top-Hat",
        ]

        self.kernel_var = tk.IntVar(value=3)
        self.sigma_color = tk.IntVar(value=10)
        self.sigma_space = tk.IntVar(value=10)
        self.sigma_range = tk.DoubleVar(value=0.5)
        self.threshold_var = tk.DoubleVar(value=0.30)
        self.iteracoes_var = tk.IntVar(value=1)
        self.constante_var = tk.DoubleVar(value=0.0)

        self.limiar_min_obj = tk.IntVar(value=0)
        self.limiar_max_obj = tk.IntVar(value=1000)

        self.v_geo_thresh = tk.DoubleVar(value=0.002)
        self.v_geo_dist = tk.IntVar(value=120)
        self.v_geo_score = tk.DoubleVar(value=0.60)
        self.v_geo_fator = tk.DoubleVar(value=8.0)
        self.v_geo_area_min = tk.DoubleVar(value=0.25)
        self.v_geo_tangente = tk.DoubleVar(value=0.45)

        # Compatibilidade UI antiga / análise
        self.v_adapt_valley = tk.DoubleVar(value=10.0)
        self.v_adapt_gap = tk.DoubleVar(value=8.0)
        self.v_adapt_guard = tk.IntVar(value=6)
        self.v_adapt_thickness = tk.IntVar(value=1)
        self.v_adapt_angulos = tk.IntVar(value=720)
        self.v_adapt_min_run = tk.IntVar(value=5)

        # Topologia refinada para análise
        self.v_hole_area = tk.IntVar(value=3000)
        self.v_hole_dilate_kernel = tk.IntVar(value=7)
        self.v_hole_dilate_iters = tk.IntVar(value=2)
        self.v_intersection_kernel = tk.IntVar(value=3)
        self.v_intersection_iters = tk.IntVar(value=1)
        self.v_topology_close_kernel = tk.IntVar(value=3)
        self.v_topology_close_iters = tk.IntVar(value=1)

        # Novo filtro por elipse iterativo por distância
        self.v_ellipse_dist_base = tk.DoubleVar(value=18.0)
        self.v_ellipse_compat_min = tk.DoubleVar(value=0.20)
        self.v_ellipse_area_rel = tk.DoubleVar(value=0.01)
        self.v_ellipse_max_iter = tk.IntVar(value=8)
        self.v_ellipse_sample_step = tk.IntVar(value=3)
        self.v_ellipse_use_gap = tk.BooleanVar(value=True)
        self.v_ellipse_use_mad = tk.BooleanVar(value=True)
        self.v_ellipse_keep_main = tk.BooleanVar(value=True)

        self.info_filtro_var = tk.StringVar()
        self.info_qtd_obj = tk.StringVar(value="Objetos encontrados 0")

        self.R_var = tk.DoubleVar(value=0.5)
        self.k_var = tk.DoubleVar(value=0.25)
        self.p_var = tk.DoubleVar(value=2.0)
        self.q_var = tk.DoubleVar(value=10.0)

        # Configurar UI e Atalhos de Teclado
        self._build_ui()
        self.carregar_parametros_padrao()
        self._configurar_atalhos()
        
        self.atualizar_imagem()

    def _configurar_atalhos(self):
        def on_enter(event):
            if isinstance(event.widget, tk.Text):
                return
            self.atualizar_filtro()

        self.root.bind("<Return>", on_enter)
        
        def handle_undo(event):
            if isinstance(event.widget, (tk.Entry, ttk.Entry, tk.Text)):
                pass
            self.desfazer()

        def handle_redo(event):
            if isinstance(event.widget, (tk.Entry, ttk.Entry, tk.Text)):
                pass
            self.refazer()

        self.root.bind("<Control-z>", handle_undo)
        self.root.bind("<Control-y>", handle_redo)

    def _build_ui(self):
        self.main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.frame_lateral = ttk.Frame(self.main_paned, width=650)
        self.main_paned.add(self.frame_lateral, weight=0)
        self.frame_lateral.pack_propagate(False)

        self.canvas_controle = tk.Canvas(self.frame_lateral, borderwidth=0, highlightthickness=0)
        self.scrollbar_controle = ttk.Scrollbar(self.frame_lateral, orient="vertical", command=self.canvas_controle.yview)
        
        self.frame_controle = ttk.Frame(self.canvas_controle)
        self.canvas_janela = self.canvas_controle.create_window((0, 0), window=self.frame_controle, anchor="nw")

        self.frame_controle.bind("<Configure>", lambda e: self.canvas_controle.configure(scrollregion=self.canvas_controle.bbox("all")))
        self.canvas_controle.bind("<Configure>", lambda e: self.canvas_controle.itemconfig(self.canvas_janela, width=e.width))
        self.canvas_controle.configure(yscrollcommand=self.scrollbar_controle.set)

        self.canvas_controle.pack(side="left", fill="both", expand=True)
        self.scrollbar_controle.pack(side="right", fill="y")

        def _on_mousewheel(event):
            self.canvas_controle.yview_scroll(int(-1*(event.delta/120)), "units")
        self.canvas_controle.bind_all("<MouseWheel>", _on_mousewheel)

        self.frame_controle.columnconfigure(0, weight=2)
        self.frame_controle.columnconfigure(1, weight=1)

        self.frame_filtros = ttk.Frame(self.frame_controle)
        self.frame_filtros.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.frame_filtros.columnconfigure(1, weight=1)

        self.label_filtro = ttk.Label(self.frame_filtros, text="Escolha o(s) filtro(s):")
        self.label_filtro.grid(row=0, column=0, sticky="w", padx=5, pady=5)

        self.combo_filtro = ttk.Combobox(
            self.frame_filtros, values=self.combo_values, state="readonly", width=32,
        )
        self.combo_filtro.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        self.combo_filtro.bind("<<ComboboxSelected>>", self.atualizar_filtro)

        self.botao_adicionar_filtro = ttk.Button(self.frame_filtros, text="Adicionar Filtro", command=self.adicionar_filtro)
        self.botao_adicionar_filtro.grid(row=0, column=2, padx=5, pady=5)

        self.botao_injetar = ttk.Button(self.frame_filtros, text="Colar Filtros", command=self.abrir_janela_injecao)
        self.botao_injetar.grid(row=0, column=3, padx=5, pady=5)

        self.label_intensidade = ttk.Label(self.frame_filtros, text="Kernel/Limiar:")
        self.label_intensidade.grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.slider_intensidade = ttk.Scale(self.frame_filtros, from_=2, to=255, variable=self.kernel_var, orient=tk.HORIZONTAL, command=self.atualizar_filtro)
        self.slider_intensidade.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        self.entry_intensidade = ttk.Entry(self.frame_filtros, textvariable=self.kernel_var, width=7)
        self.entry_intensidade.grid(row=1, column=2, padx=5, pady=5)

        self.label_iteracoes = ttk.Label(self.frame_filtros, text="Iteracoes:")
        self.label_iteracoes.grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.scale_iteracoes = ttk.Scale(self.frame_filtros, from_=1, to=10, variable=self.iteracoes_var, orient=tk.HORIZONTAL, command=self.atualizar_filtro)
        self.scale_iteracoes.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
        self.entry_iteracoes = ttk.Entry(self.frame_filtros, textvariable=self.iteracoes_var, width=7)
        self.entry_iteracoes.grid(row=2, column=2, padx=5, pady=5)

        self.label_constante = ttk.Label(self.frame_filtros, text="Constante/Fator/H:")
        self.label_constante.grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.scale_constante = ttk.Scale(self.frame_filtros, from_=-50, to=150, variable=self.constante_var, orient=tk.HORIZONTAL, command=self.atualizar_filtro)
        self.scale_constante.grid(row=3, column=1, sticky="ew", padx=5, pady=5)
        self.entry_constante = ttk.Entry(self.frame_filtros, textvariable=self.constante_var, width=7)
        self.entry_constante.grid(row=3, column=2, padx=5, pady=5)

        self.label_sigma_space = ttk.Label(self.frame_filtros, text="Sigma space/Search:")
        self.label_sigma_space.grid(row=4, column=0, sticky="w", padx=5, pady=5)
        self.scale_sigma_space = ttk.Scale(self.frame_filtros, from_=0, to=100, variable=self.sigma_space, orient=tk.HORIZONTAL, command=self.atualizar_filtro)
        self.scale_sigma_space.grid(row=4, column=1, sticky="ew", padx=5, pady=5)
        self.entry_sigma_space = ttk.Entry(self.frame_filtros, textvariable=self.sigma_space, width=7)
        self.entry_sigma_space.grid(row=4, column=2, padx=5, pady=5)

        self.label_sigma_color = ttk.Label(self.frame_filtros, text="Sigma color:")
        self.label_sigma_color.grid(row=5, column=0, sticky="w", padx=5, pady=5)
        self.scale_sigma_color = ttk.Scale(self.frame_filtros, from_=0, to=100, variable=self.sigma_color, orient=tk.HORIZONTAL, command=self.atualizar_filtro)
        self.scale_sigma_color.grid(row=5, column=1, sticky="ew", padx=5, pady=5)
        self.entry_sigma_color = ttk.Entry(self.frame_filtros, textvariable=self.sigma_color, width=7)
        self.entry_sigma_color.grid(row=5, column=2, padx=5, pady=5)

        self.label_sigma_range = ttk.Label(self.frame_filtros, text="Sigma range:")
        self.label_sigma_range.grid(row=6, column=0, sticky="w", padx=5, pady=5)
        self.scale_sigma_range = ttk.Scale(self.frame_filtros, from_=0, to=1, variable=self.sigma_range, orient=tk.HORIZONTAL, command=self.atualizar_filtro)
        self.scale_sigma_range.grid(row=6, column=1, sticky="ew", padx=5, pady=5)
        self.entry_sigma_range = ttk.Entry(self.frame_filtros, textvariable=self.sigma_range, width=7)
        self.entry_sigma_range.grid(row=6, column=2, padx=5, pady=5)

        self.label_threshold = ttk.Label(self.frame_filtros, text="Threshold:")
        self.label_threshold.grid(row=7, column=0, sticky="w", padx=5, pady=5)
        self.scale_threshold = ttk.Scale(self.frame_filtros, from_=0, to=2, variable=self.threshold_var, orient=tk.HORIZONTAL, command=self.atualizar_filtro)
        self.scale_threshold.grid(row=7, column=1, sticky="ew", padx=5, pady=5)
        self.entry_threshold = ttk.Entry(self.frame_filtros, textvariable=self.threshold_var, width=7)
        self.entry_threshold.grid(row=7, column=2, padx=5, pady=5)

        self.label_info_filtro = ttk.Label(self.frame_filtros, textvariable=self.info_filtro_var, wraplength=400, justify=tk.LEFT)
        self.label_info_filtro.grid(row=8, column=0, columnspan=4, sticky="w", padx=5, pady=5)

        self.label_R = ttk.Label(self.frame_filtros, text="Parâmetro R:")
        self.scale_R = ttk.Scale(self.frame_filtros, from_=0.1, to=1.0, variable=self.R_var, orient=tk.HORIZONTAL, command=self.atualizar_filtro)
        self.entry_R = ttk.Entry(self.frame_filtros, textvariable=self.R_var, width=7)

        self.label_k = ttk.Label(self.frame_filtros, text="Parâmetro k:")
        self.scale_k = ttk.Scale(self.frame_filtros, from_=0.0, to=1.0, variable=self.k_var, orient=tk.HORIZONTAL, command=self.atualizar_filtro)
        self.entry_k = ttk.Entry(self.frame_filtros, textvariable=self.k_var, width=7)

        self.label_p = ttk.Label(self.frame_filtros, text="Parâmetro p:")
        self.scale_p = ttk.Scale(self.frame_filtros, from_=0.0, to=10.0, variable=self.p_var, orient=tk.HORIZONTAL, command=self.atualizar_filtro)
        self.entry_p = ttk.Entry(self.frame_filtros, textvariable=self.p_var, width=7)

        self.label_q = ttk.Label(self.frame_filtros, text="Parâmetro q:")
        self.scale_q = ttk.Scale(self.frame_filtros, from_=0.0, to=20.0, variable=self.q_var, orient=tk.HORIZONTAL, command=self.atualizar_filtro)
        self.entry_q = ttk.Entry(self.frame_filtros, textvariable=self.q_var, width=7)

        self._hide_phansalkar_controls()

        # --- SEÇÃO DE AÇÕES E PARÂMETROS ---
        self.frame_botoes_acao = ttk.Frame(self.frame_controle)
        self.frame_botoes_acao.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        self.frame_botoes_acao.columnconfigure((0, 1, 2, 3, 4), weight=1)

        ttk.Button(self.frame_botoes_acao, text="Carregar Imagem", command=self.carregar_imagem).grid(row=0, column=0, columnspan=5, sticky="ew", padx=5, pady=3)
        ttk.Button(self.frame_botoes_acao, text="Salvar Receita e Imagem", command=self.salvar_imagem_e_filtros).grid(row=1, column=0, columnspan=5, sticky="ew", padx=5, pady=3)
        
        ttk.Button(self.frame_botoes_acao, text="Desfazer (Ctrl+Z)", command=self.desfazer).grid(row=2, column=0, columnspan=2, sticky="ew", padx=5, pady=3)
        ttk.Button(self.frame_botoes_acao, text="Refazer (Ctrl+Y)", command=self.refazer).grid(row=2, column=2, columnspan=3, sticky="ew", padx=5, pady=3)
        
        ttk.Button(self.frame_botoes_acao, text="Resetar", command=self.resetar).grid(row=3, column=0, columnspan=5, sticky="ew", padx=5, pady=3)

        ttk.Button(self.frame_botoes_acao, text="Remocao objetos", command=self.remover_componentes).grid(row=4, column=0, sticky="ew", padx=5, pady=3)
        self.scale_obj_peq = ttk.Scale(self.frame_botoes_acao, from_=0, to=255, variable=self.limiar_min_obj, orient=tk.HORIZONTAL)
        self.scale_obj_peq.grid(row=4, column=1, columnspan=2, sticky="ew", padx=5, pady=3)
        self.entry_obj_min = ttk.Entry(self.frame_botoes_acao, textvariable=self.limiar_min_obj, width=8)
        self.entry_obj_min.grid(row=4, column=3, padx=2, pady=3)
        self.entry_obj_max = ttk.Entry(self.frame_botoes_acao, textvariable=self.limiar_max_obj, width=8)
        self.entry_obj_max.grid(row=4, column=4, padx=2, pady=3)

        ttk.Button(self.frame_botoes_acao, text="Limpar com mascara", command=self.limpa_mascara).grid(row=5, column=0, columnspan=5, sticky="ew", padx=5, pady=3)
        ttk.Button(self.frame_botoes_acao, text="Somar Imagem", command=self.somar_imagem).grid(row=6, column=0, columnspan=5, sticky="ew", padx=5, pady=3)

        ttk.Button(self.frame_botoes_acao, text="AND", command=self.bitwise_and).grid(row=7, column=0, columnspan=2, sticky="ew", padx=5, pady=3)
        ttk.Button(self.frame_botoes_acao, text="OR", command=self.bitwise_or).grid(row=7, column=2, columnspan=3, sticky="ew", padx=5, pady=3)
        ttk.Button(self.frame_botoes_acao, text="NOT", command=self.bitwise_not).grid(row=8, column=0, columnspan=2, sticky="ew", padx=5, pady=3)
        ttk.Button(self.frame_botoes_acao, text="XOR", command=self.bitwise_xor).grid(row=8, column=2, columnspan=3, sticky="ew", padx=5, pady=3)

        ttk.Button(self.frame_botoes_acao, text="Mostrar Contornos Transparentes", command=self.mostrar_contornos_transparentes).grid(row=9, column=0, columnspan=2, sticky="ew", padx=5, pady=3)
        ttk.Button(self.frame_botoes_acao, text="Mostrar Contornos Precisos", command=self.mostrar_contornos_verdes).grid(row=9, column=2, columnspan=3, sticky="ew", padx=5, pady=3)

        ttk.Button(self.frame_botoes_acao, text="Segmentar (Aplicar)", command=self.processar_imagem).grid(row=10, column=0, columnspan=2, sticky="ew", padx=2, pady=3)
        ttk.Button(self.frame_botoes_acao, text="Análise Visual", command=self.analise_visual).grid(row=10, column=2, columnspan=2, sticky="ew", padx=2, pady=3)
        ttk.Button(self.frame_botoes_acao, text="Gráfico", command=self.grafico_interativo).grid(row=10, column=4, sticky="ew", padx=2, pady=3)

        ttk.Label(self.frame_botoes_acao, text="Limiar Conc.:").grid(row=11, column=0, sticky="w", padx=2)
        ttk.Entry(self.frame_botoes_acao, textvariable=self.v_geo_thresh, width=8).grid(row=11, column=1, padx=2)
        ttk.Label(self.frame_botoes_acao, text="Max Dist:").grid(row=11, column=2, sticky="w", padx=2)
        ttk.Entry(self.frame_botoes_acao, textvariable=self.v_geo_dist, width=8).grid(row=11, column=3, padx=2)

        ttk.Label(self.frame_botoes_acao, text="Min Score:").grid(row=12, column=0, sticky="w", padx=2)
        ttk.Entry(self.frame_botoes_acao, textvariable=self.v_geo_score, width=8).grid(row=12, column=1, padx=2)
        ttk.Label(self.frame_botoes_acao, text="Fator Realce:").grid(row=12, column=2, sticky="w", padx=2)
        ttk.Entry(self.frame_botoes_acao, textvariable=self.v_geo_fator, width=8).grid(row=12, column=3, padx=2)

        ttk.Label(self.frame_botoes_acao, text="Fator Área Mín:").grid(row=13, column=0, sticky="w", padx=2)
        ttk.Entry(self.frame_botoes_acao, textvariable=self.v_geo_area_min, width=8).grid(row=13, column=1, padx=2)
        ttk.Label(self.frame_botoes_acao, text="Limiar Tangente:").grid(row=13, column=2, sticky="w", padx=2)
        ttk.Entry(self.frame_botoes_acao, textvariable=self.v_geo_tangente, width=8).grid(row=13, column=3, padx=2)

        ttk.Label(self.frame_botoes_acao, text="Hole Area:").grid(row=14, column=0, sticky="w", padx=2)
        ttk.Entry(self.frame_botoes_acao, textvariable=self.v_hole_area, width=8).grid(row=14, column=1, padx=2)
        ttk.Label(self.frame_botoes_acao, text="Hole Dil K:").grid(row=14, column=2, sticky="w", padx=2)
        ttk.Entry(self.frame_botoes_acao, textvariable=self.v_hole_dilate_kernel, width=8).grid(row=14, column=3, padx=2)

        ttk.Label(self.frame_botoes_acao, text="Hole Dil It:").grid(row=15, column=0, sticky="w", padx=2)
        ttk.Entry(self.frame_botoes_acao, textvariable=self.v_hole_dilate_iters, width=8).grid(row=15, column=1, padx=2)
        ttk.Label(self.frame_botoes_acao, text="Inter K:").grid(row=15, column=2, sticky="w", padx=2)
        ttk.Entry(self.frame_botoes_acao, textvariable=self.v_intersection_kernel, width=8).grid(row=15, column=3, padx=2)

        ttk.Label(self.frame_botoes_acao, text="Inter It:").grid(row=16, column=0, sticky="w", padx=2)
        ttk.Entry(self.frame_botoes_acao, textvariable=self.v_intersection_iters, width=8).grid(row=16, column=1, padx=2)
        ttk.Label(self.frame_botoes_acao, text="Top Close K:").grid(row=16, column=2, sticky="w", padx=2)
        ttk.Entry(self.frame_botoes_acao, textvariable=self.v_topology_close_kernel, width=8).grid(row=16, column=3, padx=2)

        ttk.Label(self.frame_botoes_acao, text="Top Close It:").grid(row=17, column=0, sticky="w", padx=2)
        ttk.Entry(self.frame_botoes_acao, textvariable=self.v_topology_close_iters, width=8).grid(row=17, column=1, padx=2)
        ttk.Label(self.frame_botoes_acao, text="Esp. Elipse:").grid(row=17, column=2, sticky="w", padx=2)
        ttk.Entry(self.frame_botoes_acao, textvariable=self.v_adapt_thickness, width=8).grid(row=17, column=3, padx=2)

        ttk.Label(self.frame_botoes_acao, text="Elipse Dist Base:").grid(row=18, column=0, sticky="w", padx=2)
        ttk.Entry(self.frame_botoes_acao, textvariable=self.v_ellipse_dist_base, width=8).grid(row=18, column=1, padx=2)
        ttk.Label(self.frame_botoes_acao, text="Compat Min:").grid(row=18, column=2, sticky="w", padx=2)
        ttk.Entry(self.frame_botoes_acao, textvariable=self.v_ellipse_compat_min, width=8).grid(row=18, column=3, padx=2)

        ttk.Label(self.frame_botoes_acao, text="Área Rel Min:").grid(row=19, column=0, sticky="w", padx=2)
        ttk.Entry(self.frame_botoes_acao, textvariable=self.v_ellipse_area_rel, width=8).grid(row=19, column=1, padx=2)
        ttk.Label(self.frame_botoes_acao, text="Max Iter:").grid(row=19, column=2, sticky="w", padx=2)
        ttk.Entry(self.frame_botoes_acao, textvariable=self.v_ellipse_max_iter, width=8).grid(row=19, column=3, padx=2)

        ttk.Label(self.frame_botoes_acao, text="Sample Step:").grid(row=20, column=0, sticky="w", padx=2)
        ttk.Entry(self.frame_botoes_acao, textvariable=self.v_ellipse_sample_step, width=8).grid(row=20, column=1, padx=2)

        ttk.Checkbutton(self.frame_botoes_acao, text="Usar Gap Dinâmico", variable=self.v_ellipse_use_gap).grid(row=21, column=0, columnspan=2, sticky="w", padx=2)
        ttk.Checkbutton(self.frame_botoes_acao, text="Usar MAD Dinâmico", variable=self.v_ellipse_use_mad).grid(row=21, column=2, columnspan=2, sticky="w", padx=2)
        ttk.Checkbutton(self.frame_botoes_acao, text="Manter Maior Componente", variable=self.v_ellipse_keep_main).grid(row=22, column=0, columnspan=3, sticky="w", padx=2)

        ttk.Button(self.frame_botoes_acao, text="Preview Topologia Refinada", command=self.preview_elipse_adaptativa).grid(row=23, column=0, columnspan=5, sticky="ew", padx=5, pady=4)
        ttk.Button(self.frame_botoes_acao, text="Exportar Sinais (TXT)", command=self.exportar_sinais_txt).grid(row=24, column=0, columnspan=5, sticky="ew", padx=5, pady=4)
        
        ttk.Button(self.frame_botoes_acao, text="Salvar Valores como Padrão", command=self.salvar_parametros_padrao).grid(row=25, column=0, columnspan=5, sticky="ew", padx=5, pady=4)

        self.notebook_imagens = ttk.Notebook(self.main_paned)
        self.main_paned.add(self.notebook_imagens, weight=1)

        self.f_orig = ttk.Frame(self.notebook_imagens)
        self.notebook_imagens.add(self.f_orig, text="Original")
        self.label_imagem_original = ttk.Label(self.f_orig, anchor="center")
        self.label_imagem_original.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.f_cont = ttk.Frame(self.notebook_imagens)
        self.notebook_imagens.add(self.f_cont, text="Contornos")
        self.label_imagem_contornos = ttk.Label(self.f_cont, anchor="center")
        self.label_imagem_contornos.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.f_filt = ttk.Frame(self.notebook_imagens)
        self.notebook_imagens.add(self.f_filt, text="Preview Filtro")
        self.label_imagem_filtrada = ttk.Label(self.f_filt, anchor="center")
        self.label_imagem_filtrada.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.notebook_imagens.bind("<Configure>", self.on_resize)

    def salvar_parametros_padrao(self):
        config = {
            "v_geo_thresh": self.v_geo_thresh.get(),
            "v_geo_dist": self.v_geo_dist.get(),
            "v_geo_score": self.v_geo_score.get(),
            "v_geo_fator": self.v_geo_fator.get(),
            "v_geo_area_min": self.v_geo_area_min.get(),
            "v_geo_tangente": self.v_geo_tangente.get(),
            "v_hole_area": self.v_hole_area.get(),
            "v_hole_dilate_kernel": self.v_hole_dilate_kernel.get(),
            "v_hole_dilate_iters": self.v_hole_dilate_iters.get(),
            "v_intersection_kernel": self.v_intersection_kernel.get(),
            "v_intersection_iters": self.v_intersection_iters.get(),
            "v_topology_close_kernel": self.v_topology_close_kernel.get(),
            "v_topology_close_iters": self.v_topology_close_iters.get(),
            "v_adapt_thickness": self.v_adapt_thickness.get(),
            "v_ellipse_dist_base": self.v_ellipse_dist_base.get(),
            "v_ellipse_compat_min": self.v_ellipse_compat_min.get(),
            "v_ellipse_area_rel": self.v_ellipse_area_rel.get(),
            "v_ellipse_max_iter": self.v_ellipse_max_iter.get(),
            "v_ellipse_sample_step": self.v_ellipse_sample_step.get(),
            "v_ellipse_use_gap": self.v_ellipse_use_gap.get(),
            "v_ellipse_use_mad": self.v_ellipse_use_mad.get(),
            "v_ellipse_keep_main": self.v_ellipse_keep_main.get(),
        }
        try:
            with open(self.ARQUIVO_CONFIG, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            messagebox.showinfo("Sucesso", "Parâmetros padrão salvos com sucesso!")
        except Exception as e:
            messagebox.showerror("Erro", f"Não foi possível salvar config:\n{e}")

    def carregar_parametros_padrao(self):
        if os.path.exists(self.ARQUIVO_CONFIG):
            try:
                with open(self.ARQUIVO_CONFIG, "r", encoding="utf-8") as f:
                    config = json.load(f)
                
                if "v_geo_thresh" in config: self.v_geo_thresh.set(config["v_geo_thresh"])
                if "v_geo_dist" in config: self.v_geo_dist.set(config["v_geo_dist"])
                if "v_geo_score" in config: self.v_geo_score.set(config["v_geo_score"])
                if "v_geo_fator" in config: self.v_geo_fator.set(config["v_geo_fator"])
                if "v_geo_area_min" in config: self.v_geo_area_min.set(config["v_geo_area_min"])
                if "v_geo_tangente" in config: self.v_geo_tangente.set(config["v_geo_tangente"])
                
                if "v_hole_area" in config: self.v_hole_area.set(config["v_hole_area"])
                if "v_hole_dilate_kernel" in config: self.v_hole_dilate_kernel.set(config["v_hole_dilate_kernel"])
                if "v_hole_dilate_iters" in config: self.v_hole_dilate_iters.set(config["v_hole_dilate_iters"])
                if "v_intersection_kernel" in config: self.v_intersection_kernel.set(config["v_intersection_kernel"])
                if "v_intersection_iters" in config: self.v_intersection_iters.set(config["v_intersection_iters"])
                if "v_topology_close_kernel" in config: self.v_topology_close_kernel.set(config["v_topology_close_kernel"])
                if "v_topology_close_iters" in config: self.v_topology_close_iters.set(config["v_topology_close_iters"])
                if "v_adapt_thickness" in config: self.v_adapt_thickness.set(config["v_adapt_thickness"])
                
                if "v_ellipse_dist_base" in config: self.v_ellipse_dist_base.set(config["v_ellipse_dist_base"])
                if "v_ellipse_compat_min" in config: self.v_ellipse_compat_min.set(config["v_ellipse_compat_min"])
                if "v_ellipse_area_rel" in config: self.v_ellipse_area_rel.set(config["v_ellipse_area_rel"])
                if "v_ellipse_max_iter" in config: self.v_ellipse_max_iter.set(config["v_ellipse_max_iter"])
                if "v_ellipse_sample_step" in config: self.v_ellipse_sample_step.set(config["v_ellipse_sample_step"])
                if "v_ellipse_use_gap" in config: self.v_ellipse_use_gap.set(config["v_ellipse_use_gap"])
                if "v_ellipse_use_mad" in config: self.v_ellipse_use_mad.set(config["v_ellipse_use_mad"])
                if "v_ellipse_keep_main" in config: self.v_ellipse_keep_main.set(config["v_ellipse_keep_main"])

            except Exception as e:
                print(f"Erro ao carregar {self.ARQUIVO_CONFIG}: {e}")

    def _carregar_parametros_na_ui(self, filtro_dict):
        nome = filtro_dict.get("nome", "")
        params = filtro_dict.get("parametros", {})

        if nome in self.combo_values:
            self.combo_filtro.set(nome)
        
        if "tamanho_kernel" in params: self.kernel_var.set(int(params["tamanho_kernel"]))
        elif "kernel" in params: self.kernel_var.set(int(params["kernel"]))
        elif "espessura_banda" in params: self.kernel_var.set(int(params["espessura_banda"]))
        
        if "constante_sigma" in params: self.constante_var.set(float(params["constante_sigma"]))
        elif "constante" in params: self.constante_var.set(float(params["constante"]))
        elif "fator_contraste" in params: self.constante_var.set(float(params["fator_contraste"]))
        elif "forca_filtro" in params: self.constante_var.set(float(params["forca_filtro"]))
        elif "kappa" in params: self.constante_var.set(float(params["kappa"]))

        if "threshold" in params: self.threshold_var.set(float(params["threshold"]))
        elif "threshold_val" in params: self.threshold_var.set(float(params["threshold_val"]))

        if "iteracoes" in params: self.iteracoes_var.set(int(params["iteracoes"]))
        elif "iters" in params: self.iteracoes_var.set(int(params["iters"]))

        if "sigma_space" in params: self.sigma_space.set(int(params["sigma_space"]))
        elif "search_window" in params: self.sigma_space.set(int(params["search_window"]))
        elif "dilatacao_extra" in params: self.sigma_space.set(int(params["dilatacao_extra"]))

        if "sigma_color" in params: self.sigma_color.set(int(params["sigma_color"]))
        if "sigma_range" in params: self.sigma_range.set(float(params["sigma_range"]))
        
        if "k" in params: self.k_var.set(float(params["k"]))
        if "R" in params: self.R_var.set(float(params["R"]))
        if "p" in params: self.p_var.set(float(params["p"]))
        if "q" in params: self.q_var.set(float(params["q"]))
        
        if "distancia_base" in params: self.v_ellipse_dist_base.set(float(params["distancia_base"]))
        if "compatibilidade_minima" in params: self.v_ellipse_compat_min.set(float(params["compatibilidade_minima"]))
        if "area_min_relativa" in params: self.v_ellipse_area_rel.set(float(params["area_min_relativa"]))
        if "max_iter" in params: self.v_ellipse_max_iter.set(int(params["max_iter"]))
        if "amostrar_contorno_passo" in params: self.v_ellipse_sample_step.set(int(params["amostrar_contorno_passo"]))
        if "usar_gap_dinamico" in params: self.v_ellipse_use_gap.set(bool(params["usar_gap_dinamico"]))
        if "usar_mad_dinamico" in params: self.v_ellipse_use_mad.set(bool(params["usar_mad_dinamico"]))
        if "manter_maior_componente" in params: self.v_ellipse_keep_main.set(bool(params["manter_maior_componente"]))

        self.atualizar_filtro()

    def exportar_sinais_txt(self):
        if self.imagem_filtrada_mostrada is None:
            messagebox.showwarning("Aviso", "Carregue e filtre uma imagem primeiro.")
            return

        try:
            img_topologia, res_data, _, _ = analisar_dados_anel(
                self.imagem_filtrada_mostrada.copy(),
                limiar_px_ext=self.v_geo_thresh.get(),
                limiar_px_int=self.v_geo_thresh.get(),
                fator_detalhe=self.v_geo_fator.get(),
                auto_binarizar=False,
                **self._coletar_parametros_adaptativos(),
            )
        except TypeError:
            img_topologia, res_data = self._analisar_com_parametros_adaptativos()

        if not res_data:
            messagebox.showwarning("Aviso", "Nenhum dado gerado na análise.")
            return

        caminho_arquivo = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Arquivo de Texto", "*.txt")],
            title="Salvar Sinais 1D",
        )

        if not caminho_arquivo:
            return

        ang_ext = res_data.get("ang_ext", [])
        sinal_ext = res_data.get("sinal_diff_ext", [])
        conc_ext_indices = [c["idx"] for c in res_data.get("conc_ext", [])]

        ang_int = res_data.get("ang_int", [])
        sinal_int = res_data.get("sinal_diff_int", [])
        conc_int_indices = [c["idx"] for c in res_data.get("conc_int", [])]

        with open(caminho_arquivo, "w", encoding="utf-8") as f:
            f.write("=== SINAL EXTERNO ===\n")
            f.write("Indice\tAngulo(rad)\tSinal(R_Razao)\tMarcado_Concavidade\n")
            for i in range(len(ang_ext)):
                is_conc = "SIM" if i in conc_ext_indices else "NAO"
                f.write(f"{i}\t{ang_ext[i]:.4f}\t{sinal_ext[i]:.4f}\t{is_conc}\n")

            f.write("\n=== SINAL INTERNO ===\n")
            f.write("Indice\tAngulo(rad)\tSinal(R_Razao)\tMarcado_Concavidade\n")
            for i in range(len(ang_int)):
                is_conc = "SIM" if i in conc_int_indices else "NAO"
                f.write(f"{i}\t{ang_int[i]:.4f}\t{sinal_int[i]:.4f}\t{is_conc}\n")

        messagebox.showinfo("Sucesso", f"Dados exportados para:\n{caminho_arquivo}")

    def _hide_phansalkar_controls(self):
        self.label_R.grid_remove()
        self.scale_R.grid_remove()
        self.entry_R.grid_remove()
        self.label_k.grid_remove()
        self.scale_k.grid_remove()
        self.entry_k.grid_remove()
        self.label_p.grid_remove()
        self.scale_p.grid_remove()
        self.entry_p.grid_remove()
        self.label_q.grid_remove()
        self.scale_q.grid_remove()
        self.entry_q.grid_remove()

    def _show_phansalkar_controls(self):
        self.label_R.grid(row=9, column=0, sticky="w", padx=5, pady=5)
        self.scale_R.grid(row=9, column=1, sticky="ew", padx=5, pady=5)
        self.entry_R.grid(row=9, column=2, padx=5, pady=5)

        self.label_k.grid(row=10, column=0, sticky="w", padx=5, pady=5)
        self.scale_k.grid(row=10, column=1, sticky="ew", padx=5, pady=5)
        self.entry_k.grid(row=10, column=2, padx=5, pady=5)

        self.label_p.grid(row=11, column=0, sticky="w", padx=5, pady=5)
        self.scale_p.grid(row=11, column=1, sticky="ew", padx=5, pady=5)
        self.entry_p.grid(row=11, column=2, padx=5, pady=5)

        self.label_q.grid(row=12, column=0, sticky="w", padx=5, pady=5)
        self.scale_q.grid(row=12, column=1, sticky="ew", padx=5, pady=5)
        self.entry_q.grid(row=12, column=2, padx=5, pady=5)

    def _ensure_uint8(self, img):
        if img is None:
            return None
        if img.dtype == np.uint8:
            return img
        if np.max(img) <= 1.0 and img.dtype != np.uint8:
            img = img * 255.0
        return np.clip(img, 0, 255).astype(np.uint8)

    def _executar_filtro_centralizado(self, imagem_base, filtro_dict):
        receita_original = copy.deepcopy(self.runner.receita_atual)
        try:
            self.runner.receita_atual = {
                "nome_receita": "preview_tmp",
                "pipeline_filtros": [copy.deepcopy(filtro_dict)],
            }
            return self.runner.executar(imagem_base)
        finally:
            self.runner.receita_atual = receita_original

    def _aplicar_operacao_imediata(self, filtro_dict, descricao):
        if self.imagem_original is None or self.estado_atual is None:
            return

        self.filtro_atual_dict = copy.deepcopy(filtro_dict)
        self.imagem_filtrada_mostrada = self._executar_filtro_centralizado(
            self.estado_atual.copy(), self.filtro_atual_dict
        )
        self.info_filtro_var.set(descricao)
        self.atualizar_imagem()
        self.adicionar_filtro()

    def _montar_filtro_atual(self):
        filtro = self.combo_filtro.get()
        kernel_raw = int(self.kernel_var.get())
        iteracoes = int(self.iteracoes_var.get())
        constante = float(self.constante_var.get())
        sigma_space = int(self.sigma_space.get())
        sigma_color = int(self.sigma_color.get())
        sigma_range = float(self.sigma_range.get())
        threshold = float(self.threshold_var.get())

        if filtro == "Binarizacao Phansalkar":
            return {
                "nome": "Binarizacao Phansalkar",
                "parametros": {
                    "tamanho_kernel": kernel_raw,
                    "R": float(self.R_var.get()),
                    "k": float(self.k_var.get()),
                    "p": float(self.p_var.get()),
                    "q": float(self.q_var.get()),
                },
            }

        if filtro == "Binarizacao Normal":
            return {
                "nome": "Binarizacao Normal",
                "parametros": {"threshold_val": threshold},
            }

        if filtro == "Binarizacao Adaptativa (Media)":
            return {
                "nome": "Binarizacao Adaptativa (Media)",
                "parametros": {"kernel": kernel_raw, "constante_sigma": constante},
            }

        if filtro == "Binarizacao Adaptativa (Gaussiana)":
            return {
                "nome": "Binarizacao Adaptativa (Gaussiana)",
                "parametros": {"kernel": kernel_raw, "constante_sigma": constante},
            }

        if filtro == "Erosao":
            return {
                "nome": "Erosao",
                "parametros": {"tamanho_kernel": kernel_raw, "iteracoes": iteracoes},
            }

        if filtro == "Dilatacao":
            return {
                "nome": "Dilatacao",
                "parametros": {"tamanho_kernel": kernel_raw, "iteracoes": iteracoes},
            }

        if filtro == "Abertura":
            return {
                "nome": "Abertura",
                "parametros": {"kernel": kernel_raw, "iters": iteracoes},
            }

        if filtro == "Fechamento":
            return {
                "nome": "Fechamento",
                "parametros": {"kernel": kernel_raw, "iters": iteracoes},
            }

        if filtro == "Filtro Gaussiano":
            return {
                "nome": "Filtro Gaussiano",
                "parametros": {
                    "tamanho_kernel": kernel_raw,
                    "constante_sigma": constante,
                },
            }

        if filtro == "Contraste":
            return {
                "nome": "Contraste",
                "parametros": {"kernel": kernel_raw, "fator_contraste": constante},
            }

        if filtro == "MedianBlur":
            return {"nome": "MedianBlur", "parametros": {"kernel": kernel_raw}}

        if filtro == "Filtro Bilateral":
            return {
                "nome": "Filtro Bilateral",
                "parametros": {
                    "kernel": kernel_raw,
                    "sigma_color": sigma_color,
                    "sigma_space": sigma_space,
                },
            }

        if filtro == "Non-Local Means":
            return {
                "nome": "Non-Local Means",
                "parametros": {
                    "forca_filtro": constante,
                    "template_window": kernel_raw,
                    "search_window": sigma_space,
                },
            }

        if filtro == "Binarizacao Sauvola":
            return {
                "nome": "Binarizacao Sauvola",
                "parametros": {"tamanho_kernel": kernel_raw, "threshold": threshold},
            }

        if filtro == "Filtro de Preservacao":
            return {
                "nome": "Filtro de Preservacao",
                "parametros": {"sigma_space": sigma_space, "sigma_range": sigma_range},
            }

        if filtro == "Homomorphic Filter":
            return {"nome": "Homomorphic Filter", "parametros": {"sigma": sigma_space}}

        if filtro == "Histogram Equalization":
            return {"nome": "Histogram Equalization", "parametros": {}}

        if filtro == "Background Subtraction":
            return {
                "nome": "Background Subtraction",
                "parametros": {"kernel_size": kernel_raw},
            }

        if filtro == "Filtro de Detalhes":
            return {
                "nome": "Filtro de Detalhes",
                "parametros": {"sigma_space": sigma_space, "sigma_range": sigma_range},
            }

        if filtro == "Single Scale Retinex":
            return {
                "nome": "Single Scale Retinex",
                "parametros": {"sigma": sigma_space},
            }

        if filtro == "Top-Hat":
            return {
                "nome": "Top-Hat",
                "parametros": {"kernel": kernel_raw, "iters": iteracoes},
            }

        if filtro == "Conectar Falhas":
            return {"nome": "Conectar Falhas", "parametros": {"kernel": kernel_raw}}

        if filtro == "Realce contraste":
            return {
                "nome": "Realce contraste",
                "parametros": {"kernel": kernel_raw, "fator_contraste": constante},
            }

        if filtro == "Anisotropic Diffusion":
            return {
                "nome": "Anisotropic Diffusion",
                "parametros": {"iters": iteracoes, "kappa": kernel_raw},
            }

        if filtro == "Somar Imagem":
            return {"nome": "Somar Imagem", "parametros": {}}

        if filtro == "Filtrar por Elipse":
            return {
                "nome": "Filtrar por Elipse",
                "parametros": {
                    "espessura_banda": kernel_raw,
                    "dilatacao_extra": sigma_space,
                    "distancia_base": float(self.v_ellipse_dist_base.get()),
                    "compatibilidade_minima": float(self.v_ellipse_compat_min.get()),
                    "area_min_relativa": float(self.v_ellipse_area_rel.get()),
                    "max_iter": int(self.v_ellipse_max_iter.get()),
                    "amostrar_contorno_passo": int(self.v_ellipse_sample_step.get()),
                    "usar_gap_dinamico": bool(self.v_ellipse_use_gap.get()),
                    "usar_mad_dinamico": bool(self.v_ellipse_use_mad.get()),
                    "manter_maior_componente": bool(self.v_ellipse_keep_main.get()),
                },
            }

        return None

    def _descrever_filtro(self, filtro_dict):
        nome = filtro_dict.get("nome", "Filtro")
        params = filtro_dict.get("parametros", {})

        if nome == "Binarizacao Phansalkar":
            return (
                f"{nome} - Kernel: {params.get('tamanho_kernel')}, "
                f"R: {params.get('R')}, k: {params.get('k')}, "
                f"p: {params.get('p')}, q: {params.get('q')}"
            )

        if nome == "Filtrar por Elipse":
            return (
                f"{nome} - Banda: {params.get('espessura_banda')}, "
                f"Dil Extra: {params.get('dilatacao_extra')}, "
                f"Dist Base: {params.get('distancia_base')}, "
                f"Compat: {params.get('compatibilidade_minima')}, "
                f"Area Rel: {params.get('area_min_relativa')}, "
                f"Iter: {params.get('max_iter')}"
            )

        return f"{nome} - Parametros: {params}"

    def _coletar_parametros_adaptativos(self):
        return {
            "valley_strength_base": float(self.v_adapt_valley.get()),
            "ellipse_gap_base": float(self.v_adapt_gap.get()),
            "angle_guard_base": int(self.v_adapt_guard.get()),
            "thickness_base": int(self.v_adapt_thickness.get()),
            "n_angulos": int(self.v_adapt_angulos.get()),
            "min_run_len": int(self.v_adapt_min_run.get()),
            "ellipse_thickness_base": int(self.v_adapt_thickness.get()),
            "bolsoes_area_max_base": int(self.v_hole_area.get()),
            "bolsoes_dilate_kernel_base": int(self.v_hole_dilate_kernel.get()),
            "bolsoes_dilate_iterations": int(self.v_hole_dilate_iters.get()),
            "intersection_dilate_kernel_base": int(self.v_intersection_kernel.get()),
            "intersection_dilate_iterations": int(self.v_intersection_iters.get()),
            "topology_close_kernel_base": int(self.v_topology_close_kernel.get()),
            "topology_close_iterations": int(self.v_topology_close_iters.get()),
            "refine_ellipse_with_holes": True,
            "use_ellipse_topology": True,
        }

    def _normalizar_binaria_para_analise(self, img):
        img = self._ensure_uint8(img)
        _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        return img_bin

    def _analisar_com_parametros_adaptativos(self):
        if self.imagem_filtrada_mostrada is None:
            return None, None

        img_bin = self._normalizar_binaria_para_analise(
            self.imagem_filtrada_mostrada.copy()
        )
        analyzer = RingConcavityAnalyzer(self.scaler)

        params_adapt = self._coletar_parametros_adaptativos()
        try:
            res = analyzer.processar(
                img_bin,
                limiar_px_ext=self.v_geo_thresh.get(),
                limiar_px_int=self.v_geo_thresh.get(),
                fator_detalhe=self.v_geo_fator.get(),
                **params_adapt,
            )
        except TypeError:
            res = analyzer.processar(
                img_bin,
                limiar_px_ext=self.v_geo_thresh.get(),
                limiar_px_int=self.v_geo_thresh.get(),
                fator_detalhe=self.v_geo_fator.get(),
            )

        img_topologia = None
        if res is not None:
            img_topologia = res.get("img_topologia", img_bin)

        return img_topologia, res

    def _montar_imagem_debug_adaptativa(self, img_topologia, res):
        if img_topologia is None:
            return None

        base = cv2.cvtColor(img_topologia, cv2.COLOR_GRAY2BGR)

        if res is None:
            return base

        if res.get("ellipse") is not None:
            cv2.ellipse(base, res["ellipse"], (0, 255, 255), 1, cv2.LINE_AA)

        ellipse_mask = res.get("ellipse_mask")
        if ellipse_mask is not None and np.any(ellipse_mask > 0):
            conts_adapt, _ = cv2.findContours(
                ellipse_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(base, conts_adapt, -1, (0, 200, 255), 1)

        if res.get("cont_externo") is not None:
            cv2.drawContours(base, [res["cont_externo"]], -1, (0, 255, 0), 2)

        if res.get("cont_interno") is not None:
            cv2.drawContours(base, [res["cont_interno"]], -1, (255, 0, 255), 2)

        for c in res.get("conc_ext", []):
            cv2.circle(base, c["ponto"], 4, (0, 0, 255), -1)

        for c in res.get("conc_int", []):
            cv2.circle(base, c["ponto"], 4, (255, 0, 0), -1)

        if "centro" in res:
            cx, cy = res["centro"]
            cv2.circle(base, (int(cx), int(cy)), 3, (0, 255, 255), -1)

        return base

    def _gerar_contorno_via_json(self, base_img, nucleos_json, thickness=1):
        if base_img is None:
            base = cv2.cvtColor(self.imagem_original.copy(), cv2.COLOR_GRAY2RGB)
        else:
            base = base_img.copy()
            if len(base.shape) == 2:
                base = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)

        draw_thickness = max(1, int(round(thickness * self.scaler.scale_factor)))

        for nucleo in nucleos_json:
            pts = np.array(nucleo["pontos"], dtype=np.int32)
            cv2.drawContours(base, [pts], -1, (0, 255, 0), draw_thickness, lineType=cv2.LINE_AA)

        return base

    def _gerar_mascara_transparente_via_json(self, base_img, nucleos_json, alpha=0.4):
        if base_img is None:
            base = cv2.cvtColor(self.imagem_original.copy(), cv2.COLOR_GRAY2RGB)
        else:
            base = base_img.copy()
            if len(base.shape) == 2:
                base = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)

        camada_cor = np.zeros_like(base)
        for nucleo in nucleos_json:
            pts = np.array(nucleo["pontos"], dtype=np.int32)
            cv2.fillPoly(camada_cor, [pts], (0, 255, 0))

        mask_indices = np.any(camada_cor > 0, axis=-1)
        resultado = base.copy()
        resultado[mask_indices] = cv2.addWeighted(base, 1 - alpha, camada_cor, alpha, 0)[mask_indices]
        
        return resultado

    def _contar_objetos(self, mask_img):
        if mask_img is None:
            return 0
        mask_uint8 = self._ensure_uint8(mask_img)
        num_labels, _, _, _ = cv2.connectedComponentsWithStats(
            mask_uint8, connectivity=8
        )
        return max(0, num_labels - 1)

    def _gerar_mapa_de_cores(self, shape, nucleos_json):
        mapa = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
        
        for nucleo in nucleos_json:
            n_id = nucleo["id"]
            pts = np.array(nucleo["pontos"], dtype=np.int32)
            
            r = (n_id * 75) % 200 + 55
            g = (n_id * 123) % 200 + 55
            b = (n_id * 211) % 200 + 55
            cor = (b, g, r)
            
            cv2.drawContours(mapa, [pts], -1, cor, thickness=cv2.FILLED)
            cv2.drawContours(mapa, [pts], -1, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
            
        return mapa

    def processar_imagem(self):
        if self.imagem_filtrada_mostrada is None:
            return

        limiar_conc = self.v_geo_thresh.get()
        max_dist = self.v_geo_dist.get()
        min_score = self.v_geo_score.get()
        fator_det = self.v_geo_fator.get()
        fator_area_min = self.v_geo_area_min.get()
        limiar_tang = self.v_geo_tangente.get()

        params_adapt = self._coletar_parametros_adaptativos()

        debug_img, mask_result, n_cortes, self.nucleos_json_resultado = aplicar_segmentacao_refinada(
            self.imagem_filtrada_mostrada.copy(),
            self.scaler,
            limiar_concavidade=limiar_conc,
            max_dist=max_dist,
            min_score=min_score,
            fator_detalhe=fator_det,
            auto_binarizar=False,
            fator_area_minima=fator_area_min,
            limiar_tangente=limiar_tang,
            **params_adapt
        )

        self.imagem_resultado = self._ensure_uint8(mask_result)
        self.imagem_resultado2 = self._ensure_uint8(debug_img)

        qtd_objetos = self._contar_objetos(self.imagem_resultado)
        self.info_qtd_obj.set(f"Objetos: {qtd_objetos} | Cortes: {n_cortes}")

        self.imagem_contornada = self._gerar_contorno_via_json(
            base_img=self.imagem_original, 
            nucleos_json=self.nucleos_json_resultado,
            thickness=1
        )

        self.atualizar_imagem()
        cv2.imshow("Debug Cortes (Segmentacao Geometrica)", self.imagem_resultado2)

        mapa_cores = self._gerar_mapa_de_cores(self.imagem_resultado.shape, self.nucleos_json_resultado)
        cv2.imshow("Mapa de Instancias (Cores Unicas)", mapa_cores)

    def analise_visual(self):
        if self.imagem_filtrada_mostrada is None:
            return

        img_topologia, res_data = self._analisar_com_parametros_adaptativos()
        if res_data is None:
            messagebox.showwarning("Aviso", "Não foi possível gerar dados de concavidade.")
            return

        analyzer = RingConcavityAnalyzer(self.scaler)
        
        img_gray = self._ensure_uint8(self.imagem_filtrada_mostrada.copy())
        if len(img_gray.shape) == 2:
            bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        else:
            bgr = img_gray.copy()

        res_img = analyzer.desenhar_resultados(bgr, res_data)

        if res_img is not None:
            cv2.imshow("Análise Visual de Concavidades", res_img)

    def grafico_interativo(self):
        if self.imagem_filtrada_mostrada is None:
            return

        try:
            img_topologia, res_data, limiar_ext, limiar_int = analisar_dados_anel(
                self.imagem_filtrada_mostrada.copy(),
                limiar_px_ext=self.v_geo_thresh.get(),
                limiar_px_int=self.v_geo_thresh.get(),
                fator_detalhe=self.v_geo_fator.get(),
                auto_binarizar=False,
                **self._coletar_parametros_adaptativos(),
            )
        except TypeError:
            img_topologia, res_data = self._analisar_com_parametros_adaptativos()
            limiar_ext = self.v_geo_thresh.get()
            limiar_int = self.v_geo_thresh.get()

        if res_data:
            exibir_grafico_interativo(img_topologia, res_data, limiar_ext, limiar_int)

    def preview_elipse_adaptativa(self):
        if self.imagem_filtrada_mostrada is None:
            messagebox.showwarning("Aviso", "Carregue e filtre uma imagem primeiro.")
            return

        img_topologia, res = self._analisar_com_parametros_adaptativos()
        if res is None or img_topologia is None:
            messagebox.showwarning(
                "Aviso", "Não foi possível gerar a topologia refinada."
            )
            return

        debug_img = self._montar_imagem_debug_adaptativa(img_topologia, res)
        cv2.imshow("Preview Topologia Refinada", debug_img)

    def on_resize(self, event):
        if self.resize_timer is not None:
            self.root.after_cancel(self.resize_timer)
        self.resize_timer = self.root.after(200, self.atualizar_imagem)

    def _preparar_imagem_tk(self, img_arr, w_box, h_box):
        if img_arr is None:
            return None

        h, w = img_arr.shape[:2]
        if h == 0 or w == 0:
            return None

        scale = min(w_box / w, h_box / h)
        nw, nh = int(w * scale), int(h * scale)
        if nw <= 0 or nh <= 0:
            return None

        rz = cv2.resize(img_arr, (nw, nh), interpolation=cv2.INTER_AREA)

        if len(rz.shape) == 2:
            rz = cv2.cvtColor(rz, cv2.COLOR_GRAY2RGB)
        elif len(rz.shape) == 3 and rz.shape[2] == 3:
            rz = cv2.cvtColor(rz, cv2.COLOR_BGR2RGB)

        return ImageTk.PhotoImage(Image.fromarray(rz))

    def atualizar_imagem(self):
        if self.imagem_original is None:
            return

        self.notebook_imagens.update_idletasks()
        wb = self.notebook_imagens.winfo_width() - 10
        hb = self.notebook_imagens.winfo_height() - 30 
        
        if wb < 50:
            wb, hb = 600, 600

        self.notebook_imagens.tab(self.f_cont, text=f"Contornos ({self.info_qtd_obj.get()})")

        self.tk_orig = self._preparar_imagem_tk(self.imagem_original, wb, hb)
        if self.tk_orig:
            self.label_imagem_original.configure(image=self.tk_orig)

        self.tk_filt = self._preparar_imagem_tk(self.imagem_filtrada_mostrada, wb, hb)
        if self.tk_filt:
            self.label_imagem_filtrada.configure(image=self.tk_filt)

        if self.imagem_contornada is not None:
            self.tk_cont = self._preparar_imagem_tk(self.imagem_contornada, wb, hb)
            if self.tk_cont:
                self.label_imagem_contornos.configure(image=self.tk_cont)

    def remover_componentes(self):
        if self.imagem_original is None or self.estado_atual is None:
            return

        filtro = {
            "nome": "Remocao objetos",
            "parametros": {
                "qnt_pixels_minimo": int(self.limiar_min_obj.get()),
                "qnt_pixels_maximo": int(self.limiar_max_obj.get()),
            },
        }
        self._aplicar_operacao_imediata(
            filtro,
            (
                f"Remocao objetos - Area Min: {self.limiar_min_obj.get()}, "
                f"Area Max: {self.limiar_max_obj.get()}"
            ),
        )

    def limpa_mascara(self):
        if self.imagem_original is None or self.estado_atual is None:
            return

        filtro = {
            "nome": "Remocao objetos",
            "parametros": {
                "qnt_pixels_minimo": 0,
                "qnt_pixels_maximo": int(self.limiar_min_obj.get()),
            },
        }
        self._aplicar_operacao_imediata(
            filtro,
            f"Limpa mascara - Area Max: {self.limiar_min_obj.get()}",
        )

    def resetar(self):
        if self.imagem_original is None:
            return

        self.estado_atual = self.imagem_original.copy()
        self.imagem_filtrada_mostrada = self.imagem_original.copy()
        self.imagem_contornada = None
        self.imagem_resultado = None
        self.imagem_resultado2 = None
        self.imagem_resultado3 = None
        
        self.nucleos_json_resultado = []

        self.estados_anteriores.clear()
        self.lista_filtros.clear()
        self.lista_receita_json.clear()
        
        self.estados_refazer.clear()
        self.lista_filtros_refazer.clear()
        self.lista_receita_json_refazer.clear()
        
        self.filtro_atual_dict = {}
        self.info_filtro_var.set("Pipeline resetado")
        self.info_qtd_obj.set("Objetos encontrados 0")
        self.atualizar_imagem()

    def adicionar_filtro(self):
        if self.imagem_filtrada_mostrada is None or not self.filtro_atual_dict:
            return

        self.estados_refazer.clear()
        self.lista_filtros_refazer.clear()
        self.lista_receita_json_refazer.clear()

        self.estados_anteriores.append(self.estado_atual.copy())
        self.estado_atual = self.imagem_filtrada_mostrada.copy()
        
        self.lista_filtros.append(self.info_filtro_var.get())
        self.lista_receita_json.append(copy.deepcopy(self.filtro_atual_dict))
        
        print(f"Filtro Adicionado: {self.info_filtro_var.get()}")
        self.atualizar_imagem()

    def desfazer(self):
        if not self.estados_anteriores:
            return

        self.estados_refazer.append(self.estado_atual.copy())
        
        filtro_desfeito = None
        if self.lista_receita_json:
            filtro_desfeito = self.lista_receita_json.pop()
            self.lista_receita_json_refazer.append(filtro_desfeito)
            
        if self.lista_filtros:
            self.lista_filtros_refazer.append(self.lista_filtros.pop())

        self.estado_atual = self.estados_anteriores.pop()
        self.imagem_filtrada_mostrada = self.estado_atual.copy()

        if self.lista_filtros:
            self.info_filtro_var.set(f"Desfez para: {self.lista_filtros[-1]}")
        else:
            self.info_filtro_var.set("Imagem Original")

        if filtro_desfeito:
            self._carregar_parametros_na_ui(filtro_desfeito)
        else:
            self.atualizar_imagem()

    def refazer(self):
        if not self.estados_refazer:
            return

        self.estados_anteriores.append(self.estado_atual.copy())
        
        self.estado_atual = self.estados_refazer.pop()
        self.imagem_filtrada_mostrada = self.estado_atual.copy()

        if self.lista_receita_json_refazer:
            self.lista_receita_json.append(self.lista_receita_json_refazer.pop())
            
        if self.lista_filtros_refazer:
            self.lista_filtros.append(self.lista_filtros_refazer.pop())
            self.info_filtro_var.set(f"Refez: {self.lista_filtros[-1]}")
        
        self.atualizar_imagem()

    def atualizar_filtro(self, event=None):
        self._hide_phansalkar_controls()

        if self.imagem_original is None or self.estado_atual is None:
            self.filtro_atual_dict = {}
            return

        filtro_selecionado = self.combo_filtro.get()
        if filtro_selecionado == "Binarizacao Phansalkar":
            self._show_phansalkar_controls()

        filtro_dict = self._montar_filtro_atual()
        if not filtro_dict:
            self.filtro_atual_dict = {}
            return

        try:
            self.imagem_filtrada_mostrada = self._executar_filtro_centralizado(
                self.estado_atual.copy(), filtro_dict
            )
            self.filtro_atual_dict = copy.deepcopy(filtro_dict)
            self.info_filtro_var.set(self._descrever_filtro(filtro_dict))
            self.atualizar_imagem()
        except Exception as exc:
            self.filtro_atual_dict = {}
            self.info_filtro_var.set(f"Erro ao pré-visualizar filtro: {exc}")

    def mostrar_contornos_verdes(self):
        if not self.nucleos_json_resultado:
            return

        self.imagem_contornada = self._gerar_contorno_via_json(
            base_img=self.imagem_original,
            nucleos_json=self.nucleos_json_resultado,
            thickness=1
        )
        self.atualizar_imagem()

    def mostrar_contornos_transparentes(self):
        if not self.nucleos_json_resultado:
            return

        self.imagem_contornada = self._gerar_mascara_transparente_via_json(
            base_img=self.imagem_original,
            nucleos_json=self.nucleos_json_resultado,
            alpha=0.4
        )
        self.atualizar_imagem()

    def carregar_imagem(self):
        pasta_padrao = os.getcwd()
        caminho_imagem = filedialog.askopenfilename(
            initialdir=pasta_padrao,
            title="Selecione a Imagem",
            filetypes=(
                ("Arquivos de Imagem", "*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff"),
                ("Todos os Arquivos", "*.*"),
            ),
        )

        if not caminho_imagem:
            return

        img = cv2.imread(caminho_imagem, cv2.IMREAD_UNCHANGED)
        if img is None:
            messagebox.showerror("Erro", "Não foi possível carregar a imagem.")
            return

        self.origin_file = caminho_imagem
        if len(img.shape) == 2:
            self.imagem_original = img.copy()
        elif len(img.shape) == 3 and img.shape[2] == 1:
            self.imagem_original = img[:, :, 0].copy()
        else:
            self.imagem_original = converte_em_cinza(img)
        self.scaler.update_from_image(self.imagem_original)

        self.resetar()
        self.info_filtro_var.set("Imagem Carregada")
        self.atualizar_imagem()

    def salvar_imagem_e_filtros(self):
        if self.imagem_filtrada_mostrada is None or self.origin_file is None:
            return

        pasta_atual = os.getcwd()
        nome_pasta_registros = os.path.join(pasta_atual, "Dados_modificados")
        os.makedirs(nome_pasta_registros, exist_ok=True)

        nome_pasta_origem = (
            os.path.basename(os.path.dirname(self.origin_file)) or "amostra"
        )
        numero_pasta_modificada = 1

        while True:
            nome_pasta_modificada = os.path.join(
                nome_pasta_registros,
                f"{nome_pasta_origem}_modificado{numero_pasta_modificada}",
            )
            if not os.path.exists(nome_pasta_modificada):
                os.makedirs(nome_pasta_modificada)
                break
            numero_pasta_modificada += 1

        cv2.imwrite(
            os.path.join(nome_pasta_modificada, "original.jpg"), self.imagem_original
        )
        cv2.imwrite(
            os.path.join(nome_pasta_modificada, "filtro.jpg"),
            self.imagem_filtrada_mostrada,
        )

        if self.imagem_contornada is not None:
            cv2.imwrite(
                os.path.join(nome_pasta_modificada, "contorno.jpg"),
                cv2.cvtColor(self.imagem_contornada, cv2.COLOR_RGB2BGR),
            )

        if self.imagem_resultado is not None:
            cv2.imwrite(
                os.path.join(nome_pasta_modificada, "segmentada.jpg"),
                self._ensure_uint8(self.imagem_resultado),
            )

            seg_contorno = self._gerar_contorno_via_json(self.imagem_original, self.nucleos_json_resultado)
            if seg_contorno is not None:
                cv2.imwrite(
                    os.path.join(nome_pasta_modificada, "segmentada_contorno.jpg"),
                    cv2.cvtColor(seg_contorno, cv2.COLOR_RGB2BGR),
                )

        if self.imagem_resultado2 is not None:
            cv2.imwrite(
                os.path.join(nome_pasta_modificada, "externa.jpg"),
                self._ensure_uint8(self.imagem_resultado2),
            )

            debug_contorno = self._gerar_contorno_via_json(self.imagem_resultado2, self.nucleos_json_resultado)
            if debug_contorno is not None:
                cv2.imwrite(
                    os.path.join(nome_pasta_modificada, "externa_contorno.jpg"),
                    cv2.cvtColor(debug_contorno, cv2.COLOR_RGB2BGR),
                )

        try:
            img_topologia, res = self._analisar_com_parametros_adaptativos()
            if res is not None and img_topologia is not None:
                preview_adapt = self._montar_imagem_debug_adaptativa(img_topologia, res)
                cv2.imwrite(
                    os.path.join(
                        nome_pasta_modificada, "preview_topologia_refinada.jpg"
                    ),
                    self._ensure_uint8(preview_adapt),
                )
                cv2.imwrite(
                    os.path.join(nome_pasta_modificada, "topologia_refinada.jpg"),
                    self._ensure_uint8(img_topologia),
                )
        except Exception:
            pass

        txt_filename = os.path.join(nome_pasta_modificada, "info.txt")
        with open(txt_filename, "w", encoding="utf-8") as f:
            f.write("Filtros utilizados:\n")
            for filtro in self.lista_filtros:
                f.write(f"{filtro}\n")
            f.write("\n" + self.info_qtd_obj.get() + "\n")
            f.write(
                "Segmentação Geométrica - "
                f"Limiar: {self.v_geo_thresh.get()}, "
                f"Max Dist: {self.v_geo_dist.get()}, "
                f"Score: {self.v_geo_score.get()}, "
                f"Fator: {self.v_geo_fator.get()}, "
                f"Área Mín: {self.v_geo_area_min.get()}, "
                f"Limiar Tang: {self.v_geo_tangente.get()}\n"
            )
            f.write(
                "Topologia Refinada - "
                f"Ellipse Thickness: {self.v_adapt_thickness.get()}, "
                f"Hole Area: {self.v_hole_area.get()}, "
                f"Hole Dil K: {self.v_hole_dilate_kernel.get()}, "
                f"Hole Dil It: {self.v_hole_dilate_iters.get()}, "
                f"Inter K: {self.v_intersection_kernel.get()}, "
                f"Inter It: {self.v_intersection_iters.get()}, "
                f"Top Close K: {self.v_topology_close_kernel.get()}, "
                f"Top Close It: {self.v_topology_close_iters.get()}\n"
            )
            f.write(
                "Filtro por Elipse Iterativo - "
                f"Dist Base: {self.v_ellipse_dist_base.get()}, "
                f"Compat Min: {self.v_ellipse_compat_min.get()}, "
                f"Área Rel Min: {self.v_ellipse_area_rel.get()}, "
                f"Max Iter: {self.v_ellipse_max_iter.get()}, "
                f"Sample Step: {self.v_ellipse_sample_step.get()}, "
                f"Gap Dinâmico: {self.v_ellipse_use_gap.get()}, "
                f"MAD Dinâmico: {self.v_ellipse_use_mad.get()}, "
                f"Manter Maior: {self.v_ellipse_keep_main.get()}\n"
            )

        json_filename = os.path.join(nome_pasta_modificada, "receita.json")
        texto_obj = self.info_qtd_obj.get()
        match = re.search(r"\d+", texto_obj)
        qtd_objetos = int(match.group()) if match else 0

        estrutura_json = {
            "nome_receita": f"Receita_{nome_pasta_origem}_mod{numero_pasta_modificada}",
            "pipeline_filtros": self.lista_receita_json,
            "parametros_analise_final": {
                "limiar_concavidade": self.v_geo_thresh.get(),
                "max_dist": self.v_geo_dist.get(),
                "min_score": self.v_geo_score.get(),
                "fator_detalhe": self.v_geo_fator.get(),
                "fator_area_minima": self.v_geo_area_min.get(),
                "limiar_tangente": self.v_geo_tangente.get(),
                "topologia_refinada": {
                    "ellipse_thickness_base": self.v_adapt_thickness.get(),
                    "bolsoes_area_max_base": self.v_hole_area.get(),
                    "bolsoes_dilate_kernel_base": self.v_hole_dilate_kernel.get(),
                    "bolsoes_dilate_iterations": self.v_hole_dilate_iters.get(),
                    "intersection_dilate_kernel_base": self.v_intersection_kernel.get(),
                    "intersection_dilate_iterations": self.v_intersection_iters.get(),
                    "topology_close_kernel_base": self.v_topology_close_kernel.get(),
                    "topology_close_iterations": self.v_topology_close_iters.get(),
                    "refine_ellipse_with_holes": True,
                    "use_ellipse_topology": True,
                },
            },
            "parametros_filtro_elipse_iterativo": {
                "distancia_base": self.v_ellipse_dist_base.get(),
                "compatibilidade_minima": self.v_ellipse_compat_min.get(),
                "area_min_relativa": self.v_ellipse_area_rel.get(),
                "max_iter": self.v_ellipse_max_iter.get(),
                "amostrar_contorno_passo": self.v_ellipse_sample_step.get(),
                "usar_gap_dinamico": self.v_ellipse_use_gap.get(),
                "usar_mad_dinamico": self.v_ellipse_use_mad.get(),
                "manter_maior_componente": self.v_ellipse_keep_main.get(),
            },
        }

        with open(json_filename, "w", encoding="utf-8") as f_json:
            json.dump(estrutura_json, f_json, indent=4, ensure_ascii=False)

        resultados_filename = os.path.join(nome_pasta_modificada, "resultados_esperados.json")
        estrutura_resultados = {
            "objetos_encontrados": qtd_objetos,
            "nucleos": self.nucleos_json_resultado
        }
        with open(resultados_filename, "w", encoding="utf-8") as f_res:
            json.dump(estrutura_resultados, f_res, indent=4, ensure_ascii=False)

        messagebox.showinfo("Sucesso", f"Receita salva em:\n{nome_pasta_modificada}")

    def bitwise_and(self):
        self._aplicar_operacao_imediata(
            {"nome": "Bitwise AND", "parametros": {}}, "Bitwise AND"
        )

    def bitwise_not(self):
        self._aplicar_operacao_imediata(
            {"nome": "Bitwise NOT", "parametros": {}}, "Bitwise NOT"
        )

    def bitwise_or(self):
        self._aplicar_operacao_imediata(
            {"nome": "Bitwise OR", "parametros": {}}, "Bitwise OR"
        )

    def bitwise_xor(self):
        self._aplicar_operacao_imediata(
            {"nome": "Bitwise XOR", "parametros": {}}, "Bitwise XOR"
        )

    def somar_imagem(self):
        self._aplicar_operacao_imediata(
            {"nome": "Somar Imagem", "parametros": {}}, "Somar Imagem"
        )

    def abrir_janela_injecao(self):
        if self.imagem_original is None:
            messagebox.showwarning("Aviso", "Carregue uma imagem primeiro.")
            return

        janela = tk.Toplevel(self.root)
        janela.title("Injetar Filtros por Texto")
        janela.geometry("700x450")

        lbl = ttk.Label(
            janela,
            text="Cole aqui o trecho da receita (JSON) ou log (Nome | Param: {...}):",
        )
        lbl.pack(pady=5, padx=10, anchor="w")

        text_area = tk.Text(janela, wrap=tk.WORD, height=18)
        text_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        def confirmar():
            texto = text_area.get("1.0", tk.END)
            self.injetar_filtros(texto)
            janela.destroy()

        btn = ttk.Button(janela, text="Aplicar Filtros", command=confirmar)
        btn.pack(pady=10)

    def injetar_filtros(self, texto):
        filtros_parseados = []

        padrao1 = re.compile(r"(.+?)\s*\|\s*Param:\s*(\{.*\})", re.IGNORECASE)
        for linha in texto.split("\n"):
            m = padrao1.search(linha)
            if m:
                try:
                    nome = m.group(1).strip()
                    params = ast.literal_eval(m.group(2).strip())
                    filtros_parseados.append({"nome": nome, "parametros": params})
                except Exception:
                    pass

        if not filtros_parseados:
            padrao2 = re.compile(
                r'(?:["\'])nome(?:["\'])\s*:\s*(?:["\'])([^"\']+)(?:["\'])\s*,\s*'
                r'(?:["\'])parametros(?:["\'])\s*:\s*(\{.*?\})',
                re.DOTALL,
            )
            for m in padrao2.finditer(texto):
                try:
                    nome = m.group(1).strip()
                    p_str = m.group(2).strip()
                    try:
                        params = json.loads(p_str)
                    except Exception:
                        params = ast.literal_eval(p_str)
                    filtros_parseados.append({"nome": nome, "parametros": params})
                except Exception:
                    pass

        if not filtros_parseados:
            messagebox.showwarning(
                "Aviso",
                "Não foi possível identificar filtros válidos no texto colado.",
            )
            return

        nome_map = {
            "gaussiano": "Filtro Gaussiano",
            "filtro gaussiano": "Filtro Gaussiano",
            "non-local means": "Non-Local Means",
            "non local means": "Non-Local Means",
            "realce contraste (clahe)": "Realce contraste",
            "median": "MedianBlur",
            "somar imagem": "Somar Imagem",
            "filtrar por elipse": "Filtrar por Elipse",
        }

        for f in filtros_parseados:
            nome = f["nome"]
            params = f.get("parametros", {}) or {}
            nome_norm = nome.lower().strip()

            if nome_norm in ("remocao objetos", "remover componentes pequenos"):
                self.limiar_min_obj.set(int(params.get("qnt_pixels_minimo", 0)))
                self.limiar_max_obj.set(int(params.get("qnt_pixels_maximo", 200)))
                self.remover_componentes()
                continue

            if nome_norm == "bitwise not":
                self.bitwise_not()
                continue
            if nome_norm == "bitwise or":
                self.bitwise_or()
                continue
            if nome_norm == "bitwise xor":
                self.bitwise_xor()
                continue
            if nome_norm == "bitwise and":
                self.bitwise_and()
                continue

            nome_mapeado = nome_map.get(nome_norm, nome)
            if nome_mapeado in self.combo_values:
                self.combo_filtro.set(nome_mapeado)

            if "tamanho_kernel" in params:
                self.kernel_var.set(int(params["tamanho_kernel"]))
            elif "kernel" in params:
                self.kernel_var.set(int(params["kernel"]))
            elif "template_window" in params:
                self.kernel_var.set(int(params["template_window"]))
            elif "espessura_banda" in params:
                self.kernel_var.set(int(params["espessura_banda"]))

            if "constante_sigma" in params:
                self.constante_var.set(float(params["constante_sigma"]))
            elif "constante" in params:
                self.constante_var.set(float(params["constante"]))
            elif "fator_contraste" in params:
                self.constante_var.set(float(params["fator_contraste"]))
            elif "forca_filtro" in params:
                self.constante_var.set(float(params["forca_filtro"]))
            elif "kappa" in params:
                self.constante_var.set(float(params["kappa"]))

            if "threshold" in params:
                self.threshold_var.set(float(params["threshold"]))
            elif "threshold_val" in params:
                self.threshold_var.set(float(params["threshold_val"]))

            if "iteracoes" in params:
                self.iteracoes_var.set(int(params["iteracoes"]))
            elif "iters" in params:
                self.iteracoes_var.set(int(params["iters"]))

            if "sigma_space" in params:
                self.sigma_space.set(int(params["sigma_space"]))
            elif "search_window" in params:
                self.sigma_space.set(int(params["search_window"]))
            elif "dilatacao_extra" in params:
                self.sigma_space.set(int(params["dilatacao_extra"]))

            if "sigma_color" in params:
                self.sigma_color.set(int(params["sigma_color"]))
            if "sigma_range" in params:
                self.sigma_range.set(float(params["sigma_range"]))

            if "distancia_base" in params:
                self.v_ellipse_dist_base.set(float(params["distancia_base"]))
            if "compatibilidade_minima" in params:
                self.v_ellipse_compat_min.set(float(params["compatibilidade_minima"]))
            if "area_min_relativa" in params:
                self.v_ellipse_area_rel.set(float(params["area_min_relativa"]))
            if "max_iter" in params:
                self.v_ellipse_max_iter.set(int(params["max_iter"]))
            if "amostrar_contorno_passo" in params:
                self.v_ellipse_sample_step.set(int(params["amostrar_contorno_passo"]))
            if "usar_gap_dinamico" in params:
                self.v_ellipse_use_gap.set(bool(params["usar_gap_dinamico"]))
            if "usar_mad_dinamico" in params:
                self.v_ellipse_use_mad.set(bool(params["usar_mad_dinamico"]))
            if "manter_maior_componente" in params:
                self.v_ellipse_keep_main.set(bool(params["manter_maior_componente"]))

            if "k" in params:
                self.k_var.set(float(params["k"]))
            if "R" in params:
                self.R_var.set(float(params["R"]))
            if "p" in params:
                self.p_var.set(float(params["p"]))
            if "q" in params:
                self.q_var.set(float(params["q"]))

            self.atualizar_filtro()
            self.adicionar_filtro()

        messagebox.showinfo(
            "Sucesso",
            f"{len(filtros_parseados)} filtro(s) injetado(s) e aplicado(s) com sucesso!",
        )

def main():
    root = tk.Tk()
    app = FiltroApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()