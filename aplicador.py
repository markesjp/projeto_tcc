# aplicador.py
import os
import json
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk

from utils_scaling import ImageScaler
from auxiliares.concavities import converte_em_cinza
from segmentacao_aneis import aplicar_segmentacao_refinada
from receita_pipeline import RecipeRunner

class FiltroApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Processamento de Imagens em Lote (Receitas JSON)")
        self.root.geometry("1200x760")
        self.scaler = ImageScaler(reference_dim=1000.0)
        self.runner = RecipeRunner(self.scaler)
        self.receita_atual = {"nome_receita": "Padrão", "pipeline_filtros": [], "parametros_analise_final": {}}
        self.imagem_original = self.imagem_pre = self.imagem_processada1 = None
        self.nucleos_json_resultado = []
        self._build_ui()

    def _build_ui(self):
        self.frame_controle = ttk.Frame(self.root)
        self.frame_controle.pack(padx=10, pady=10, fill=tk.X)
        self.frame_botoes = ttk.Frame(self.frame_controle)
        self.frame_botoes.pack(side=tk.LEFT, padx=5, pady=5)

        ttk.Button(self.frame_botoes, text="Carregar Imagem Única", command=self.carregar_imagem).grid(row=0, column=0, pady=5, sticky="ew")
        ttk.Button(self.frame_botoes, text="Carregar Receita (JSON)", command=self.carregar_receita_json).grid(row=1, column=0, pady=5, sticky="ew")
        self.botao_aplicar = ttk.Button(self.frame_botoes, text="Aplicar Receita na Imagem", command=self.aplicar_filtros, state=tk.DISABLED)
        self.botao_aplicar.grid(row=2, column=0, pady=5, sticky="ew")
        self.botao_salvar = ttk.Button(self.frame_botoes, text="Salvar Resultado", command=self.salvar_imagem, state=tk.DISABLED)
        self.botao_salvar.grid(row=3, column=0, pady=5, sticky="ew")
        ttk.Separator(self.frame_botoes, orient=tk.HORIZONTAL).grid(row=4, column=0, sticky="ew", pady=10)
        ttk.Button(self.frame_botoes, text="Selecionar Pasta Base", command=self.carregar_pasta).grid(row=5, column=0, pady=5, sticky="ew")
        self.botao_lote = ttk.Button(self.frame_botoes, text="Processar Pasta em Lote", command=self.aplicar_filtros_pasta, state=tk.DISABLED)
        self.botao_lote.grid(row=6, column=0, pady=5, sticky="ew")

        self.frame_info = ttk.LabelFrame(self.frame_controle, text="Resumo da Receita")
        self.frame_info.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.scrollbar = ttk.Scrollbar(self.frame_info)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_info = tk.Text(self.frame_info, height=12, width=80, wrap=tk.WORD, yscrollcommand=self.scrollbar.set)
        self.text_info.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.config(command=self.text_info.yview)

        self.frame_imagem = ttk.Frame(self.root)
        self.frame_imagem.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.frame_imagem.columnconfigure((0, 1), weight=1); self.frame_imagem.rowconfigure(0, weight=1)

        self.box_original = ttk.LabelFrame(self.frame_imagem, text="Original")
        self.box_original.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.label_original = ttk.Label(self.box_original, anchor="center")
        self.label_original.pack(fill=tk.BOTH, expand=True)

        self.box_resultado = ttk.LabelFrame(self.frame_imagem, text="Pós-pipeline")
        self.box_resultado.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.label_resultado = ttk.Label(self.box_resultado, anchor="center")
        self.label_resultado.pack(fill=tk.BOTH, expand=True)
        self.atualizar_visor_receita()

    def _ensure_uint8(self, img):
        if img is None: return None
        if img.dtype == np.uint8: return img
        if img.max() <= 1.0: img = (img * 255)
        return np.clip(img, 0, 255).astype(np.uint8)

    def contorno_objeto_json(self, imagem_base, nucleos_json):
        if imagem_base is None or not nucleos_json: return None
        base = cv2.cvtColor(imagem_base.copy(), cv2.COLOR_GRAY2RGB) if len(imagem_base.shape)==2 else imagem_base.copy()
        draw_t = max(1, int(round(1 * self.scaler.scale_factor)))
        for n in nucleos_json:
            pts = np.array(n["pontos"], dtype=np.int32)
            cv2.drawContours(base, [pts], -1, (0, 255, 0), draw_t, lineType=cv2.LINE_AA)
        return base

    def atualizar_visor_receita(self):
        self.text_info.config(state=tk.NORMAL)
        self.text_info.delete("1.0", tk.END)
        self.text_info.insert(tk.END, f"Receita: {self.receita_atual.get('nome_receita', '')}\n\n")
        for i, f in enumerate(self.receita_atual.get("pipeline_filtros", []), 1):
            self.text_info.insert(tk.END, f"{i}. {f.get('nome')} | Param: {f.get('parametros')}\n")
        self.text_info.insert(tk.END, f"\nEscala atual: {self.scaler.get_info()}\n")
        self.text_info.config(state=tk.DISABLED)

    def carregar_receita_json(self):
        cam = filedialog.askopenfilename(title="Selecione a Receita", filetypes=[("JSON", "*.json")])
        if not cam: return
        with open(cam, "r", encoding="utf-8") as f: self.receita_atual = json.load(f)
        self.runner.definir_receita(self.receita_atual)
        self.atualizar_visor_receita()
        if self.imagem_original is not None:
            self.botao_aplicar.config(state=tk.NORMAL)
            self.botao_lote.config(state=tk.NORMAL)

    def carregar_imagem(self):
        cam = filedialog.askopenfilename(title="Selecione a Imagem", filetypes=(("Imagens", "*.jpg;*.png;*.tif"), ("Tudo", "*.*")))
        if not cam: return
        img = cv2.imread(cam, cv2.IMREAD_UNCHANGED)
        self.imagem_original = converte_em_cinza(img) if len(img.shape)==3 else img.copy()
        self.scaler.update_from_image(self.imagem_original)
        self.imagem_pre = self.imagem_original.copy()
        
        self.exibir_imagem(self.imagem_original, self.label_original)
        self.exibir_imagem(self.imagem_pre, self.label_resultado)
        if self.receita_atual.get("pipeline_filtros"): self.botao_aplicar.config(state=tk.NORMAL); self.botao_lote.config(state=tk.NORMAL)
        self.atualizar_visor_receita()

    def carregar_pasta(self):
        if pasta := filedialog.askdirectory(title="Selecione a Pasta Base"):
            self.pasta_selecionada = pasta
            if self.receita_atual.get("pipeline_filtros"): self.botao_lote.config(state=tk.NORMAL)

    def exibir_imagem(self, img, label):
        if img is None: return
        img_rgb = cv2.cvtColor(self._ensure_uint8(img), cv2.COLOR_GRAY2RGB) if len(img.shape)==2 else cv2.cvtColor(self._ensure_uint8(img), cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        pil_img.thumbnail((max(500, label.winfo_width()), max(500, label.winfo_height())), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(pil_img)
        label.configure(image=tk_img); label.image = tk_img

    def aplicar_filtros(self):
        self.imagem_pre = self.runner.executar(self.imagem_original)
        p = self.receita_atual.get("parametros_analise_final", {})
        topologia_kwargs = p.get("topologia_refinada", {})
        
        db_img, mask_bin, _, self.nucleos_json_resultado = aplicar_segmentacao_refinada(
            self.imagem_pre.copy(), self.scaler, auto_binarizar=False,
            limiar_concavidade=p.get("limiar_concavidade", 0.005),
            max_dist=p.get("max_dist", 120),
            min_score=p.get("min_score", 0.6),
            fator_detalhe=p.get("fator_detalhe", 8.0),
            fator_area_minima=p.get("fator_area_minima", 0.25),
            limiar_tangente=p.get("limiar_tangente", 0.45),
            **topologia_kwargs
        )

        self.imagem_processada1 = self._ensure_uint8(mask_bin)
        self.imagem_processada2 = self._ensure_uint8(db_img)
        self.imagem_contornada1 = self.contorno_objeto_json(self.imagem_original, self.nucleos_json_resultado)

        self.exibir_imagem(self.imagem_original, self.label_original)
        self.exibir_imagem(self.imagem_pre, self.label_resultado)
        self.botao_salvar.config(state=tk.NORMAL)

    def salvar_imagem(self):
        subpasta = os.path.join("mascara", str(max([int(n) for n in os.listdir("mascara") if n.isdigit()] + [0]) + 1))
        os.makedirs(subpasta, exist_ok=True)
        cv2.imwrite(os.path.join(subpasta, "orig.jpg"), self.imagem_original)
        cv2.imwrite(os.path.join(subpasta, "pre.jpg"), self.imagem_pre)
        if self.imagem_processada1 is not None: cv2.imwrite(os.path.join(subpasta, "seg.jpg"), self.imagem_processada1)
        if self.imagem_contornada1 is not None: cv2.imwrite(os.path.join(subpasta, "seg_contorno.jpg"), cv2.cvtColor(self.imagem_contornada1, cv2.COLOR_RGB2BGR))
        with open(os.path.join(subpasta, "resultados.json"), "w", encoding="utf-8") as f:
            json.dump({"objetos": len(self.nucleos_json_resultado), "nucleos": self.nucleos_json_resultado}, f, indent=2)
        messagebox.showinfo("Sucesso", f"Salvo em: {subpasta}")

    def aplicar_filtros_pasta(self):
        p = self.receita_atual.get("parametros_analise_final", {})
        topologia_kwargs = p.get("topologia_refinada", {})
        pasta_out = os.path.join(os.getcwd(), "PREPROCESSADAS")
        proc = 0

        for r, _, files in os.walk(self.pasta_selecionada):
            for f in [x for x in files if x.lower().endswith((".png", ".jpg", ".tif"))]:
                img = cv2.imread(os.path.join(r, f), cv2.IMREAD_UNCHANGED)
                self.imagem_original = converte_em_cinza(img) if len(img.shape)==3 else img.copy()
                self.scaler.update_from_image(self.imagem_original)
                
                img_pre = self.runner.executar(self.imagem_original)
                db_img, mask_bin, n_c, n_json = aplicar_segmentacao_refinada(
                    img_pre.copy(), self.scaler, auto_binarizar=False,
                    limiar_concavidade=p.get("limiar_concavidade", 0.005),
                    max_dist=p.get("max_dist", 120),
                    min_score=p.get("min_score", 0.6),
                    fator_detalhe=p.get("fator_detalhe", 8.0),
                    fator_area_minima=p.get("fator_area_minima", 0.25),
                    limiar_tangente=p.get("limiar_tangente", 0.45),
                    **topologia_kwargs
                )

                sub = os.path.join(pasta_out, os.path.relpath(r, self.pasta_selecionada))
                os.makedirs(sub, exist_ok=True)
                
                cv2.imwrite(os.path.join(sub, f"orig_{f}"), self.imagem_original)
                cv2.imwrite(os.path.join(sub, f"seg_{f}"), self._ensure_uint8(mask_bin))
                if cont := self.contorno_objeto_json(self.imagem_original, n_json):
                    cv2.imwrite(os.path.join(sub, f"cont_{f}"), cv2.cvtColor(cont, cv2.COLOR_RGB2BGR))
                with open(os.path.join(sub, f"res_{f}.json"), "w", encoding="utf-8") as jf:
                    json.dump({"objetos": len(n_json), "nucleos": n_json}, jf, indent=2)
                proc += 1
                print(f"[OK] {f} | Obj: {len(n_json)}")

        messagebox.showinfo("Concluído", f"Lote finalizado! {proc} imagens em PREPROCESSADAS.")

if __name__ == "__main__":
    root = tk.Tk(); root.state("zoomed"); FiltroApp(root); root.mainloop()