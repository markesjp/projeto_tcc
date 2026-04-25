# gera_resultados.py
import os
import json
import csv
import cv2
import numpy as np

from utils_scaling import ImageScaler
from receita_pipeline import RecipeRunner
from segmentacao_aneis import aplicar_segmentacao_refinada
from auxiliares.concavities import converte_em_cinza


class AvaliadorTCC:
    def __init__(self, pasta_dataset, caminho_receita):
        self.pasta_dataset = pasta_dataset

        self.scaler = ImageScaler(reference_dim=1000.0)
        self.runner = RecipeRunner(self.scaler)
        self.runner.carregar_receita_json(caminho_receita)

        params_finais = self.runner.receita_atual.get("parametros_analise_final", {})
        self.limiar_conc = float(params_finais.get("limiar_concavidade", 3.0))
        self.max_dist = int(params_finais.get("max_dist", 120))
        self.min_score = float(params_finais.get("min_score", 0.60))
        self.fator_det = float(params_finais.get("fator_detalhe", 8.0))
        self.fator_area = float(params_finais.get("fator_area_minima", 0.25))
        self.limiar_tang = float(params_finais.get("limiar_tangente", 0.45))

        self.resultados = []

    def _ensure_uint8(self, img):
        if img is None:
            return None
        if img.dtype == np.uint8:
            return img
        if np.max(img) <= 1.0 and img.dtype != np.uint8:
            img = img * 255.0
        return np.clip(img, 0, 255).astype(np.uint8)

    def _to_bgr(self, img):
        img = self._ensure_uint8(img)
        if img is None:
            return None
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img.copy()

    def _draw_contorno_via_json(self, base_img, nucleos_json, thickness=1, color=(0, 255, 0)):
        if base_img is None:
            return None
            
        base = self._to_bgr(base_img)
        draw_thickness = max(1, int(round(thickness * self.scaler.scale_factor)))
        
        for nucleo in nucleos_json:
            pts = np.array(nucleo["pontos"], dtype=np.int32)
            cv2.drawContours(base, [pts], -1, color, draw_thickness, lineType=cv2.LINE_AA)
            
        return base

    def _gerar_mascara_anylabeling(self, caminho_json, shape_escalado):
        target_h, target_w = shape_escalado[:2]
        mascara_gt_instancia = np.zeros((target_h, target_w), dtype=np.int32)

        if not os.path.exists(caminho_json):
            print(f"[AVISO] JSON não encontrado: {caminho_json}")
            return mascara_gt_instancia, []

        with open(caminho_json, "r", encoding="utf-8") as f:
            dados = json.load(f)

        json_w = dados.get("imageWidth")
        json_h = dados.get("imageHeight")

        if not json_w or not json_h:
            json_w, json_h = target_w, target_h

        label_id = 1
        gt_json = []

        for shape in dados.get("shapes", []):
            pontos = shape.get("points", [])
            if not pontos:
                continue

            pontos_projetados = []
            for x, y in pontos:
                x_proj = int(round((x / json_w) * target_w))
                y_proj = int(round((y / json_h) * target_h))
                pontos_projetados.append([x_proj, y_proj])

            pts_array = np.array(pontos_projetados, dtype=np.int32)
            cv2.fillPoly(mascara_gt_instancia, [pts_array], label_id)
            
            gt_json.append({
                "id": label_id,
                "pontos": pontos_projetados
            })
            
            label_id += 1

        return mascara_gt_instancia, gt_json

    def calcular_metricas_instancia(self, labels_pred, mask_gt_instancia, qtd_gt_json, iou_threshold=0.5):
        ids_pred = np.unique(labels_pred)
        ids_pred = ids_pred[ids_pred > 0]
        num_pred_reais = len(ids_pred)

        ids_gt = np.unique(mask_gt_instancia)
        ids_gt = ids_gt[ids_gt > 0]

        gt_matched = set()
        pred_matched = set()
        tp = 0

        for i_pred in ids_pred:
            mascara_obj_pred = (labels_pred == i_pred).astype(np.uint8)
            best_iou = 0
            best_gt_match = -1

            for i_gt in ids_gt:
                if i_gt in gt_matched:
                    continue
                mascara_obj_gt = (mask_gt_instancia == i_gt).astype(np.uint8)

                intersecao = np.logical_and(mascara_obj_pred, mascara_obj_gt).sum()
                if intersecao > 0:
                    uniao = np.logical_or(mascara_obj_pred, mascara_obj_gt).sum()
                    iou = intersecao / (uniao + 1e-6)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_match = i_gt

            if best_iou >= iou_threshold:
                tp += 1
                gt_matched.add(best_gt_match)
                pred_matched.add(i_pred)

        fp = num_pred_reais - len(pred_matched)
        fn = qtd_gt_json - len(gt_matched)

        return tp, fp, fn, num_pred_reais

    def calcular_metricas_pixel(self, mask_pred_bin, mask_gt_instancia):
        pred_bool = mask_pred_bin > 127
        gt_bool = mask_gt_instancia > 0

        tp_pix = np.logical_and(pred_bool, gt_bool).sum()
        fp_pix = np.logical_and(pred_bool, ~gt_bool).sum()
        fn_pix = np.logical_and(~pred_bool, gt_bool).sum()

        massa_pred = pred_bool.sum()
        massa_gt = gt_bool.sum()

        f1_pixel = (2 * tp_pix) / ((2 * tp_pix) + fp_pix + fn_pix + 1e-6)

        return massa_pred, massa_gt, f1_pixel

    def gerar_imagem_inspecao(self, img_original, labels_pred, mask_gt_instancia, caminho_salvar):
        base_color = self._to_bgr(img_original)
        h, w = labels_pred.shape
        base_color = cv2.resize(base_color, (w, h))

        b = (labels_pred > 0).astype(np.uint8) * 255
        g = (mask_gt_instancia > 0).astype(np.uint8) * 255
        r = np.zeros_like(b)

        overlay = cv2.merge([b, g, r])
        base_escura = cv2.addWeighted(base_color, 0.4, np.zeros_like(base_color), 0, 0)
        resultado_visual = cv2.addWeighted(base_escura, 1.0, overlay, 0.6, 0)

        cv2.imwrite(caminho_salvar, resultado_visual)

    def avaliar_dataset(self):
        pasta_saida = os.path.join(os.getcwd(), "output_process")
        os.makedirs(pasta_saida, exist_ok=True)
        csv_saida = os.path.join(pasta_saida, "relatorio_metricas.csv")
        
        params_finais = self.runner.receita_atual.get("parametros_analise_final", {})
        topologia_kwargs = params_finais.get("topologia_refinada", {})

        extensoes = [".png", ".jpg", ".jpeg", ".tif", ".bmp"]
        print("Iniciando avaliação rigorosa com busca recursiva...\n")
        processadas = 0

        for root_dir, dirs, files in os.walk(self.pasta_dataset):
            for arquivo in files:
                nome_base, extensao = os.path.splitext(arquivo)

                if extensao.lower() not in extensoes:
                    continue

                caminho_img = os.path.join(root_dir, arquivo)
                caminho_json = os.path.join(root_dir, f"{nome_base}.json")

                if not os.path.exists(caminho_json):
                    continue

                caminho_relativo = os.path.relpath(root_dir, self.pasta_dataset)

                if caminho_relativo == ".":
                    pasta_imagem = os.path.join(pasta_saida, nome_base)
                else:
                    pasta_imagem = os.path.join(pasta_saida, caminho_relativo, nome_base)

                os.makedirs(pasta_imagem, exist_ok=True)

                img = cv2.imread(caminho_img, cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue

                img_gray = converte_em_cinza(img)
                self.scaler.update_from_image(img_gray)

                img_pre = self.runner.executar(img_gray)

                # Injeção dinâmica da topologia
                debug_img, mask_pred_bin, n_cortes, nucleos_json = aplicar_segmentacao_refinada(
                    img_pre.copy(),
                    self.scaler,
                    limiar_concavidade=self.limiar_conc,
                    max_dist=self.max_dist,
                    min_score=self.min_score,
                    fator_detalhe=self.fator_det,
                    auto_binarizar=False,
                    fator_area_minima=self.fator_area,
                    limiar_tangente=self.limiar_tang,
                    **topologia_kwargs
                )

                mask_pred_bin = self._ensure_uint8(mask_pred_bin)
                debug_img = self._ensure_uint8(debug_img)

                if mask_pred_bin.max() == 1:
                    mask_pred_bin *= 255
                    
                labels_pred = np.zeros(mask_pred_bin.shape, dtype=np.int32)
                for nucleo in nucleos_json:
                    pts = np.array(nucleo["pontos"], dtype=np.int32)
                    cv2.fillPoly(labels_pred, [pts], nucleo["id"])

                mask_gt_instancia, gt_json = self._gerar_mascara_anylabeling(caminho_json, mask_pred_bin.shape)
                qtd_gt = len(gt_json)

                tp, fp, fn, qtd_pred = self.calcular_metricas_instancia(labels_pred, mask_gt_instancia, qtd_gt)
                massa_pred, massa_gt, f1_pixel = self.calcular_metricas_pixel(mask_pred_bin, mask_gt_instancia)

                f1_instancia = (2 * tp) / ((2 * tp) + fp + fn + 1e-6)

                # Salvamento
                cv2.imwrite(os.path.join(pasta_imagem, f"{nome_base}_1_original.jpg"), img)
                cv2.imwrite(os.path.join(pasta_imagem, f"{nome_base}_2_pre_processada.jpg"), img_pre)

                mask_gt_vis = (mask_gt_instancia > 0).astype(np.uint8) * 255
                cv2.imwrite(os.path.join(pasta_imagem, f"{nome_base}_3_ground_truth.jpg"), mask_gt_vis)
                cv2.imwrite(os.path.join(pasta_imagem, f"{nome_base}_4_predicao_algoritmo.jpg"), mask_pred_bin)
                cv2.imwrite(os.path.join(pasta_imagem, f"{nome_base}_4_debug_segmentacao.jpg"), debug_img)

                gt_cont = self._draw_contorno_via_json(img_gray, gt_json, thickness=1, color=(0, 255, 0))
                pred_cont = self._draw_contorno_via_json(img_gray, nucleos_json, thickness=1, color=(255, 0, 255))
                debug_cont = self._draw_contorno_via_json(debug_img, nucleos_json, thickness=1, color=(255, 0, 255))

                if gt_cont is not None:
                    cv2.imwrite(os.path.join(pasta_imagem, f"{nome_base}_3b_ground_truth_contorno.jpg"), gt_cont)
                if pred_cont is not None:
                    cv2.imwrite(os.path.join(pasta_imagem, f"{nome_base}_4b_predicao_algoritmo_contorno.jpg"), pred_cont)
                if debug_cont is not None:
                    cv2.imwrite(os.path.join(pasta_imagem, f"{nome_base}_4c_debug_segmentacao_contorno.jpg"), debug_cont)

                caminho_visual = os.path.join(pasta_imagem, f"{nome_base}_5_inspecao_overlay.jpg")
                self.gerar_imagem_inspecao(img, labels_pred, mask_gt_instancia, caminho_visual)

                linha = {
                    "Amostra": arquivo,
                    "Caminho_Relativo": caminho_relativo,
                    "Massa_Referencia (px)": massa_gt,
                    "Massa_Algoritmo (px)": massa_pred,
                    "Qtd_Ref_AnyLabeling": qtd_gt,
                    "Qtd_Algoritmo": qtd_pred,
                    "Cortes_Geometricos": n_cortes,
                    "Acuracia_Objetos (TP)": tp,
                    "Falsos_Positivos (FP)": fp,
                    "Falsos_Negativos (FN)": fn,
                    "F1_Score_Instancia": round(f1_instancia, 4),
                    "F1_Score_Massa (Dice)": round(f1_pixel, 4)
                }
                self.resultados.append(linha)
                processadas += 1

                print(
                    f"[{arquivo}] F1-Objetos: {f1_instancia:.2f} | "
                    f"F1-Massa: {f1_pixel:.2f} | TP: {tp} | FP: {fp} | FN: {fn} | Cortes: {n_cortes}"
                )

        if self.resultados:
            chaves = self.resultados[0].keys()
            with open(csv_saida, "w", newline="", encoding="utf-8") as f_csv:
                writer = csv.DictWriter(f_csv, fieldnames=chaves, delimiter=";")
                writer.writeheader()
                writer.writerows(self.resultados)
            print(f"\n[SUCESSO] Relatório gerado! {processadas} amostras salvas na estrutura em: {pasta_saida}")
        else:
            print("\n[ERRO] Nenhum resultado gerado. Certifique-se de que as imagens e os JSONs estão nas subpastas e com o mesmo nome.")


if __name__ == "__main__":
    PASTA_DATASET = r"Dataset_TCC"
    CAMINHO_RECEITA = r"receita_oficial.json"

    avaliador = AvaliadorTCC(PASTA_DATASET, CAMINHO_RECEITA)
    avaliador.avaliar_dataset()