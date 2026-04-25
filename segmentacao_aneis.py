# segmentacao_aneis.py
import cv2
import numpy as np

from utils_scaling import ImageScaler
from analisa_concavidades_anel import RingConcavityAnalyzer


class RingSegmenterForced:
    def __init__(self, scaler: ImageScaler):
        self.scaler = scaler
        self.analyzer = RingConcavityAnalyzer(scaler)

    # ---------------------------------------------------------
    # Geometria básica
    # ---------------------------------------------------------
    def _ccw(self, A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def _intersect(self, A, B, C, D):
        return self._ccw(A, C, D) != self._ccw(B, C, D) and self._ccw(
            A, B, C
        ) != self._ccw(A, B, D)

    def _candidate_thicknesses(self, cut_type="EXT_INT"):
        if cut_type in ("EXT_INT", "FALLBACK"):
            bases = [2, 3, 4, 5, 6, 8, 10, 12, 14]
        else:  
            bases = [1, 2, 3, 4, 5, 6, 8, 10]

        vals = []
        for base in bases:
            t = max(1, int(round(base * self.scaler.scale_factor)))
            vals.append(t)
        return sorted(set(vals))

    def _kernel_odd_scaled(self, base, min_val=3):
        v = max(int(round(float(base) * self.scaler.scale_factor)), min_val)
        if v % 2 == 0:
            v += 1
        return v

    def _build_closed_segmentation_topology(
        self, img_topologia, close_kernel_base=3, close_iters=1
    ):
        if img_topologia is None:
            return None

        img = np.where(img_topologia > 0, 255, 0).astype(np.uint8)

        if close_kernel_base <= 1:
            return img

        k = self.scaler.scale_kernel(close_kernel_base, min_val=3, make_odd=True)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        img_closed = cv2.morphologyEx(
            img, cv2.MORPH_CLOSE, kernel, iterations=max(1, int(close_iters))
        )
        return np.where(img_closed > 0, 255, 0).astype(np.uint8)

    # ---------------------------------------------------------
    # Utilidades para componentes e JSON Logic
    # ---------------------------------------------------------
    def _get_label_near_point(self, labels, pt, radius=2):
        h, w = labels.shape[:2]
        x, y = int(pt[0]), int(pt[1])

        x0 = max(0, x - radius)
        y0 = max(0, y - radius)
        x1 = min(w, x + radius + 1)
        y1 = min(h, y + radius + 1)

        roi = labels[y0:y1, x0:x1]
        valid = roi[roi > 0]
        if valid.size == 0:
            return 0

        vals, counts = np.unique(valid, return_counts=True)
        return int(vals[np.argmax(counts)])

    def _component_mask_from_point(self, labels, pt):
        lbl = self._get_label_near_point(labels, pt, radius=2)
        if lbl <= 0:
            return None, 0

        comp_mask = np.zeros(labels.shape, dtype=np.uint8)
        comp_mask[labels == lbl] = 255
        return comp_mask, lbl

    def _get_ring_reference_center(self, resultado):
        ellipse = resultado.get("ellipse") if resultado else None
        if ellipse is not None:
            try:
                cx, cy = ellipse[0]
                return (int(round(cx)), int(round(cy)))
            except Exception:
                pass

        centro = resultado.get("centro") if resultado else None
        if centro is not None:
            try:
                return (int(round(centro[0])), int(round(centro[1])))
            except Exception:
                pass

        cont_externo = resultado.get("cont_externo") if resultado else None
        if cont_externo is not None:
            m = cv2.moments(cont_externo)
            if abs(m.get("m00", 0.0)) > 1e-8:
                return (
                    int(round(m["m10"] / m["m00"])),
                    int(round(m["m01"] / m["m00"])),
                )

        return (0, 0)

    def _extract_polygon_metrics_from_mask(self, mask):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
        metrics = []
        
        trava_poeira = max(1.0, self.scaler.scale_area(15.0))
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < trava_poeira:
                continue
                
            comp_mask = np.zeros_like(mask)
            comp_mask[labels == i] = 255
            
            conts, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if conts:
                cnt = max(conts, key=cv2.contourArea)
                area_final = cv2.contourArea(cnt)
                
                trava_final = max(1.0, self.scaler.scale_area(10.0))
                
                if area_final >= trava_final:
                    hull = cv2.convexHull(cnt)
                    area_hull = cv2.contourArea(hull)
                    solidez = (area_final / area_hull) if area_hull > 0 else 1.0
                    metrics.append({
                        "area": area_final,
                        "solidez": solidez
                    })
        return metrics

    # ---------------------------------------------------------
    # Linha reforçada de corte
    # ---------------------------------------------------------
    def _draw_reinforced_cut_mask(self, shape, p1, p2, thickness, cut_type="EXT_INT"):
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.line(mask, p1, p2, 255, thickness, cv2.LINE_8)

        if cut_type == "EXT_INT":
            extra_base = 3
        elif cut_type == "FALLBACK":
            extra_base = 3
        else:
            extra_base = 2

        k = self._kernel_odd_scaled(extra_base, min_val=3)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, kernel, iterations=1)

        return np.where(mask > 0, 255, 0).astype(np.uint8)

    def _apply_cut_mask(self, comp_mask, cut_mask):
        out = comp_mask.copy()
        out[cut_mask > 0] = 0
        return out

    # ---------------------------------------------------------
    # Simulação e validação de cortes
    # ---------------------------------------------------------
    def _sample_line_points(self, p1, p2, n=32):
        p1 = np.array(p1, dtype=np.float32)
        p2 = np.array(p2, dtype=np.float32)
        pts = []
        for t in np.linspace(0.0, 1.0, n):
            p = p1 * (1.0 - t) + p2 * t
            pts.append((int(round(p[0])), int(round(p[1]))))
        return pts

    def _get_local_cut_segment(self, comp_mask, p_ext, p_center):
        h, w = comp_mask.shape[:2]
        p_ext_arr = np.array(p_ext, dtype=np.float32)
        p_center_arr = np.array(p_center, dtype=np.float32)
        
        vec = p_center_arr - p_ext_arr
        dist = np.linalg.norm(vec)
        if dist < 1e-3:
            return p_ext, p_center
            
        dir_vec = vec / dist
        
        buffer_out = max(5.0, 10.0 * self.scaler.scale_factor)
        p_start = p_ext_arr - dir_vec * buffer_out
        
        total_dist = np.linalg.norm(p_center_arr - p_start)
        pts = self._sample_line_points(
            (int(p_start[0]), int(p_start[1])), 
            (int(p_center_arr[0]), int(p_center_arr[1])), 
            n=int(total_dist + 1)
        )
        
        inside_idxs = []
        for i, (x, y) in enumerate(pts):
            if 0 <= x < w and 0 <= y < h and comp_mask[y, x] > 0:
                inside_idxs.append(i)
                
        if not inside_idxs:
            return p_ext, p_center
            
        first_in = inside_idxs[0]
        last_in = inside_idxs[0]
        
        for idx in inside_idxs[1:]:
            if idx <= last_in + 4: 
                last_in = idx
            else:
                break
                
        buffer_in = max(5.0, 10.0 * self.scaler.scale_factor)
        cut_start_idx = max(0, first_in - int(buffer_out))
        cut_end_idx = min(len(pts) - 1, last_in + int(buffer_in))
        
        return pts[cut_start_idx], pts[cut_end_idx]

    def _simulate_cut(self, comp_mask, p1, p2, thickness, cut_type="EXT_INT"):
        cut_mask = self._draw_reinforced_cut_mask(
            comp_mask.shape, p1, p2, thickness, cut_type=cut_type
        )
        test_mask = self._apply_cut_mask(comp_mask, cut_mask)

        nl, _, st, _ = cv2.connectedComponentsWithStats(test_mask, connectivity=8)
        areas = st[1:, cv2.CC_STAT_AREA] if nl > 1 else np.array([], dtype=np.int32)
        
        trava_poeira = max(1.0, self.scaler.scale_area(15.0))
        areas_significativas = areas[areas >= trava_poeira]

        removed_pixels = int(np.count_nonzero(comp_mask) - np.count_nonzero(test_mask))

        return {
            "mask": test_mask,
            "cut_mask": cut_mask,
            "num_components": max(0, nl - 1),
            "areas": areas,
            "min_area": (
                float(np.min(areas_significativas))
                if len(areas_significativas) > 0
                else 0.0
            ),
            "removed_pixels": removed_pixels,
        }

    def _evaluate_post_cut_metrics(self, comp_mask, p1, p2, thickness, expected_area):
        cut_mask = self._draw_reinforced_cut_mask(comp_mask.shape, p1, p2, thickness)
        test_mask = self._apply_cut_mask(comp_mask, cut_mask)

        metrics = self._extract_polygon_metrics_from_mask(test_mask)
        if len(metrics) <= 1:
            return 0.0

        score_shape = 0.0
        score_size = 0.0
        valid_parts = 0

        for m in metrics:
            area = m["area"]
            solidez = m["solidez"]
            
            if expected_area > 0:
                ratio_size = min(area, expected_area) / max(area, expected_area)
            else:
                ratio_size = 1.0
                
            score_size += ratio_size
            score_shape += solidez
            valid_parts += 1

        if valid_parts == 0:
            return 0.0

        avg_shape = score_shape / valid_parts
        avg_size = score_size / valid_parts

        return (avg_shape * 0.6) + (avg_size * 0.4)

    def _line_crosses_object_mass(
        self, comp_mask, p1, p2, min_ratio=0.40, n_samples=32
    ):
        h, w = comp_mask.shape[:2]
        pts = self._sample_line_points(p1, p2, n=n_samples)

        vals = []
        for x, y in pts:
            if 0 <= x < w and 0 <= y < h:
                vals.append(1 if comp_mask[y, x] > 0 else 0)

        if not vals:
            return False, 0.0

        ratio = float(np.mean(vals))
        return ratio >= min_ratio, ratio

    def _find_best_valid_cut_thickness(
        self,
        comp_mask,
        p1,
        p2,
        area_minima_corte,
        cut_type="EXT_INT",
        min_removed_base=20,
    ):
        min_removed = max(5, int(round(min_removed_base * self.scaler.scale_factor)))

        if cut_type == "FALLBACK":
            ratio_exigido = 0.03
        elif cut_type == "EXT_INT":
            ratio_exigido = 0.35
        else:
            ratio_exigido = 0.25

        dist_linha = np.linalg.norm(np.array(p1) - np.array(p2))
        n_amostras = max(48, int(dist_linha))

        crosses_ok, cross_ratio = self._line_crosses_object_mass(
            comp_mask, p1, p2, min_ratio=ratio_exigido, n_samples=n_amostras
        )

        if not crosses_ok:
            return None, None

        best_thickness = None
        best_sim = None
        best_score = -1.0

        for thickness in self._candidate_thicknesses(cut_type=cut_type):
            sim = self._simulate_cut(comp_mask, p1, p2, thickness, cut_type=cut_type)

            if sim["removed_pixels"] < min_removed:
                continue

            metrics = self._extract_polygon_metrics_from_mask(sim["mask"])
            if len(metrics) == 0:
                continue

            areas_limpas = [m["area"] for m in metrics]

            if area_minima_corte > 0:
                area_da_menor_parte = min(areas_limpas)
                if area_da_menor_parte < area_minima_corte:
                    continue

            if sim["num_components"] <= 1:
                if cut_type == "EXT_EXT":
                    continue

            if cut_type in ("EXT_INT", "FALLBACK"):
                score = (
                    float(sim["removed_pixels"]) * 1.0
                    + float(sim["num_components"]) * 500.0
                )
            else:
                score = (
                    float(sim["num_components"]) * 1000.0
                    + float(sim["removed_pixels"]) * 0.5
                    + float(sim["min_area"]) * 0.1
                )

            if score > best_score:
                best_score = score
                best_thickness = thickness
                best_sim = sim

        return best_thickness, best_sim

    def _find_best_simple_fallback_cut_thickness(
        self,
        comp_mask,
        p1,
        p2,
        area_minima_bolinha,
        area_referencia_bolinha=0.0,
        log_prefix="[FALLBACK]",
        area_max_factor=1.75,
        min_removed_base=10,
    ):
        min_removed = max(3, int(round(min_removed_base * self.scaler.scale_factor)))

        dist_linha = float(np.linalg.norm(np.array(p1, dtype=np.float32) - np.array(p2, dtype=np.float32)))
        if dist_linha < 2.0:
            return None, None

        n_amostras = max(10, int(round(dist_linha)))
        crosses_ok, cross_ratio = self._line_crosses_object_mass(
            comp_mask, p1, p2, min_ratio=0.10, n_samples=n_amostras
        )

        if not crosses_ok:
            return None, None

        area_max_bolinha = None
        if area_referencia_bolinha and area_referencia_bolinha > 0:
            area_max_bolinha = max(
                float(area_minima_bolinha),
                float(area_referencia_bolinha) * float(area_max_factor),
            )

        best_thickness = None
        best_sim = None
        best_score = -1.0
        best_areas_str = ""

        for thickness in self._candidate_thicknesses(cut_type="FALLBACK"):
            sim = self._simulate_cut(
                comp_mask, p1, p2, thickness, cut_type="FALLBACK"
            )

            if sim["removed_pixels"] < min_removed:
                continue

            metrics = self._extract_polygon_metrics_from_mask(sim["mask"])
            
            # Poeira < 15px já foi filtrada internamente pelo método de métricas
            pedacos_reais = [m for m in metrics]

            if len(pedacos_reais) == 0:
                continue

            if len(pedacos_reais) == 1:
                # FENDA: O corte abriu um buraco, mas manteve a massa conectada.
                remaining_area = pedacos_reais[0]["area"]
                if remaining_area < area_minima_bolinha:
                    continue  
                score_qualidade = 0.5 + (sim["removed_pixels"] * 0.001) 
            else:
                pedacos_reais.sort(key=lambda x: x["area"])
                detached_area = pedacos_reais[0]["area"]
                
                # NENHUM pedaço real pode ser menor que a área estrita permitida!
                if detached_area < area_minima_bolinha:
                    continue
                    
                if area_max_bolinha and detached_area > area_max_bolinha:
                    continue
                    
                ratio_area = min(detached_area, area_referencia_bolinha) / max(detached_area, area_referencia_bolinha) if area_referencia_bolinha > 0 else 1.0
                score_qualidade = 1.0 + (ratio_area * 0.4) + (pedacos_reais[0]["solidez"] * 0.6)

            if score_qualidade > best_score:
                best_score = score_qualidade
                best_thickness = thickness
                best_sim = sim
                best_areas_str = " | ".join([f"{m['area']:.1f}px" for m in pedacos_reais])

        if best_thickness is not None:
            print(f"{log_prefix} Aprovado th={best_thickness} | score={best_score:.3f} | Fragmentos gerados: [{best_areas_str}]")
            
        return best_thickness, best_sim

    # ---------------------------------------------------------
    # Segmentação Principal
    # ---------------------------------------------------------
    def segmentar_forcado(
        self,
        img_in,
        limiar_concavidade=3.0,
        max_dist_base=120,
        min_score=0.60,
        fator_detalhe=8.0,
        auto_binarizar=True,
        fator_area_minima=0.25,
        limiar_tangente=0.45,
        **kwargs
    ):
        if len(img_in.shape) == 3:
            gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_in.copy()

        if auto_binarizar:
            blur = cv2.medianBlur(gray, 5)
            _, img_bin = cv2.threshold(
                blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
            )
        else:
            _, img_bin = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        res = self.analyzer.processar(
            img_bin,
            limiar_px_ext=limiar_concavidade,
            limiar_px_int=limiar_concavidade,
            fator_detalhe=fator_detalhe,
            **kwargs
        )

        if not res:
            return cv2.cvtColor(img_bin, cv2.COLOR_GRAY2BGR), img_bin, 0, []

        img_topologia = res.get("img_topologia", img_bin).copy()
        ellipse_mask = res.get("ellipse_mask")
        img_bin_original = res.get("img_bin_original", img_bin).copy()

        img_segmentacao_base = self._build_closed_segmentation_topology(
            img_topologia, close_kernel_base=3, close_iters=1
        )

        conc_ext = res["conc_ext"]
        conc_int = res["conc_int"]
        centro_ref = self._get_ring_reference_center(res)

        img_debug = cv2.cvtColor(img_topologia, cv2.COLOR_GRAY2BGR)
        cv2.circle(img_debug, centro_ref, 3, (0, 255, 255), -1)

        LIMIT_DIST_MATCH = self.scaler.scale_scalar(max_dist_base)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            img_segmentacao_base, connectivity=8
        )

        print("\n" + "=" * 60)
        print(f"[SEGMENTAÇÃO GLOBAL] Fator Detalhe: {fator_detalhe} | Limiar Conc.: {limiar_concavidade}")
        print(f"Limites - Max Dist: {LIMIT_DIST_MATCH:.1f} | Min Score: {min_score} | Limiar Tangente: {limiar_tangente}")
        print("=" * 60)

        pares_potenciais = []

        # =====================================================
        # Pareamento EXT_INT Convencional
        # =====================================================
        for c_ext in conc_ext:
            pext = np.array(c_ext["ponto"])
            vext = np.array(c_ext["vetor_dir"])

            vetor_centro = np.array(
                [centro_ref[0] - pext[0], centro_ref[1] - pext[1]], dtype=np.float64
            )
            norm_centro = np.linalg.norm(vetor_centro)
            dir_centro = vetor_centro / norm_centro if norm_centro > 0 else vext

            if conc_int:
                for c_int in conc_int:
                    pint = np.array(c_int["ponto"])
                    vint = np.array(c_int["vetor_dir"])

                    dist = np.linalg.norm(pint - pext)
                    if dist > LIMIT_DIST_MATCH or dist == 0:
                        continue

                    conn_dir = (pint - pext) / dist
                    score_ortogonal_ext = np.dot(vext, conn_dir)
                    score_ortogonal_int = np.dot(vint, -conn_dir)
                    score_radial = np.dot(conn_dir, dir_centro)
                    score_paralelismo = -np.dot(vext, vint)

                    if score_ortogonal_ext < 0.64 or score_ortogonal_int < 0.35:
                        continue

                    midpoint = (pext + pint) / 2.0
                    vec_mid_center = np.array(centro_ref) - midpoint
                    norm_mid = np.linalg.norm(vec_mid_center)
                    score_radial_mid = 0.0
                    if norm_mid > 0:
                        dir_mid_center = vec_mid_center / norm_mid
                        score_radial_mid = abs(np.dot(conn_dir, dir_mid_center))
                        if score_radial_mid < limiar_tangente:
                            continue

                    fator_distancia = 1.0 - (dist / LIMIT_DIST_MATCH)
                    bonus_proximidade = 0.5 if fator_distancia > 0.85 else 0.0

                    score_final = (
                        (score_ortogonal_ext * 0.40)
                        + (score_ortogonal_int * 0.25)
                        + (fator_distancia * 0.20)
                        + (abs(score_radial) * 0.10)
                        + (score_paralelismo * 0.05)
                        + bonus_proximidade
                    )

                    if score_final > min_score:
                        pares_potenciais.append(
                            {
                                "c1": c_ext,
                                "c2": c_int,
                                "idx1": c_ext["idx"],
                                "idx2": c_int["idx"],
                                "score": score_final,
                                "dist": dist,
                                "tipo": "EXT_INT",
                                "score_radial_mid": score_radial_mid,
                            }
                        )

        pares_potenciais.sort(key=lambda x: x["score"], reverse=True)

        pares_sem_cruzamentos_ruins = []
        for i, parA in enumerate(pares_potenciais):
            p1A = (int(parA["c1"]["ponto"][0]), int(parA["c1"]["ponto"][1]))
            p2A = (int(parA["c2"]["ponto"][0]), int(parA["c2"]["ponto"][1]))

            mantem_A = True
            for j, parB in enumerate(pares_potenciais):
                if i == j: continue
                p1B = (int(parB["c1"]["ponto"][0]), int(parB["c1"]["ponto"][1]))
                p2B = (int(parB["c2"]["ponto"][0]), int(parB["c2"]["ponto"][1]))

                if self._intersect(p1A, p2A, p1B, p2B):
                    if parB["score_radial_mid"] > (parA["score_radial_mid"] + 0.02):
                        mantem_A = False
                        break

            if mantem_A:
                pares_sem_cruzamentos_ruins.append(parA)

        pares_potenciais = pares_sem_cruzamentos_ruins

        img_teste_cortes = img_segmentacao_base.copy()
        for par in pares_potenciais:
            p1_teste = (int(par["c1"]["ponto"][0]), int(par["c1"]["ponto"][1]))
            p2_teste = (int(par["c2"]["ponto"][0]), int(par["c2"]["ponto"][1]))
            cv2.line(img_teste_cortes, p1_teste, p2_teste, 0, 2, cv2.LINE_8)

        metrics_teste = self._extract_polygon_metrics_from_mask(img_teste_cortes)
        areas_teste_validas = [m["area"] for m in metrics_teste if m["area"] > 50]

        mediana_area_dinamica = np.median(areas_teste_validas) if len(areas_teste_validas) > 0 else 0
        area_minima_corte_dinamica = mediana_area_dinamica * fator_area_minima
        
        limite_corte_seguro = mediana_area_dinamica * 1.25

        print(f"[ÁREA] Mediana inicial simulada: {mediana_area_dinamica:.1f}px")

        for par in pares_potenciais:
            p1_tuple = (int(par["c1"]["ponto"][0]), int(par["c1"]["ponto"][1]))
            p2_tuple = (int(par["c2"]["ponto"][0]), int(par["c2"]["ponto"][1]))

            comp_mask, _ = self._component_mask_from_point(labels, p1_tuple)
            if comp_mask is None: continue

            thickness_teste = max(2, int(round(3 * self.scaler.scale_factor)))
            visual_score = self._evaluate_post_cut_metrics(
                comp_mask, p1_tuple, p2_tuple, thickness_teste, mediana_area_dinamica
            )
            par["score"] = (par["score"] * 0.3) + (visual_score * 0.7)

        pares_potenciais.sort(key=lambda x: x["score"], reverse=True)

        cortes_finais = []
        ext_usados = set()
        int_usados = set()

        for par in pares_potenciais:
            idx1 = par["idx1"]
            idx2 = par["idx2"]
            tipo = par["tipo"]

            if tipo == "EXT_INT":
                if idx1 in ext_usados or idx2 in int_usados:
                    continue

            p1_tuple = (int(par["c1"]["ponto"][0]), int(par["c1"]["ponto"][1]))
            p2_tuple = (int(par["c2"]["ponto"][0]), int(par["c2"]["ponto"][1]))

            cruzou = False
            for corte_aceito in cortes_finais:
                if self._intersect(p1_tuple, p2_tuple, corte_aceito["p1"], corte_aceito["p2"]):
                    cruzou = True
                    break
            if cruzou: continue

            comp_mask, lbl = self._component_mask_from_point(labels, p1_tuple)
            if comp_mask is None: continue
            
            area_comp = np.count_nonzero(comp_mask)
            if area_comp <= limite_corte_seguro:
                continue

            thickness, sim = self._find_best_valid_cut_thickness(
                comp_mask, p1_tuple, p2_tuple, area_minima_corte_dinamica,
                cut_type=tipo, min_removed_base=24 if tipo == "EXT_INT" else 14,
            )

            if thickness is None: continue

            ext_usados.add(idx1)
            int_usados.add(idx2)

            cortes_finais.append({
                "p1": p1_tuple,
                "p2": p2_tuple,
                "tipo": tipo,
                "thickness": thickness,
                "cut_mask": sim["cut_mask"],
            })
            print(f"  => [EXT_INT APROVADO] {p1_tuple} <-> {p2_tuple}")

        # =================================================================
        # ATUALIZAÇÃO DA TOPOLOGIA PARA O FALLBACK 
        # =================================================================
        img_cortada_parcial = img_segmentacao_base.copy()
        espessura_corte_fisico = max(2, int(round(2 * self.scaler.scale_factor)))

        for c in cortes_finais:
            cv2.line(img_cortada_parcial, c["p1"], c["p2"], 0, espessura_corte_fisico, cv2.LINE_8)

        num_labels_parcial, labels_parcial, _, _ = cv2.connectedComponentsWithStats(
            img_cortada_parcial, connectivity=8
        )

        # =====================================================
        # FALLBACK: Usando a topologia parcialmente cortada
        # =====================================================
        print("\n--- INICIANDO FALLBACK EM MÁSCARA PARCIAL ---")
        for c_ext in conc_ext:
            idx_ext = c_ext["idx"]
            if idx_ext in ext_usados:
                continue

            pext_tuple = (int(c_ext["ponto"][0]), int(c_ext["ponto"][1]))
            p_final_tuple = (int(centro_ref[0]), int(centro_ref[1]))

            comp_mask, lbl = self._component_mask_from_point(labels_parcial, pext_tuple)
            if comp_mask is None:
                continue

            area_comp = np.count_nonzero(comp_mask)
            if area_comp <= limite_corte_seguro:
                continue

            p_cut_start, p_cut_end = self._get_local_cut_segment(comp_mask, pext_tuple, p_final_tuple)

            cruzou = False
            for corte_aceito in cortes_finais:
                if self._intersect(p_cut_start, p_cut_end, corte_aceito["p1"], corte_aceito["p2"]):
                    cruzou = True
                    break
            if cruzou: continue

            thickness, sim = self._find_best_simple_fallback_cut_thickness(
                comp_mask, p_cut_start, p_cut_end, 
                area_minima_corte_dinamica,
                area_referencia_bolinha=mediana_area_dinamica,
                log_prefix=f"[FALLBACK][idx={idx_ext}]"
            )

            if thickness is None:
                continue

            cortes_finais.append({
                "p1": p_cut_start,
                "p2": p_cut_end,
                "tipo": "FALLBACK",
                "thickness": thickness,
                "cut_mask": sim["cut_mask"],
            })
            
            cv2.line(img_cortada_parcial, p_cut_start, p_cut_end, 0, espessura_corte_fisico, cv2.LINE_8)
            _, labels_parcial, _, _ = cv2.connectedComponentsWithStats(img_cortada_parcial, connectivity=8)

        # =================================================================
        # SEGMENTAÇÃO FINA E RECONSTRUÇÃO DE POLÍGONOS
        # =================================================================
        print("\n--- REGISTRANDO NÚCLEOS NO JSON ---")
        img_cortada = img_segmentacao_base.copy()

        for c in cortes_finais:
            color = (0, 255, 0) if c["tipo"] == "EXT_INT" else (255, 0, 255)
            if c["tipo"] == "FALLBACK":
                color = (0, 0, 255)

            cv2.line(img_debug, c["p1"], c["p2"], color, 1, cv2.LINE_AA)
            cv2.circle(img_debug, c["p1"], 3, (0, 0, 255), -1)
            cv2.circle(img_debug, c["p2"], 3, color, -1)

            cv2.line(img_cortada, c["p1"], c["p2"], 0, espessura_corte_fisico, cv2.LINE_8)

        if ellipse_mask is not None:
            pontes_artificiais = cv2.bitwise_and(ellipse_mask, cv2.bitwise_not(img_bin_original))
            img_cortada[pontes_artificiais > 0] = 0

        massa_final_permitida = img_segmentacao_base.copy()
        if ellipse_mask is not None:
            massa_final_permitida[pontes_artificiais > 0] = 0

        num_labels_cut, labels_cut, stats_cut, centroids_cut = cv2.connectedComponentsWithStats(
            img_cortada, connectivity=4
        )

        objetos_json = []
        mask_final_reconstruida = np.zeros_like(img_segmentacao_base)
        
        # Qualquer objeto estritamente menor que a poeira tolerável é descartado. 
        # Não há mais subtração cega da máscara reconstruída.
        trava_poeira = max(5.0, self.scaler.scale_area(15.0))

        for i in range(1, num_labels_cut):
            area_bruta = int(stats_cut[i, cv2.CC_STAT_AREA])

            if area_bruta < trava_poeira: continue

            comp_isolado = np.zeros_like(img_segmentacao_base)
            comp_isolado[labels_cut == i] = 255

            contornos, _ = cv2.findContours(comp_isolado, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contornos: continue

            maior_contorno = max(contornos, key=cv2.contourArea)

            comp_preenchido = np.zeros_like(img_segmentacao_base)
            cv2.drawContours(comp_preenchido, [maior_contorno], -1, 255, thickness=cv2.FILLED)

            kernel_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            comp_dilatado = cv2.dilate(comp_preenchido, kernel_dil, iterations=1)

            # Restringe ao tamanho original para a dilatação não vazar a borda,
            # mas PERMITE que encoste em outros objetos adjacentes sem apagá-los!
            comp_recuperado = cv2.bitwise_and(comp_dilatado, massa_final_permitida)

            contornos_finais, _ = cv2.findContours(comp_recuperado, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contornos_finais: continue

            contorno_final_json = max(contornos_finais, key=cv2.contourArea)
            pontos = contorno_final_json[:, 0, :].tolist()

            area_final = int(cv2.contourArea(contorno_final_json))
            
            if area_final < trava_poeira: continue

            cx, cy = centroids_cut[i]
            nucleo_id = len(objetos_json) + 1

            objetos_json.append({
                "id": nucleo_id,
                "centroide": [float(cx), float(cy)],
                "area": area_final,
                "pontos": pontos,
            })
            
            print(f"  -> Núcleo {nucleo_id:03d} registrado: Área = {area_final:.1f}px")

            mask_final_reconstruida = cv2.bitwise_or(mask_final_reconstruida, comp_recuperado)

        print(f"\n[FINAL] Cortes aceitos: {len(cortes_finais)} | Núcleos Registrados: {len(objetos_json)}")

        return img_debug, mask_final_reconstruida, len(cortes_finais), objetos_json


def aplicar_segmentacao_refinada(
    img_atual, scaler, limiar_concavidade=3.0, max_dist=120, min_score=0.60,
    fator_detalhe=8.0, auto_binarizar=True, fator_area_minima=0.25, limiar_tangente=0.45,
    **kwargs
):
    seg = RingSegmenterForced(scaler)
    debug, mask, n_cortes, nucleos_json = seg.segmentar_forcado(
        img_atual, limiar_concavidade=limiar_concavidade, max_dist_base=max_dist,
        min_score=min_score, fator_detalhe=fator_detalhe, auto_binarizar=auto_binarizar,
        fator_area_minima=fator_area_minima, limiar_tangente=limiar_tangente,
        **kwargs
    )
    return debug, mask, n_cortes, nucleos_json