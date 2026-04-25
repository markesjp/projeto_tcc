# analisa_concavidades_anel.py
import logging
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from utils_scaling import ImageScaler


LOG_LEVEL = logging.INFO

logger = logging.getLogger("RingConcavityAnalyzer")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(LOG_LEVEL)
logger.propagate = False


class RingConcavityAnalyzer:
    def __init__(self, scaler: ImageScaler = None):
        self.scaler = scaler if scaler else ImageScaler()

    def _ordenar_contorno_sentido_horario(self, contorno, centro_ref):
        pts = contorno[:, 0, :].astype(np.float32)
        cx, cy = centro_ref

        dx = pts[:, 0] - cx
        dy = pts[:, 1] - cy

        ang = np.arctan2(-dy, dx)
        ang = np.mod(ang, 2 * np.pi)
        r = np.sqrt(dx * dx + dy * dy)

        ordem = np.argsort(ang)[::-1]
        return pts[ordem], ang[ordem], r[ordem]

    def _raio_eliptico_normalizado(self, pts, ellipse):
        if ellipse is None:
            cx = np.mean(pts[:, 0])
            cy = np.mean(pts[:, 1])
            dists = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
            return dists / (np.mean(dists) + 1e-6)

        (cx, cy), (MA, ma), angle = ellipse
        a = MA / 2.0
        b = ma / 2.0

        if a <= 0 or b <= 0:
            cx = np.mean(pts[:, 0])
            cy = np.mean(pts[:, 1])
            dists = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
            return dists / (np.mean(dists) + 1e-6)

        dx = pts[:, 0] - cx
        dy = pts[:, 1] - cy

        theta_rad = np.deg2rad(angle)
        cos_t = np.cos(theta_rad)
        sin_t = np.sin(theta_rad)

        x_rot = dx * cos_t + dy * sin_t
        y_rot = -dx * sin_t + dy * cos_t

        phi = np.arctan2(y_rot, x_rot)
        r_ponto = np.sqrt(x_rot ** 2 + y_rot ** 2)

        denom = np.sqrt((b * np.cos(phi)) ** 2 + (a * np.sin(phi)) ** 2)
        r_teorico = (a * b) / (denom + 1e-6)

        return r_ponto / (r_teorico + 1e-6)

    def _suaviza_sinal(self, r, janela_base=5, sigma_base=0):
        janela = self.scaler.scale_kernel(janela_base, min_val=3, make_odd=True)
        if len(r) < janela or len(r) < 3:
            return r

        half = janela // 2
        r_pad = np.concatenate([r[-half:], r, r[:half]])
        r_pad_mat = r_pad.reshape(1, -1).astype(np.float32)

        r_smooth = cv2.GaussianBlur(r_pad_mat, (janela, 1), sigmaX=sigma_base)
        return r_smooth.flatten()[half: half + len(r)]

    def _aplicar_segunda_derivada(
        self,
        r_s,
        fator_amplificacao=5.0,
        janela_maior_base=5,
        janela_menor_base=3,
    ):
        if len(r_s) < 5:
            return r_s

        janela_maior = self.scaler.scale_kernel(
            janela_maior_base,
            min_val=3,
            make_odd=True
        )
        janela_menor = self.scaler.scale_kernel(
            janela_menor_base,
            min_val=1,
            make_odd=True
        )

        fator_compensado = fator_amplificacao * (self.scaler.scale_factor ** 2)

        half = janela_maior // 2
        r_pad = np.concatenate([r_s[-half:], r_s, r_s[:half]])

        r_smooth = cv2.GaussianBlur(
            r_pad.reshape(1, -1).astype(np.float32),
            (janela_maior, 1),
            0
        ).flatten()

        d1 = np.gradient(r_smooth)
        d2 = np.gradient(d1)
        d2_core = d2[half: half + len(r_s)]

        r_enhanced = r_s - (fator_compensado * d2_core)

        if janela_menor <= 1:
            return r_enhanced

        half_min = janela_menor // 2
        r_final_pad = np.concatenate(
            [r_enhanced[-half_min:], r_enhanced, r_enhanced[:half_min]]
        )
        r_final = cv2.GaussianBlur(
            r_final_pad.reshape(1, -1).astype(np.float32),
            (janela_menor, 1),
            0
        ).flatten()

        return r_final[half_min: half_min + len(r_s)]

    def _fit_ellipse_safe(self, img_bin):
        pts_nonzero = cv2.findNonZero(img_bin)
        if pts_nonzero is None or len(pts_nonzero) < 5:
            return None
        try:
            ellipse = cv2.fitEllipse(pts_nonzero)
            (cx, cy), (MA, ma), angle = ellipse
            logger.info(
                f"RingConcavityAnalyzer | Elipse ajustada | centro=({cx:.2f}, {cy:.2f}) | eixos=({MA:.2f}, {ma:.2f}) | ang={angle:.2f}"
            )
            return ellipse
        except cv2.error:
            return None

    def _build_ellipse_mask(self, shape, ellipse, thickness_base=1):
        mask = np.zeros(shape, dtype=np.uint8)
        if ellipse is None:
            return mask

        thickness = self.scaler.scale_kernel(
            thickness_base,
            min_val=1,
            make_odd=False
        )
        cv2.ellipse(mask, ellipse, 255, thickness=thickness, lineType=cv2.LINE_AA)
        return np.where(mask > 0, 255, 0).astype(np.uint8)

    def _close_binary(self, img_bin, kernel_base=3, iterations=1):
        if kernel_base <= 1:
            return img_bin.copy()

        k = self.scaler.scale_kernel(kernel_base, min_val=3, make_odd=True)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    def _manter_componentes_pequenos(self, img_bin, area_max):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            img_bin,
            connectivity=8
        )
        out = np.zeros_like(img_bin)

        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area <= area_max:
                out[labels == label] = 255

        return out

    def _preencher_buracos_internos(self, img_bin):
        h, w = img_bin.shape[:2]
        flood = img_bin.copy()
        mask_ff = np.zeros((h + 2, w + 2), dtype=np.uint8)

        cv2.floodFill(flood, mask_ff, (0, 0), 255)
        flood_inv = cv2.bitwise_not(flood)
        filled = cv2.bitwise_or(img_bin, flood_inv)
        return np.where(filled > 0, 255, 0).astype(np.uint8)

    def _refinar_elipse_por_bolsoes(
        self,
        img_bin,
        ellipse,
        thickness_base=1,
        bolsoes_area_max_base=3000,
        bolsoes_dilate_kernel_base=7,
        bolsoes_dilate_iterations=2,
        intersection_dilate_kernel_base=3,
        intersection_dilate_iterations=1,
    ):
        ellipse_mask_original = self._build_ellipse_mask(
            img_bin.shape,
            ellipse,
            thickness_base=thickness_base
        )

        if ellipse is None or not np.any(ellipse_mask_original > 0):
            debug = {
                "ellipse_mask_original": ellipse_mask_original,
                "union_or": img_bin.copy(),
                "union_not": cv2.bitwise_not(img_bin),
                "bolsoes_small": np.zeros_like(img_bin),
                "bolsoes_dilatados": np.zeros_like(img_bin),
                "intersecao": np.zeros_like(img_bin),
                "intersecao_expandida": np.zeros_like(img_bin),
                "area_max": None,
            }
            return ellipse_mask_original, debug

        union_or = cv2.bitwise_or(img_bin, ellipse_mask_original)
        union_not = cv2.bitwise_not(union_or)

        area_max_scaled = max(
            1,
            int(round(float(bolsoes_area_max_base) * (self.scaler.scale_factor ** 2)))
        )

        bolsoes_small = self._manter_componentes_pequenos(
            union_not,
            area_max=area_max_scaled
        )

        hole_k = self.scaler.scale_kernel(
            bolsoes_dilate_kernel_base,
            min_val=3,
            make_odd=True
        )
        hole_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (hole_k, hole_k))
        bolsoes_dilatados = cv2.dilate(
            bolsoes_small,
            hole_kernel,
            iterations=max(1, int(bolsoes_dilate_iterations))
        )

        intersecao = cv2.bitwise_and(ellipse_mask_original, bolsoes_dilatados)

        inter_k = self.scaler.scale_kernel(
            intersection_dilate_kernel_base,
            min_val=1,
            make_odd=True
        )
        inter_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inter_k, inter_k))
        intersecao_expandida = cv2.dilate(
            intersecao,
            inter_kernel,
            iterations=max(1, int(intersection_dilate_iterations))
        )

        ellipse_mask_refinada = ellipse_mask_original.copy()
        ellipse_mask_refinada[intersecao_expandida > 0] = 0
        ellipse_mask_refinada = np.where(ellipse_mask_refinada > 0, 255, 0).astype(np.uint8)

        qtd_bolsoes = max(
            0,
            cv2.connectedComponentsWithStats(bolsoes_small, connectivity=8)[0] - 1
        )

        logger.info(
            "RingConcavityAnalyzer | Refino da elipse por bolsões | "
            f"area_max={area_max_scaled} | bolsoes={qtd_bolsoes} | "
            f"px_elipse_original={int(np.count_nonzero(ellipse_mask_original))} | "
            f"px_intersecao={int(np.count_nonzero(intersecao_expandida))} | "
            f"px_elipse_refinada={int(np.count_nonzero(ellipse_mask_refinada))}"
        )

        debug = {
            "ellipse_mask_original": ellipse_mask_original,
            "union_or": union_or,
            "union_not": union_not,
            "bolsoes_small": bolsoes_small,
            "bolsoes_dilatados": bolsoes_dilatados,
            "intersecao": intersecao,
            "intersecao_expandida": intersecao_expandida,
            "area_max": area_max_scaled,
        }

        return ellipse_mask_refinada, debug

    def _extrair_topologia_com_elipse(
        self,
        img_bin,
        ellipse,
        ellipse_thickness_base=1,
        topology_close_kernel_base=3,
        topology_close_iterations=1,
        use_ellipse_topology=True,
        refine_ellipse_with_holes=True,
        bolsoes_area_max_base=3000,
        bolsoes_dilate_kernel_base=7,
        bolsoes_dilate_iterations=2,
        intersection_dilate_kernel_base=3,
        intersection_dilate_iterations=1,
    ):
        img_abstrata = img_bin.copy()

        if refine_ellipse_with_holes:
            ellipse_mask, debug_ellipse = self._refinar_elipse_por_bolsoes(
                img_bin=img_bin,
                ellipse=ellipse,
                thickness_base=ellipse_thickness_base,
                bolsoes_area_max_base=bolsoes_area_max_base,
                bolsoes_dilate_kernel_base=bolsoes_dilate_kernel_base,
                bolsoes_dilate_iterations=bolsoes_dilate_iterations,
                intersection_dilate_kernel_base=intersection_dilate_kernel_base,
                intersection_dilate_iterations=intersection_dilate_iterations,
            )
        else:
            ellipse_mask = self._build_ellipse_mask(
                img_bin.shape,
                ellipse,
                thickness_base=ellipse_thickness_base
            )
            debug_ellipse = {
                "ellipse_mask_original": ellipse_mask.copy(),
                "union_or": cv2.bitwise_or(img_bin, ellipse_mask),
                "union_not": cv2.bitwise_not(cv2.bitwise_or(img_bin, ellipse_mask)),
                "bolsoes_small": np.zeros_like(img_bin),
                "bolsoes_dilatados": np.zeros_like(img_bin),
                "intersecao": np.zeros_like(img_bin),
                "intersecao_expandida": np.zeros_like(img_bin),
                "area_max": None,
            }

        if use_ellipse_topology and np.any(ellipse_mask > 0):
            img_abstrata = cv2.bitwise_or(img_abstrata, ellipse_mask)

        img_abstrata = self._close_binary(
            img_abstrata,
            kernel_base=topology_close_kernel_base,
            iterations=topology_close_iterations,
        )

        # para análise: mantém a topologia refinada como está
        img_topologia = img_abstrata.copy()

        # para segmentação: remove/preenche os bolsões restantes
        img_segmentacao_base = self._preencher_buracos_internos(img_abstrata)

        contornos, hier = cv2.findContours(
            img_topologia,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_NONE
        )
        if not contornos or hier is None:
            return None, None, ellipse_mask, img_topologia, img_segmentacao_base, debug_ellipse

        hier = hier[0]
        idx_ext = int(np.argmax([cv2.contourArea(c) for c in contornos]))
        cont_externo = contornos[idx_ext]

        cont_interno = None
        idx_child = hier[idx_ext][2]
        if idx_child != -1:
            best_area = 0.0
            curr = idx_child
            while curr != -1:
                a = cv2.contourArea(contornos[curr])
                if a > best_area:
                    best_area = a
                    cont_interno = contornos[curr]
                curr = hier[curr][0]

        return cont_externo, cont_interno, ellipse_mask, img_topologia, img_segmentacao_base, debug_ellipse

    def _estimar_centro_ref(self, contorno, ellipse):
        if ellipse is not None:
            cx, cy = ellipse[0]
            return int(cx), int(cy)

        if contorno is not None:
            M = cv2.moments(contorno)
            div = M["m00"] if M["m00"] != 0 else 1.0
            return int(M["m10"] / div), int(M["m01"] / div)

        return 0, 0

    def _sample_binary(self, img_bin, x, y):
        h, w = img_bin.shape[:2]
        xi = int(round(x))
        yi = int(round(y))
        if xi < 0 or yi < 0 or xi >= w or yi >= h:
            return 0
        return 1 if img_bin[yi, xi] > 0 else 0

    def _score_direction_on_binary(self, img_bin, pt, normal, steps=12, step_px=1.0):
        score = 0
        px, py = float(pt[0]), float(pt[1])

        for i in range(1, steps + 1):
            dist = i * step_px
            x = px + normal[0] * dist
            y = py + normal[1] * dist
            score += self._sample_binary(img_bin, x, y)

        return score

    def _orientar_normal_por_massa_local(
        self,
        pt,
        normal,
        img_bin,
        centro_ref=None,
        tipo="externa",
        steps_base=12,
        step_px_base=1.0,
    ):
        norm = np.linalg.norm(normal)
        if norm <= 1e-8:
            return np.array([1.0, 0.0], dtype=np.float32)

        normal = normal.astype(np.float32) / norm

        steps = max(3, int(round(steps_base * max(1.0, self.scaler.scale_factor))))
        step_px = max(1.0, float(step_px_base) * max(1.0, self.scaler.scale_factor))

        score_pos = self._score_direction_on_binary(
            img_bin, pt, normal, steps=steps, step_px=step_px
        )
        score_neg = self._score_direction_on_binary(
            img_bin, pt, -normal, steps=steps, step_px=step_px
        )

        if score_neg > score_pos:
            normal = -normal
            score_pos, score_neg = score_neg, score_pos

        if score_pos == score_neg and centro_ref is not None:
            cx, cy = centro_ref
            to_center = np.array([cx - pt[0], cy - pt[1]], dtype=np.float32)

            if np.linalg.norm(to_center) > 1e-8:
                to_center = to_center / (np.linalg.norm(to_center) + 1e-8)

                if tipo == "externa":
                    if np.dot(normal, to_center) < 0:
                        normal = -normal
                else:
                    if np.dot(normal, to_center) > 0:
                        normal = -normal

        return normal

    def _detecta_picos_scipy(
        self,
        r_s,
        pts_ord,
        ang_ord,
        centro_ref,
        tipo,
        limiar_manual,
        img_bin_ref,
        min_dist_base=1,
        span_normal_base=10,
        direction_steps_base=12,
        direction_step_px_base=1.0,
    ):
        concavidades = []
        if len(r_s) < 5:
            return concavidades

        prominence_val = max(float(limiar_manual), 0.00005)
        min_dist_idx = max(1, self.scaler.scale_kernel(min_dist_base, min_val=1))

        signal = -r_s if tipo == "externa" else r_s
        signal_periodic = np.concatenate([signal, signal, signal])
        N = len(signal)

        peaks_all, _ = find_peaks(
            signal_periodic,
            prominence=prominence_val,
            distance=min_dist_idx
        )

        valid_indices = []
        for pk in peaks_all:
            if N <= pk < 2 * N:
                valid_indices.append(pk - N)

        span_normal = max(2, self.scaler.scale_kernel(span_normal_base, min_val=3))

        for idx in valid_indices:
            val = r_s[idx]
            pt = pts_ord[idx]

            p_prev = pts_ord[(idx - span_normal) % N]
            p_next = pts_ord[(idx + span_normal) % N]

            tangente = p_next - p_prev
            norm_t = np.linalg.norm(tangente)

            if norm_t > 0:
                tangente = tangente / norm_t
                normal = np.array([-tangente[1], tangente[0]], dtype=np.float32)
            else:
                normal = np.array([1.0, 0.0], dtype=np.float32)

            normal = self._orientar_normal_por_massa_local(
                pt=pt,
                normal=normal,
                img_bin=img_bin_ref,
                centro_ref=centro_ref,
                tipo=tipo,
                steps_base=direction_steps_base,
                step_px_base=direction_step_px_base,
            )

            concavidades.append(
                {
                    "idx": int(idx),
                    "ponto": (int(pt[0]), int(pt[1])),
                    "profundidade": float(val),
                    "angulo": float(ang_ord[idx]),
                    "ponto_base": tuple(pt),
                    "vetor_dir": tuple(normal),
                    "tipo": tipo,
                }
            )

        logger.info(
            f"RingConcavityAnalyzer | Concavidades (Picos Reais) | tipo={tipo} | qtd={len(concavidades)} | prominence={prominence_val:.5f} | min_dist={min_dist_idx}"
        )

        return concavidades

    def processar(
        self,
        img_bin,
        limiar_px_ext=3.0,
        limiar_px_int=3.0,
        fator_detalhe=5.0,
        **kwargs
    ):
        img_bin = np.clip(img_bin, 0, 255).astype(np.uint8)

        ellipse_thickness_base = int(kwargs.get("ellipse_thickness_base", kwargs.get("thickness_base", 1)))
        topology_close_kernel_base = int(kwargs.get("topology_close_kernel_base", 3))
        topology_close_iterations = int(kwargs.get("topology_close_iterations", 1))
        use_ellipse_topology = bool(kwargs.get("use_ellipse_topology", True))

        refine_ellipse_with_holes = bool(kwargs.get("refine_ellipse_with_holes", True))
        bolsoes_area_max_base = int(kwargs.get("bolsoes_area_max_base", 3000))
        bolsoes_dilate_kernel_base = int(kwargs.get("bolsoes_dilate_kernel_base", 7))
        bolsoes_dilate_iterations = int(kwargs.get("bolsoes_dilate_iterations", 2))
        intersection_dilate_kernel_base = int(kwargs.get("intersection_dilate_kernel_base", 3))
        intersection_dilate_iterations = int(kwargs.get("intersection_dilate_iterations", 1))

        valley_strength_base = float(kwargs.get("valley_strength_base", 10.0))
        ellipse_gap_base = float(kwargs.get("ellipse_gap_base", 8.0))
        angle_guard_base = int(kwargs.get("angle_guard_base", 6))
        n_angulos = int(kwargs.get("n_angulos", 720))
        min_run_len = int(kwargs.get("min_run_len", 5))

        janela_suavizacao_ext = int(kwargs.get("janela_suavizacao_ext", 5))
        janela_suavizacao_int = int(kwargs.get("janela_suavizacao_int", 5))

        segunda_deriv_janela_maior_ext = int(kwargs.get("segunda_deriv_janela_maior_ext", 5))
        segunda_deriv_janela_menor_ext = int(kwargs.get("segunda_deriv_janela_menor_ext", 3))
        segunda_deriv_janela_maior_int = int(kwargs.get("segunda_deriv_janela_maior_int", 5))
        segunda_deriv_janela_menor_int = int(kwargs.get("segunda_deriv_janela_menor_int", 3))

        min_dist_picos_ext = int(kwargs.get("min_dist_picos_ext", 1))
        min_dist_picos_int = int(kwargs.get("min_dist_picos_int", 1))

        span_normal_ext = int(kwargs.get("span_normal_ext", 10))
        span_normal_int = int(kwargs.get("span_normal_int", 10))

        direction_steps_ext = int(kwargs.get("direction_steps_ext", 10))
        direction_step_px_ext = float(kwargs.get("direction_step_px_ext", 1.0))
        direction_steps_int = int(kwargs.get("direction_steps_int", 12))
        direction_step_px_int = float(kwargs.get("direction_step_px_int", 1.0))

        logger.info(
            "RingConcavityAnalyzer | Processar iniciado | "
            f"shape={img_bin.shape} | lim_ext={limiar_px_ext:.5f} | lim_int={limiar_px_int:.5f} | "
            f"fator={fator_detalhe:.4f} | ellipse_thick={ellipse_thickness_base} | "
            f"close_kernel={topology_close_kernel_base} | use_ellipse={use_ellipse_topology} | "
            f"refinar_elipse={refine_ellipse_with_holes}"
        )

        ellipse = self._fit_ellipse_safe(img_bin)

        cont_externo, cont_interno, ellipse_mask, img_topologia, img_segmentacao_base, debug_ellipse = (
            self._extrair_topologia_com_elipse(
                img_bin,
                ellipse,
                ellipse_thickness_base=ellipse_thickness_base,
                topology_close_kernel_base=topology_close_kernel_base,
                topology_close_iterations=topology_close_iterations,
                use_ellipse_topology=use_ellipse_topology,
                refine_ellipse_with_holes=refine_ellipse_with_holes,
                bolsoes_area_max_base=bolsoes_area_max_base,
                bolsoes_dilate_kernel_base=bolsoes_dilate_kernel_base,
                bolsoes_dilate_iterations=bolsoes_dilate_iterations,
                intersection_dilate_kernel_base=intersection_dilate_kernel_base,
                intersection_dilate_iterations=intersection_dilate_iterations,
            )
        )

        if cont_externo is None:
            logger.warning("RingConcavityAnalyzer | Sem contorno externo na topologia")
            return None

        centro_ref = self._estimar_centro_ref(cont_externo, ellipse)

        pts_ext, ang_ext, r_s_ext = [], [], []
        conc_ext = []

        if len(cont_externo) > 5:
            pts_ext, ang_ext, _ = self._ordenar_contorno_sentido_horario(
                cont_externo,
                centro_ref
            )

            if len(pts_ext) > 5:
                r_base_ext = self._suaviza_sinal(
                    self._raio_eliptico_normalizado(pts_ext, ellipse),
                    janela_base=janela_suavizacao_ext
                )

                r_s_ext = self._aplicar_segunda_derivada(
                    r_base_ext,
                    fator_amplificacao=fator_detalhe,
                    janela_maior_base=segunda_deriv_janela_maior_ext,
                    janela_menor_base=segunda_deriv_janela_menor_ext,
                )

                conc_ext = self._detecta_picos_scipy(
                    r_s_ext,
                    pts_ext,
                    ang_ext,
                    centro_ref,
                    "externa",
                    limiar_px_ext,
                    img_topologia,
                    min_dist_base=min_dist_picos_ext,
                    span_normal_base=span_normal_ext,
                    direction_steps_base=direction_steps_ext,
                    direction_step_px_base=direction_step_px_ext,
                )

        pts_int, ang_int, r_s_int = [], [], []
        conc_int = []

        if cont_interno is not None and len(cont_interno) > 5:
            pts_int, ang_int, _ = self._ordenar_contorno_sentido_horario(
                cont_interno,
                centro_ref
            )

            if len(pts_int) > 5:
                r_base_int = self._suaviza_sinal(
                    self._raio_eliptico_normalizado(pts_int, ellipse),
                    janela_base=janela_suavizacao_int
                )

                r_s_int = self._aplicar_segunda_derivada(
                    r_base_int,
                    fator_amplificacao=fator_detalhe,
                    janela_maior_base=segunda_deriv_janela_maior_int,
                    janela_menor_base=segunda_deriv_janela_menor_int,
                )

                conc_int = self._detecta_picos_scipy(
                    r_s_int,
                    pts_int,
                    ang_int,
                    centro_ref,
                    "interna",
                    limiar_px_int,
                    img_topologia,
                    min_dist_base=min_dist_picos_int,
                    span_normal_base=span_normal_int,
                    direction_steps_base=direction_steps_int,
                    direction_step_px_base=direction_step_px_int,
                )

        logger.info(
            f"RingConcavityAnalyzer | Processamento finalizado | conc_ext={len(conc_ext)} | conc_int={len(conc_int)}"
        )

        return {
            "cont_externo": cont_externo,
            "cont_interno": cont_interno,
            "pts_ext": pts_ext,
            "ang_ext": ang_ext,
            "sinal_diff_ext": r_s_ext,
            "pts_int": pts_int,
            "ang_int": ang_int,
            "sinal_diff_int": r_s_int,
            "conc_ext": conc_ext,
            "conc_int": conc_int,
            "centro": centro_ref,
            "ellipse": ellipse,
            "ellipse_mask": ellipse_mask,
            "img_topologia": img_topologia,
            "img_segmentacao_base": img_segmentacao_base,
            "img_bin_original": img_bin.copy(),
            "debug_ellipse_refinement": debug_ellipse,
            "debug_radial": {
                "invalid_mask": None,
                "angulos": np.linspace(0, 2 * np.pi, n_angulos, endpoint=False),
                "valley_strength_base": valley_strength_base,
                "ellipse_gap_base": ellipse_gap_base,
                "angle_guard_base": angle_guard_base,
                "min_run_len": min_run_len,
            },
            "params_runtime": {
                "ellipse_thickness_base": ellipse_thickness_base,
                "topology_close_kernel_base": topology_close_kernel_base,
                "topology_close_iterations": topology_close_iterations,
                "use_ellipse_topology": use_ellipse_topology,
                "refine_ellipse_with_holes": refine_ellipse_with_holes,
                "bolsoes_area_max_base": bolsoes_area_max_base,
                "bolsoes_dilate_kernel_base": bolsoes_dilate_kernel_base,
                "bolsoes_dilate_iterations": bolsoes_dilate_iterations,
                "intersection_dilate_kernel_base": intersection_dilate_kernel_base,
                "intersection_dilate_iterations": intersection_dilate_iterations,
                "janela_suavizacao_ext": janela_suavizacao_ext,
                "janela_suavizacao_int": janela_suavizacao_int,
                "segunda_deriv_janela_maior_ext": segunda_deriv_janela_maior_ext,
                "segunda_deriv_janela_menor_ext": segunda_deriv_janela_menor_ext,
                "segunda_deriv_janela_maior_int": segunda_deriv_janela_maior_int,
                "segunda_deriv_janela_menor_int": segunda_deriv_janela_menor_int,
                "min_dist_picos_ext": min_dist_picos_ext,
                "min_dist_picos_int": min_dist_picos_int,
                "span_normal_ext": span_normal_ext,
                "span_normal_int": span_normal_int,
                "direction_steps_ext": direction_steps_ext,
                "direction_step_px_ext": direction_step_px_ext,
                "direction_steps_int": direction_steps_int,
                "direction_step_px_int": direction_step_px_int,
                "valley_strength_base": valley_strength_base,
                "ellipse_gap_base": ellipse_gap_base,
                "angle_guard_base": angle_guard_base,
                "n_angulos": n_angulos,
                "min_run_len": min_run_len,
            }
        }

    # MÉTODO ATUALIZADO PARA DESENHO COM ANTI-ALIASING
    def desenhar_resultados(self, img_bgr, resultado, comprimento_vetor_base=40):
        if not resultado:
            return img_bgr

        base_ref = resultado.get("img_topologia")
        if base_ref is not None and len(base_ref.shape) == 2:
            vis = cv2.cvtColor(base_ref, cv2.COLOR_GRAY2BGR)
        else:
            vis = img_bgr.copy()

        if resultado.get("ellipse") is not None:
            cv2.ellipse(vis, resultado["ellipse"], (0, 255, 255), 1, cv2.LINE_AA)

        ellipse_mask = resultado.get("ellipse_mask")
        if ellipse_mask is not None and np.any(ellipse_mask > 0):
            conts_ellipse, _ = cv2.findContours(
                ellipse_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE
            )
            cv2.drawContours(vis, conts_ellipse, -1, (0, 200, 255), 1, lineType=cv2.LINE_AA)

        if resultado["cont_externo"] is not None:
            cv2.drawContours(vis, [resultado["cont_externo"]], -1, (0, 255, 0), 2, lineType=cv2.LINE_AA)
        if resultado["cont_interno"] is not None:
            cv2.drawContours(vis, [resultado["cont_interno"]], -1, (255, 0, 255), 2, lineType=cv2.LINE_AA)

        vetor_len = self.scaler.scale_scalar(comprimento_vetor_base)

        def draw_points(lista, cor_pt, cor_vec):
            for item in lista:
                pt = item["ponto"]
                base = item["ponto_base"]
                vec = item["vetor_dir"]

                cv2.circle(vis, pt, 5, cor_pt, -1, lineType=cv2.LINE_AA)
                end_pt = (
                    int(base[0] + vec[0] * vetor_len),
                    int(base[1] + vec[1] * vetor_len),
                )
                cv2.arrowedLine(
                    vis,
                    (int(base[0]), int(base[1])),
                    end_pt,
                    cor_vec,
                    2,
                    tipLength=0.3,
                    lineType=cv2.LINE_AA
                )

        draw_points(resultado["conc_ext"], (0, 0, 255), (0, 0, 255))
        draw_points(resultado["conc_int"], (255, 0, 0), (255, 0, 0))

        cx, cy = resultado["centro"]
        cv2.circle(vis, (int(cx), int(cy)), 3, (0, 255, 255), -1, lineType=cv2.LINE_AA)

        qtd_ext = len(resultado["conc_ext"])
        qtd_int = len(resultado["conc_int"])

        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (900, 190), (0, 0, 0), -1)
        vis = cv2.addWeighted(overlay, 0.4, vis, 0.6, 0)

        cv2.putText(
            vis,
            f"Conc. Externas: {qtd_ext}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            vis,
            f"Conc. Internas: {qtd_int}",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            vis,
            "Analise usando topologia refinada",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
            cv2.LINE_AA
        )

        return vis


def pipeline_anel(
    img_in,
    limiar_px_ext=3.0,
    limiar_px_int=3.0,
    fator_detalhe=5.0,
    auto_binarizar=True,
    **kwargs
):
    if len(img_in.shape) == 3:
        gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
        bgr = img_in.copy()
    else:
        gray = img_in.copy()
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    scaler = ImageScaler()
    scaler.update_from_image(gray)

    if auto_binarizar:
        blur = cv2.medianBlur(gray, 5)
        _, img_bin = cv2.threshold(
            blur,
            127,
            255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
    else:
        _, img_bin = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    analyzer = RingConcavityAnalyzer(scaler)
    res = analyzer.processar(
        img_bin,
        limiar_px_ext=limiar_px_ext,
        limiar_px_int=limiar_px_int,
        fator_detalhe=fator_detalhe,
        **kwargs
    )

    if not res:
        return bgr

    return analyzer.desenhar_resultados(bgr, res)


def analisar_dados_anel(
    img_in,
    limiar_px_ext=3.0,
    limiar_px_int=3.0,
    fator_detalhe=5.0,
    auto_binarizar=True,
    **kwargs
):
    if len(img_in.shape) == 3:
        gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_in.copy()

    scaler = ImageScaler()
    scaler.update_from_image(gray)

    if auto_binarizar:
        blur = cv2.medianBlur(gray, 5)
        _, img_bin = cv2.threshold(
            blur,
            127,
            255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
    else:
        _, img_bin = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    analyzer = RingConcavityAnalyzer(scaler)
    res = analyzer.processar(
        img_bin,
        limiar_px_ext=limiar_px_ext,
        limiar_px_int=limiar_px_int,
        fator_detalhe=fator_detalhe,
        **kwargs
    )

    img_ref = res["img_topologia"] if res and res.get("img_topologia") is not None else img_bin
    return img_ref, res, limiar_px_ext, limiar_px_int


def exibir_grafico_interativo(img_bin, resultado, limiar_viz_ext, limiar_viz_int):
    if not resultado:
        return

    img_vis = resultado.get("img_topologia", img_bin)

    pts_ext = resultado["pts_ext"]
    ang_ext = resultado["ang_ext"]
    r_ext = resultado["sinal_diff_ext"]
    conc_ext = resultado["conc_ext"]

    pts_int = resultado.get("pts_int", [])
    ang_int = resultado.get("ang_int", [])
    r_int = resultado.get("sinal_diff_int", [])
    conc_int = resultado.get("conc_int", [])

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2)

    ax_img = fig.add_subplot(gs[:, 0])
    ax_ext = fig.add_subplot(gs[0, 1])
    ax_int = fig.add_subplot(gs[1, 1])

    ax_img.set_title("Visualização Espacial")
    ax_img.imshow(img_vis, cmap="gray")
    ax_img.axis("off")

    if len(pts_ext) > 0:
        ax_img.plot(pts_ext[:, 0], pts_ext[:, 1], ",", color="green", alpha=0.5, label="Ext")
    if len(pts_int) > 0:
        ax_img.plot(pts_int[:, 0], pts_int[:, 1], ",", color="cyan", alpha=0.5, label="Int")

    for c in conc_ext:
        ax_img.plot(c["ponto"][0], c["ponto"][1], "o", color="red", markersize=6)
    for c in conc_int:
        ax_img.plot(c["ponto"][0], c["ponto"][1], "o", color="blue", markersize=6)

    ellipse_mask = resultado.get("ellipse_mask")
    if ellipse_mask is not None and np.any(ellipse_mask > 0):
        ys, xs = np.where(ellipse_mask > 0)
        ax_img.plot(xs[::12], ys[::12], ".", color="yellow", alpha=0.35, markersize=2)

    (marker_img,) = ax_img.plot([], [], "o", color="yellow", markersize=10, markeredgecolor="black", zorder=10)

    ax_ext.set_title("Sinal Externo Realçado")
    ax_ext.set_ylabel("Razão R / R_teórico")
    ax_ext.grid(True)
    if len(ang_ext) > 0:
        ax_ext.plot(ang_ext, r_ext, "g-", linewidth=1)
        ax_ext.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
        ax_ext.axhline(y=1.0 - limiar_viz_ext, color="red", linestyle=":", alpha=0.8, label="Limiar Corte")
        if conc_ext:
            cx = [ang_ext[c["idx"]] for c in conc_ext]
            cy = [r_ext[c["idx"]] for c in conc_ext]
            ax_ext.plot(cx, cy, "ro", label="Concavidades")

    (marker_ext,) = ax_ext.plot([], [], "o", color="yellow", markersize=8, zorder=10)

    ax_int.set_title("Sinal Interno Realçado")
    ax_int.set_xlabel("Ângulo (rad)")
    ax_int.set_ylabel("Razão R / R_teórico")
    ax_int.grid(True)
    if len(ang_int) > 0:
        ax_int.plot(ang_int, r_int, "c-", linewidth=1)
        ax_int.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
        ax_int.axhline(y=1.0 + limiar_viz_int, color="blue", linestyle=":", alpha=0.8, label="Limiar Corte")
        if conc_int:
            cx = [ang_int[c["idx"]] for c in conc_int]
            cy = [r_int[c["idx"]] for c in conc_int]
            ax_int.plot(cx, cy, "bo", label="Concavidades")

    (marker_int,) = ax_int.plot([], [], "o", color="yellow", markersize=8, zorder=10)

    text_info = fig.text(0.05, 0.02, "Mova o mouse...", fontsize=12, color="blue", fontweight="bold")

    def on_move(event):
        if event.inaxes == ax_ext and len(ang_ext) > 0:
            if event.xdata is None:
                return
            idx = (np.abs(ang_ext - event.xdata)).argmin()
            marker_ext.set_data([ang_ext[idx]], [r_ext[idx]])
            marker_img.set_data([pts_ext[idx][0]], [pts_ext[idx][1]])
            marker_int.set_data([], [])
            text_info.set_text(f"EXTERNO: Ang={ang_ext[idx]:.2f}, Razão={r_ext[idx]:.5f}")
            fig.canvas.draw_idle()

        elif event.inaxes == ax_int and len(ang_int) > 0:
            if event.xdata is None:
                return
            idx = (np.abs(ang_int - event.xdata)).argmin()
            marker_int.set_data([ang_int[idx]], [r_int[idx]])
            marker_img.set_data([pts_int[idx][0]], [pts_int[idx][1]])
            marker_ext.set_data([], [])
            text_info.set_text(f"INTERNO: Ang={ang_int[idx]:.2f}, Razão={r_int[idx]:.5f}")
            fig.canvas.draw_idle()

        elif event.inaxes == ax_img:
            mx, my = event.xdata, event.ydata
            if mx is None or my is None:
                return

            d_ext = float("inf")
            d_int = float("inf")

            if len(pts_ext) > 0:
                d_ext = np.min(np.sqrt((pts_ext[:, 0] - mx) ** 2 + (pts_ext[:, 1] - my) ** 2))
            if len(pts_int) > 0:
                d_int = np.min(np.sqrt((pts_int[:, 0] - mx) ** 2 + (pts_int[:, 1] - my) ** 2))

            if d_ext < d_int and d_ext < 50 and len(pts_ext) > 0:
                idx = np.argmin(np.sqrt((pts_ext[:, 0] - mx) ** 2 + (pts_ext[:, 1] - my) ** 2))
                marker_img.set_data([pts_ext[idx][0]], [pts_ext[idx][1]])
                marker_ext.set_data([ang_ext[idx]], [r_ext[idx]])
                marker_int.set_data([], [])
                text_info.set_text(f"IMAGEM (Ext): Ang={ang_ext[idx]:.2f}")
                fig.canvas.draw_idle()

            elif d_int < 50 and len(pts_int) > 0:
                idx = np.argmin(np.sqrt((pts_int[:, 0] - mx) ** 2 + (pts_int[:, 1] - my) ** 2))
                marker_img.set_data([pts_int[idx][0]], [pts_int[idx][1]])
                marker_int.set_data([ang_int[idx]], [r_int[idx]])
                marker_ext.set_data([], [])
                text_info.set_text(f"IMAGEM (Int): Ang={ang_int[idx]:.2f}")
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    plt.tight_layout()
    plt.show()