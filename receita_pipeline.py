# receita_pipeline.py
import cv2
import numpy as np
import json
from scipy.ndimage import gaussian_filter
from utils_scaling import ImageScaler

RECEITA_PADRAO = {
    "nome_receita": "Pipeline de Segmentacao e Limpeza v1",
    "pipeline_filtros": [
        {"nome": "Contraste", "parametros": {"kernel": 3, "fator_contraste": 10}},
        {"nome": "Filtro Gaussiano", "parametros": {"tamanho_kernel": 19, "constante_sigma": 5}},
        {"nome": "Binarizacao Sauvola", "parametros": {"tamanho_kernel": 41, "threshold": 0.30}},
        {"nome": "Conectar Falhas", "parametros": {"kernel": 15}},
        {
            "nome": "Filtrar por Elipse",
            "parametros": {
                "espessura_banda": 21,
                "dilatacao_extra": 5,
                "distancia_base": 18,
                "compatibilidade_minima": 0.25,
                "area_min_relativa": 0.15,
                "max_iter": 8,
                "amostrar_contorno_passo": 3,
                "usar_gap_dinamico": True,
                "usar_mad_dinamico": True,
                "manter_maior_componente": True,
            }
        }
    ]
}


class RecipeRunner:
    def __init__(self, scaler: ImageScaler):
        self.scaler = scaler
        self.receita_atual = json.loads(json.dumps(RECEITA_PADRAO))
        self.img_base_xor = None

    def carregar_receita_json(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            self.receita_atual = json.load(f)
        return self.receita_atual.get("nome_receita", "Receita Carregada")

    def definir_receita(self, receita_dict):
        self.receita_atual = json.loads(json.dumps(receita_dict))
        return self.receita_atual.get("nome_receita", "Receita Definida")

    def _ensure_uint8(self, img):
        if img is None:
            return None
        if img.dtype == np.uint8:
            return img
        if np.max(img) <= 1.0:
            img = img * 255.0
        return np.clip(img, 0, 255).astype(np.uint8)

    def _ensure_gray(self, img):
        if img is None:
            return None
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img.copy()

    def _ensure_binary(self, img, threshold=127):
        img = self._ensure_uint8(self._ensure_gray(img))
        if img is None:
            return None
        _, bin_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        return bin_img

    def _kernel_odd(self, valor, min_val=3, escalar=True):
        if escalar:
            return self.scaler.scale_kernel(int(valor), min_val=min_val, make_odd=True)
        v = max(int(valor), min_val)
        if v % 2 == 0:
            v += 1
        return v

    def _kernel_any(self, valor, min_val=1, escalar=True):
        if escalar:
            return max(int(round(float(valor) * self.scaler.scale_factor)), min_val)
        return max(int(valor), min_val)

    def _scale_area_safe(self, area_val):
        if hasattr(self.scaler, "scale_area"):
            return max(1, int(round(self.scaler.scale_area(float(area_val)))))
        sf = getattr(self.scaler, "scale_factor", 1.0)
        return max(1, int(round(float(area_val) * (sf ** 2))))

    def _phansalkar_threshold(self, image, window_size, k_val, p, q, R):
        img_float = image.astype(np.float32)
        mean = cv2.blur(img_float, (window_size, window_size))
        mean_square = cv2.blur(img_float ** 2, (window_size, window_size))
        variance = mean_square - mean ** 2
        variance[variance < 0] = 0
        stddev = np.sqrt(variance)
        threshold = mean * (1 + p * np.exp(-q * mean)) - k_val * (stddev / max(R, 1e-6))
        return (img_float > threshold).astype(np.uint8) * 255

    def homomorphic_filter(self, image, sigma=30):
        rows, cols = image.shape
        image_log = np.log1p(np.array(image, dtype="float32") / 255.0)
        dft = np.fft.fft2(image_log)
        dft_shift = np.fft.fftshift(dft)

        y, x = np.ogrid[:rows, :cols]
        center = (rows // 2, cols // 2)
        mask = np.exp(-((x - center[1]) ** 2 + (y - center[0]) ** 2) / (2 * (sigma ** 2)))
        mask = 1.0 - mask

        filtered_dft = dft_shift * mask
        filtered_dft_shift = np.fft.ifftshift(filtered_dft)
        img_back = np.fft.ifft2(filtered_dft_shift)
        img_back = np.abs(img_back)

        den = (np.max(img_back) - np.min(img_back))
        if den <= 1e-8:
            return np.zeros_like(image, dtype=np.uint8)

        homomorphic_img = np.expm1(img_back)
        den2 = (np.max(homomorphic_img) - np.min(homomorphic_img))
        if den2 <= 1e-8:
            return np.zeros_like(image, dtype=np.uint8)

        homomorphic_img = (homomorphic_img - np.min(homomorphic_img)) / den2 * 255.0
        return homomorphic_img.astype(np.uint8)

    def single_scale_retinex(self, image, sigma=30):
        img_float = image.astype(np.float32) + 1.0
        retinex = np.log1p(img_float) - np.log1p(gaussian_filter(img_float, max(float(sigma), 0.1)))

        den = (np.max(retinex) - np.min(retinex))
        if den <= 1e-8:
            return np.zeros_like(image, dtype=np.uint8)

        retinex = (retinex - np.min(retinex)) / den * 255.0
        return retinex.astype(np.uint8)

    def anisotropic_diffusion(self, image, niter=10, kappa=50, gamma=0.1, option=1):
        img = image.astype(np.float32)
        for _ in range(max(int(niter), 1)):
            deltaN = np.roll(img, -1, axis=0)
            deltaS = np.roll(img, 1, axis=0)
            deltaE = np.roll(img, -1, axis=1)
            deltaW = np.roll(img, 1, axis=1)

            if option == 1:
                cN = np.exp(-((deltaN - img) ** 2) / max(kappa ** 2, 1e-6))
                cS = np.exp(-((deltaS - img) ** 2) / max(kappa ** 2, 1e-6))
                cE = np.exp(-((deltaE - img) ** 2) / max(kappa ** 2, 1e-6))
                cW = np.exp(-((deltaW - img) ** 2) / max(kappa ** 2, 1e-6))
            else:
                cN = 1.0 / (1.0 + ((deltaN - img) / max(kappa, 1e-6)) ** 2)
                cS = 1.0 / (1.0 + ((deltaS - img) / max(kappa, 1e-6)) ** 2)
                cE = 1.0 / (1.0 + ((deltaE - img) / max(kappa, 1e-6)) ** 2)
                cW = 1.0 / (1.0 + ((deltaW - img) / max(kappa, 1e-6)) ** 2)

            img += gamma * (
                cN * (deltaN - img) +
                cS * (deltaS - img) +
                cE * (deltaE - img) +
                cW * (deltaW - img)
            )
        return np.clip(img, 0, 255).astype(np.uint8)

    # ============================================================
    # FILTRO POR ELIPSE - VERSÃO ITERATIVA POR DISTÂNCIA
    # ============================================================

    def _connected_components_with_stats(self, img_bin):
        return cv2.connectedComponentsWithStats(img_bin, connectivity=8)

    def _extract_components(self, img_bin, sample_step=3):
        num_labels, labels, stats, centroids = self._connected_components_with_stats(img_bin)
        comps = []

        for lbl in range(1, num_labels):
            area = int(stats[lbl, cv2.CC_STAT_AREA])
            if area <= 0:
                continue

            comp_mask = np.zeros_like(img_bin)
            comp_mask[labels == lbl] = 255

            contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                continue

            contour = max(contours, key=cv2.contourArea)
            pts = contour[:, 0, :].astype(np.float32)

            step = max(1, int(sample_step))
            pts_sampled = pts[::step] if len(pts) > step else pts

            x = int(stats[lbl, cv2.CC_STAT_LEFT])
            y = int(stats[lbl, cv2.CC_STAT_TOP])
            w = int(stats[lbl, cv2.CC_STAT_WIDTH])
            h = int(stats[lbl, cv2.CC_STAT_HEIGHT])

            comps.append({
                "label": lbl,
                "area": area,
                "mask": comp_mask,
                "contour": contour,
                "pts": pts,
                "pts_sampled": pts_sampled,
                "centroid": np.array(centroids[lbl], dtype=np.float32),
                "bbox": (x, y, w, h),
            })

        comps.sort(key=lambda c: c["area"], reverse=True)
        return comps, labels, stats

    def _bbox_gap(self, bbox_a, bbox_b):
        ax, ay, aw, ah = bbox_a
        bx, by, bw, bh = bbox_b

        ax2 = ax + aw - 1
        ay2 = ay + ah - 1
        bx2 = bx + bw - 1
        by2 = by + bh - 1

        dx = 0
        if ax2 < bx:
            dx = bx - ax2
        elif bx2 < ax:
            dx = ax - bx2

        dy = 0
        if ay2 < by:
            dy = by - ay2
        elif by2 < ay:
            dy = ay - by2

        return float(np.hypot(dx, dy))

    def _min_distance_between_point_sets(self, pts_a, pts_b):
        if pts_a is None or pts_b is None or len(pts_a) == 0 or len(pts_b) == 0:
            return float("inf")

        a = pts_a.astype(np.float32)
        b = pts_b.astype(np.float32)

        diff = a[:, None, :] - b[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        return float(np.sqrt(np.min(d2)))

    def _component_distance(self, comp_a, comp_b, bbox_gate=80):
        gap = self._bbox_gap(comp_a["bbox"], comp_b["bbox"])
        bbox_gate_scaled = self._kernel_any(bbox_gate, min_val=10, escalar=True)

        if gap > bbox_gate_scaled:
            return gap

        return self._min_distance_between_point_sets(comp_a["pts_sampled"], comp_b["pts_sampled"])

    def _distance_to_accepted_set(self, candidate_idx, accepted_indices, comps, dist_cache):
        best = float("inf")
        for acc_idx in accepted_indices:
            key = tuple(sorted((candidate_idx, acc_idx)))
            if key not in dist_cache:
                dist_cache[key] = self._component_distance(comps[candidate_idx], comps[acc_idx])
            best = min(best, dist_cache[key])
        return best

    def _build_mask_from_indices(self, shape, comps, indices):
        out = np.zeros(shape, dtype=np.uint8)
        for idx in indices:
            out[comps[idx]["mask"] > 0] = 255
        return out

    def _fit_ellipse_from_mask(self, mask):
        pts = cv2.findNonZero(mask)
        if pts is None or len(pts) < 5:
            return None
        try:
            return cv2.fitEllipse(pts)
        except cv2.error:
            return None

    def _build_ellipse_masks(self, shape, ellipse, espessura_banda=21, dilatacao_extra=0):
        band = np.zeros(shape, dtype=np.uint8)
        filled = np.zeros(shape, dtype=np.uint8)

        if ellipse is None:
            return band, filled

        thickness = self._kernel_any(espessura_banda, min_val=1, escalar=True)
        cv2.ellipse(band, ellipse, 255, thickness=thickness)
        cv2.ellipse(filled, ellipse, 255, -1)

        dil_extra = int(dilatacao_extra)
        if dil_extra > 0:
            k = self._kernel_odd(dil_extra, min_val=3, escalar=True)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            band = cv2.dilate(band, kernel, iterations=1)
            filled = cv2.dilate(filled, kernel, iterations=1)

        return np.where(band > 0, 255, 0).astype(np.uint8), np.where(filled > 0, 255, 0).astype(np.uint8)

    def _component_ellipse_compatibility(self, comp, ellipse_filled_mask, ellipse_band_mask):
        area = max(1, int(comp["area"]))
        comp_mask = comp["mask"] > 0

        inside_ratio = float(np.count_nonzero((ellipse_filled_mask > 0) & comp_mask)) / area
        band_ratio = float(np.count_nonzero((ellipse_band_mask > 0) & comp_mask)) / area

        score = (0.65 * band_ratio) + (0.35 * inside_ratio)
        return score, inside_ratio, band_ratio

    def _dynamic_distance_threshold(
        self,
        distances,
        base_distance=18,
        use_gap=True,
        use_mad=True
    ):
        base_scaled = float(self._kernel_any(base_distance, min_val=3, escalar=True))

        valid = [float(d) for d in distances if np.isfinite(d)]
        if len(valid) == 0:
            return base_scaled

        valid_sorted = np.sort(np.array(valid, dtype=np.float32))

        thr_candidates = [base_scaled]

        if use_mad and len(valid_sorted) >= 3:
            med = float(np.median(valid_sorted))
            mad = float(np.median(np.abs(valid_sorted - med)))
            thr_mad = med + (2.5 * max(mad, 1.0))
            thr_candidates.append(thr_mad)

        if use_gap and len(valid_sorted) >= 4:
            diffs = np.diff(valid_sorted)
            idx_gap = int(np.argmax(diffs))
            gap_val = float(diffs[idx_gap])

            if gap_val >= max(2.0, 0.20 * max(valid_sorted[idx_gap], 1.0)):
                thr_gap = float(valid_sorted[idx_gap])
                thr_candidates.append(thr_gap)

        thr = max(base_scaled, min(thr_candidates))
        return float(thr)

    def _iterative_component_growth_for_ellipse(
        self,
        img_bin,
        comps,
        espessura_banda=21,
        dilatacao_extra=5,
        distancia_base=10,
        compatibilidade_minima=0.20,
        area_min_relativa=0.12,
        max_iter=16,
        usar_gap_dinamico=True,
        usar_mad_dinamico=True,
        manter_maior_componente=True
    ):
        if not comps:
            return set(), None, None, None

        main_area = max(1, int(comps[0]["area"]))
        area_min_abs = max(1, int(round(main_area * float(area_min_relativa))))

        accepted = {0}  
        dist_cache = {}

        accepted_mask = self._build_mask_from_indices(img_bin.shape, comps, accepted)
        ellipse = self._fit_ellipse_from_mask(accepted_mask)

        for _ in range(max(1, int(max_iter))):
            remaining = [i for i in range(len(comps)) if i not in accepted]
            if not remaining:
                break

            if ellipse is None:
                accepted_mask = self._build_mask_from_indices(img_bin.shape, comps, accepted)
                ellipse = self._fit_ellipse_from_mask(accepted_mask)
                if ellipse is None:
                    break

            ellipse_band, ellipse_filled = self._build_ellipse_masks(
                img_bin.shape,
                ellipse,
                espessura_banda=espessura_banda,
                dilatacao_extra=dilatacao_extra
            )

            distances = []
            candidate_info = []

            for idx in remaining:
                comp = comps[idx]

                if comp["area"] < area_min_abs:
                    continue

                dist_to_set = self._distance_to_accepted_set(idx, accepted, comps, dist_cache)
                comp_score, inside_ratio, band_ratio = self._component_ellipse_compatibility(
                    comp, ellipse_filled, ellipse_band
                )

                distances.append(dist_to_set)
                candidate_info.append({
                    "idx": idx,
                    "dist": dist_to_set,
                    "compat": comp_score,
                    "inside_ratio": inside_ratio,
                    "band_ratio": band_ratio,
                    "area": comp["area"],
                })

            if not candidate_info:
                break

            dyn_thr = self._dynamic_distance_threshold(
                distances,
                base_distance=distancia_base,
                use_gap=bool(usar_gap_dinamico),
                use_mad=bool(usar_mad_dinamico)
            )

            new_accepts = set()

            for info in candidate_info:
                idx = info["idx"]
                dist_ok = info["dist"] <= dyn_thr
                compat_ok = info["compat"] >= float(compatibilidade_minima)

                very_close_thr = max(1.0, 0.55 * dyn_thr)
                very_close = info["dist"] <= very_close_thr

                if (dist_ok and compat_ok) or (very_close and info["inside_ratio"] > 0.05):
                    new_accepts.add(idx)

            if not new_accepts:
                break

            accepted |= new_accepts

            accepted_mask = self._build_mask_from_indices(img_bin.shape, comps, accepted)
            ellipse = self._fit_ellipse_from_mask(accepted_mask)

        if manter_maior_componente and len(comps) > 0:
            accepted.add(0)

        final_mask = self._build_mask_from_indices(img_bin.shape, comps, accepted)
        final_ellipse = self._fit_ellipse_from_mask(final_mask)

        if final_ellipse is None and len(comps) > 0:
            final_ellipse = self._fit_ellipse_from_mask(comps[0]["mask"])

        final_band, final_filled = self._build_ellipse_masks(
            img_bin.shape,
            final_ellipse,
            espessura_banda=espessura_banda,
            dilatacao_extra=dilatacao_extra
        )

        return accepted, final_ellipse, final_band, final_filled

    def _filter_components_by_ellipse(
        self,
        img,
        espessura_banda=21,
        dilatacao_extra=5,
        distancia_base=18,
        compatibilidade_minima=0.20,
        area_min_relativa=0.01,
        max_iter=8,
        amostrar_contorno_passo=3,
        usar_gap_dinamico=True,
        usar_mad_dinamico=True,
        manter_maior_componente=True
    ):
        img_bin = self._ensure_binary(img)
        if img_bin is None:
            return img

        comps, labels, stats = self._extract_components(
            img_bin,
            sample_step=max(1, int(amostrar_contorno_passo))
        )

        if not comps:
            return img_bin

        accepted, final_ellipse, final_band, final_filled = self._iterative_component_growth_for_ellipse(
            img_bin=img_bin,
            comps=comps,
            espessura_banda=espessura_banda,
            dilatacao_extra=dilatacao_extra,
            distancia_base=distancia_base,
            compatibilidade_minima=compatibilidade_minima,
            area_min_relativa=area_min_relativa,
            max_iter=max_iter,
            usar_gap_dinamico=usar_gap_dinamico,
            usar_mad_dinamico=usar_mad_dinamico,
            manter_maior_componente=manter_maior_componente
        )

        out = np.zeros_like(img_bin)
        for idx in accepted:
            out[comps[idx]["mask"] > 0] = 255

        return np.where(out > 0, 255, 0).astype(np.uint8)

    # ============================================================
    # EXECUÇÃO DOS FILTROS
    # ============================================================

    def aplicar_filtro_individual(self, img, nome, params=None, original=None):
        if img is None:
            return None

        params = params or {}
        nome_norm = (nome or "").strip().lower()
        img = self._ensure_gray(img)
        img = self._ensure_uint8(img)

        if original is None:
            original = img.copy()
        else:
            original = self._ensure_uint8(self._ensure_gray(original))

        if nome_norm in ("gaussiano", "filtro gaussiano"):
            k = self._kernel_odd(params.get("tamanho_kernel", params.get("kernel", 3)), min_val=3, escalar=True)
            sigma = float(params.get("constante_sigma", 0))
            img = cv2.GaussianBlur(img, (k, k), sigma)

        elif nome_norm in ("medianblur", "median", "filtro mediana", "median blur"):
            k = self._kernel_odd(params.get("kernel", params.get("tamanho_kernel", 3)), min_val=3, escalar=True)
            img = cv2.medianBlur(img, k)

        elif nome_norm == "filtro bilateral":
            d = self._kernel_odd(params.get("kernel", params.get("tamanho_kernel", 5)), min_val=3, escalar=True)
            sigma_color = float(params.get("sigma_color", 10))
            sigma_space = float(params.get("sigma_space", 10))
            img = cv2.bilateralFilter(img, d, sigmaColor=sigma_color, sigmaSpace=sigma_space)

        elif nome_norm in ("non-local means", "non local means"):
            h_val = float(params.get("forca_filtro", 10.0))
            k_temp = self._kernel_odd(params.get("template_window", 7), min_val=5, escalar=True)
            k_search = self._kernel_odd(params.get("search_window", 21), min_val=15, escalar=True)
            img = cv2.fastNlMeansDenoising(img, None, h=h_val, templateWindowSize=k_temp, searchWindowSize=k_search)

        elif nome_norm == "segmentacao ia (cellpose)":
            try:
                from cellpose import models
                diam_raw = params.get("diameter", 30)
                diam_eff = self.scaler.scale_scalar(diam_raw)
                model = models.CellposeModel(gpu=True)
                resultado_eval = model.eval(img, diameter=diam_eff)
                masks = resultado_eval[0]
                img = (masks > 0).astype(np.uint8) * 255
            except ImportError:
                print("[AVISO] IA Cellpose não instalada no ambiente da pipeline.")

        elif nome_norm == "abertura":
            k = self._kernel_odd(params.get("kernel", params.get("tamanho_kernel", 3)), min_val=3, escalar=True)
            it = max(int(params.get("iters", params.get("iteracoes", 1))), 1)
            kernel = np.ones((k, k), np.uint8)
            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=it)

        elif nome_norm == "fechamento":
            k = self._kernel_odd(params.get("kernel", params.get("tamanho_kernel", 3)), min_val=3, escalar=True)
            it = max(int(params.get("iters", params.get("iteracoes", 1))), 1)
            kernel = np.ones((k, k), np.uint8)
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=it)

        elif nome_norm == "dilatacao":
            k = self._kernel_odd(params.get("tamanho_kernel", params.get("kernel", 3)), min_val=3, escalar=True)
            it = max(int(params.get("iteracoes", params.get("iters", 1))), 1)
            kernel = np.ones((k, k), np.uint8)
            img = cv2.dilate(img, kernel, iterations=it)

        elif nome_norm == "erosao":
            k = self._kernel_odd(params.get("tamanho_kernel", params.get("kernel", 3)), min_val=3, escalar=True)
            it = max(int(params.get("iteracoes", params.get("iters", 1))), 1)
            kernel = np.ones((k, k), np.uint8)
            img = cv2.erode(img, kernel, iterations=it)

        elif nome_norm == "top-hat":
            k = self._kernel_odd(params.get("kernel", params.get("tamanho_kernel", 15)), min_val=3, escalar=True)
            selem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            img = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, selem)

        elif nome_norm == "background subtraction":
            k = self._kernel_odd(params.get("kernel_size", params.get("kernel", 25)), min_val=3, escalar=True)
            background = cv2.medianBlur(img, k)
            img = cv2.subtract(img, background)

        elif nome_norm == "histogram equalization":
            img = cv2.equalizeHist(img)

        elif nome_norm == "contraste":
            k = self._kernel_odd(params.get("kernel", params.get("tamanho_kernel", 3)), min_val=3, escalar=True)
            fator = float(params.get("fator_contraste", 10)) / 10.0
            blur = cv2.GaussianBlur(img, (k, k), 0)
            img = cv2.addWeighted(img, 1.0 + fator, blur, -fator, 0)

        elif nome_norm in ("realce contraste", "realce contraste (clahe)"):
            clip_limit = float(params.get("fator_contraste", params.get("clip_limit", 2.0)))
            k_grid = max(1, int(params.get("kernel", params.get("tamanho_kernel", 8))))
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(k_grid, k_grid))
            img = clahe.apply(img)

        elif nome_norm == "binarizacao normal":
            thr = float(params.get("threshold_val", params.get("threshold", params.get("limiar", 1.0))))
            if thr <= 2.0:
                thr *= 127.5
            _, img = cv2.threshold(img, thr, 255, cv2.THRESH_BINARY)

        elif nome_norm == "binarizacao adaptativa (media)":
            block = self._kernel_odd(params.get("kernel", params.get("tamanho_kernel", 11)), min_val=3, escalar=True)
            c = float(params.get("constante_sigma", 2))
            img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block, c)

        elif nome_norm == "binarizacao adaptativa (gaussiana)":
            block = self._kernel_odd(params.get("kernel", params.get("tamanho_kernel", 11)), min_val=3, escalar=True)
            c = float(params.get("constante_sigma", 2))
            img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, c)

        elif nome_norm == "binarizacao sauvola":
            k = self._kernel_odd(params.get("tamanho_kernel", params.get("kernel", 15)), min_val=3, escalar=True)
            offset = float(params.get("threshold", 0.3))
            img = cv2.ximgproc.niBlackThreshold(
                img, 255, cv2.THRESH_BINARY, k, offset, cv2.ximgproc.BINARIZATION_SAUVOLA
            )

        elif nome_norm == "binarizacao phansalkar":
            block = self._kernel_odd(params.get("tamanho_kernel", params.get("kernel", 25)), min_val=3, escalar=True)
            img = self._phansalkar_threshold(
                img,
                window_size=block,
                k_val=float(params.get("k", 0.25)),
                p=float(params.get("p", 2.0)),
                q=float(params.get("q", 10.0)),
                R=float(params.get("R", 0.5))
            )

        elif nome_norm == "conectar falhas":
            k = self._kernel_odd(params.get("kernel", params.get("tamanho_kernel", 3)), min_val=3, escalar=True)
            kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_morph)

        elif nome_norm == "filtro de detalhes":
            sigma_s = float(params.get("sigma_space", 10))
            sigma_r = float(params.get("sigma_range", 0.15))
            img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            aux = cv2.detailEnhance(img_color, sigma_s=sigma_s, sigma_r=sigma_r)
            img = cv2.cvtColor(aux, cv2.COLOR_RGB2GRAY)

        elif nome_norm == "filtro de preservacao":
            sigma_s = float(params.get("sigma_space", 10))
            sigma_r = float(params.get("sigma_range", 0.15))
            img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            filtered = cv2.edgePreservingFilter(img_color, flags=1, sigma_s=sigma_s, sigma_r=sigma_r)
            img = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)

        elif nome_norm == "anisotropic diffusion":
            kappa = int(params.get("kappa", params.get("kernel", 50)))
            it = max(int(params.get("iters", params.get("iteracoes", 1))), 1)
            img = self.anisotropic_diffusion(img, niter=it, kappa=kappa)

        elif nome_norm == "homomorphic filter":
            sigma = float(params.get("sigma", params.get("sigma_space", 30)))
            img = self.homomorphic_filter(img, sigma=sigma)

        elif nome_norm == "single scale retinex":
            sigma = float(params.get("sigma", params.get("sigma_space", 30)))
            img = self.single_scale_retinex(img, sigma=sigma)

        elif nome_norm in ("remocao objetos", "remover componentes pequenos"):
            area_min_raw = float(params.get("qnt_pixels_minimo", params.get("area_min", 0)))
            area_max_raw = float(params.get("qnt_pixels_maximo", params.get("area_max", 10**12)))

            area_min = self._scale_area_safe(area_min_raw)
            area_max = self._scale_area_safe(area_max_raw)

            img = self._ensure_binary(img)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= area_min and area <= area_max:
                    img[labels == i] = 0

        elif nome_norm == "filtrar por elipse":
            img = self._filter_components_by_ellipse(
                img,
                espessura_banda=int(params.get("espessura_banda", 21)),
                dilatacao_extra=int(params.get("dilatacao_extra", 5)),
                distancia_base=float(params.get("distancia_base", 18)),
                compatibilidade_minima=float(params.get("compatibilidade_minima", 0.20)),
                area_min_relativa=float(params.get("area_min_relativa", 0.01)),
                max_iter=int(params.get("max_iter", 8)),
                amostrar_contorno_passo=int(params.get("amostrar_contorno_passo", 3)),
                usar_gap_dinamico=bool(params.get("usar_gap_dinamico", True)),
                usar_mad_dinamico=bool(params.get("usar_mad_dinamico", True)),
                manter_maior_componente=bool(params.get("manter_maior_componente", True)),
            )

        elif nome_norm == "somar imagem":
            img = cv2.add(img, img)

        elif nome_norm == "bitwise not":
            img = cv2.bitwise_not(img)

        elif nome_norm == "bitwise or":
            img = cv2.bitwise_or(img, original)

        elif nome_norm == "bitwise and":
            img = cv2.bitwise_and(img, original)

        elif nome_norm == "bitwise xor":
            if img.shape == original.shape:
                img = cv2.bitwise_xor(img, original)

        return self._ensure_uint8(img)

    def executar(self, img_in):
        img = self._ensure_gray(img_in)
        self.scaler.update_from_image(img)

        self.img_base_xor = img.copy()
        original = img.copy()

        filtros = self.receita_atual.get("pipeline_filtros", [])
        for etapa in filtros:
            if not isinstance(etapa, dict) or not etapa.get("nome"):
                continue

            nome = etapa.get("nome", "")
            params = etapa.get("parametros", {}) or {}
            img = self.aplicar_filtro_individual(img, nome, params, original=original)

        return self._ensure_uint8(img)