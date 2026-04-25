#utils_scaling.py
import numpy as np

class ImageScaler:
    """
    Gerencia a escala uniforme para processamento de imagens.
    Garante que parâmetros (ex: tamanho de kernel 30) se comportem
    como se fossem aplicados em uma imagem de referência (ex: 1000px),
    independentemente do tamanho real da imagem.
    """
    def __init__(self, reference_dim=1000.0):
        self.ref_dim = float(reference_dim)
        self.scale_factor = 1.0
        self.current_shape = (0, 0)

    def update_from_image(self, image: np.ndarray):
        """Atualiza o fator de escala baseado na imagem atual."""
        if image is None:
            return
        h, w = image.shape[:2]
        self.current_shape = (h, w)
        max_dim = max(h, w)
        if max_dim <= 0:
            self.scale_factor = 1.0
        else:
            self.scale_factor = max_dim / self.ref_dim

    def scale_scalar(self, val: float) -> float:
        """Escala um valor escalar simples."""
        return val * self.scale_factor

    def scale_kernel(self, val: int, min_val=1, make_odd=True) -> int:
        """Escala tamanho de kernel/janela."""
        scaled = int(round(val * self.scale_factor))
        if scaled < min_val:
            scaled = min_val
        
        if make_odd and scaled % 2 == 0:
            scaled += 1
        return scaled

    def scale_area(self, area_base: float) -> float:
        """Escala área (quadrado do fator de escala)."""
        return area_base * (self.scale_factor ** 2)

    def get_info(self) -> str:
        return f"Ref: {self.ref_dim}px | Atual: {self.current_shape} | Fator: {self.scale_factor:.4f}"