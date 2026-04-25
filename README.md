# Segmentação Geométrica de Núcleos em Imagens de Drosophila

Este repositório contém uma implementação em Python de um pipeline de visão computacional para segmentação de núcleos em imagens microscópicas de embriões de *Drosophila melanogaster*. O método combina pré-processamento parametrizável, filtragem por elipse, análise geométrica de contornos, detecção de concavidades e aplicação de cortes para separação de instâncias nucleares em regiões de alta densidade celular.

O projeto foi desenvolvido no contexto de um Trabalho de Conclusão de Curso, com foco em uma abordagem interpretável, reprodutível e de baixo custo computacional para segmentação de massa nuclear e individualização de núcleos em imagens biológicas.

---

## Descrição curta do repositório

Pipeline em Python para segmentação geométrica de núcleos em imagens microscópicas de embriões de *Drosophila melanogaster*, com pré-processamento por receitas JSON, filtragem por elipse, detecção de concavidades, cortes geométricos, interface gráfica e avaliação quantitativa contra anotações manuais.

---

## Visão geral do pipeline

O fluxo principal do sistema é composto pelas seguintes etapas:

1. Carregamento da imagem microscópica.
2. Conversão para escala de cinza.
3. Pré-processamento por uma receita JSON.
4. Binarização e limpeza da máscara.
5. Filtragem por elipse para remoção de ruídos e preservação da massa principal.
6. Extração de contornos externos e internos.
7. Modelagem do contorno em representação polar.
8. Detecção de concavidades.
9. Proposição de cortes geométricos por pareamento vetorial.
10. Aplicação de mecanismo de fallback quando não há pareamento interno confiável.
11. Reconstrução das instâncias segmentadas.
12. Geração de imagens de debug, máscaras, contornos e métricas.

---

## Estrutura dos arquivos

### `gera_receita.py`

Interface gráfica principal para criação e ajuste de receitas de pré-processamento.

Permite carregar uma imagem, aplicar filtros sequenciais, ajustar parâmetros visualmente, segmentar a imagem, visualizar resultados intermediários e salvar uma receita em JSON. É o módulo mais indicado para calibração manual do pipeline.

Uso típico:

```bash
python gera_receita.py
```

Principais funções:

- carregar imagens microscópicas;
- aplicar filtros de pré-processamento;
- ajustar parâmetros de segmentação;
- visualizar máscaras e contornos;
- gerar receitas JSON reutilizáveis;
- salvar resultados e configurações.

---

### `aplicador.py`

Interface gráfica para aplicar uma receita JSON em uma imagem única ou em uma pasta de imagens.

Este módulo é voltado para uso operacional após a definição de uma receita. Ele permite processar imagens em lote e salvar máscaras, contornos e arquivos JSON de resultados.

Uso típico:

```bash
python aplicador.py
```

Principais funções:

- carregar uma receita JSON;
- aplicar o pipeline em uma imagem individual;
- processar uma pasta inteira recursivamente;
- salvar máscaras segmentadas;
- salvar contornos sobre a imagem original;
- exportar resultados em JSON.

---

### `gera_resultados.py`

Módulo de avaliação quantitativa do método.

Executa a receita oficial sobre um conjunto de dados anotado manualmente e compara as predições do algoritmo com máscaras de referência geradas a partir de arquivos JSON do AnyLabeling.

Uso típico:

```bash
python gera_resultados.py
```

Por padrão, o script espera:

```text
Dataset_TCC/
receita_oficial.json
```

Saídas principais:

```text
output_process_final/
```

Dentro dessa pasta são gerados:

- imagem original;
- imagem pré-processada;
- máscara de ground truth;
- máscara predita pelo algoritmo;
- imagens de debug;
- contornos da referência;
- contornos da predição;
- imagem de inspeção por overlay;
- relatório CSV com métricas.

O relatório final é salvo em:

```text
output_process_final/relatorio_metricas.csv
```

Métricas calculadas:

- massa de referência em pixels;
- massa predita em pixels;
- quantidade de objetos anotados;
- quantidade de objetos preditos;
- quantidade de cortes geométricos;
- verdadeiros positivos;
- falsos positivos;
- falsos negativos;
- F1-Score por instância;
- Dice/F1 por massa segmentada.

---

### `receita_pipeline.py`

Módulo responsável por executar receitas de pré-processamento.

A classe principal é:

```python
RecipeRunner
```

Ela recebe uma imagem de entrada e aplica sequencialmente os filtros definidos em uma receita JSON.

Filtros suportados incluem, entre outros:

- Filtro Gaussiano;
- MedianBlur;
- Filtro Bilateral;
- Non-Local Means;
- Binarização Normal;
- Binarização Adaptativa;
- Binarização Sauvola;
- Binarização Phansalkar;
- Dilatação;
- Erosão;
- Abertura;
- Fechamento;
- Top-Hat;
- Remoção de objetos;
- Bitwise NOT;
- Bitwise AND;
- Bitwise OR;
- Bitwise XOR;
- Filtro por Elipse.

---

### `segmentacao_aneis.py`

Módulo central da segmentação geométrica.

Contém a classe:

```python
RingSegmenterForced
```

Responsável por:

- receber a máscara pré-processada;
- chamar a análise de concavidades;
- parear concavidades externas e internas;
- calcular candidatos a cortes;
- evitar cruzamentos entre cortes;
- aplicar cortes aceitos;
- executar fallback quando necessário;
- reconstruir os objetos finais;
- gerar a lista de núcleos segmentados em JSON.

Função pública principal:

```python
aplicar_segmentacao_refinada(...)
```

Essa função é usada pelos módulos `gera_receita.py`, `aplicador.py` e `gera_resultados.py`.

---

### `analisa_concavidades_anel.py`

Módulo de análise geométrica de contornos.

Contém a classe:

```python
RingConcavityAnalyzer
```

Responsável por:

- ajustar elipse à máscara binária;
- construir uma topologia refinada;
- extrair contorno externo e interno;
- ordenar pontos do contorno angularmente;
- representar o contorno em função de `r(θ)`;
- suavizar o sinal radial;
- aplicar realce por segunda derivada;
- detectar picos associados a concavidades;
- gerar vetores normais para orientar cortes;
- desenhar imagens de debug.

Também fornece funções auxiliares para análise visual:

```python
pipeline_anel(...)
analisar_dados_anel(...)
exibir_grafico_interativo(...)
```

---

### `utils_scaling.py`

Módulo de controle dinâmico de escala.

Contém a classe:

```python
ImageScaler
```

Esse componente ajusta automaticamente parâmetros espaciais de acordo com a resolução da imagem de entrada. Isso evita que valores como tamanho de kernel, espessura de linha, distância máxima e área mínima tenham efeitos diferentes em imagens com resoluções distintas.

Principais métodos:

```python
update_from_image(...)
scale_scalar(...)
scale_kernel(...)
scale_area(...)
get_info(...)
```

---

### `pipeline_fft.py`

Implementa funções auxiliares baseadas em DCT/FFT, incluindo suavização e cálculo de derivadas no domínio da frequência.

Inclui:

- `smoothn_literal`;
- `fft_deriv`;
- `curvature_fft_literal`;
- `pipeline_matlab_fft_full`.

Este arquivo serve como apoio experimental para aproximações inspiradas em rotinas MATLAB.

---

### `segmentacao_fft_matlab.py`

Implementação alternativa/experimental de segmentação baseada em curvatura via FFT, inspirada em lógica MATLAB.

Contém funções para:

- calcular derivadas por FFT;
- estimar curvatura do contorno;
- localizar pontos candidatos a corte;
- aplicar cortes com base em picos de curvatura.

Uso principal: comparação, estudo ou experimentação com uma abordagem baseada em curvatura.

---

### `receita_oficial.json`

Receita principal utilizada nos experimentos consolidados.

Define o pipeline de pré-processamento, os parâmetros de segmentação geométrica e a configuração da topologia refinada.

É usada por padrão em:

```bash
python gera_resultados.py
```

---

### `receita_oficial copy.json`

Versão alternativa ou anterior da receita oficial.

Pode ser usada para comparação histórica ou recuperação de parâmetros antigos.

---

### `.gitignore`

Arquivo de controle do Git para definir quais arquivos e pastas devem ser versionados.

A configuração atual prioriza o versionamento dos scripts principais, receitas e possíveis diretórios de dados/saída selecionados, ignorando caches, ambientes virtuais e arquivos temporários.

---

## Formato esperado das receitas JSON

Uma receita possui a seguinte estrutura geral:

```json
{
  "nome_receita": "Nome da Receita",
  "pipeline_filtros": [
    {
      "nome": "Filtro Gaussiano",
      "parametros": {
        "tamanho_kernel": 11,
        "constante_sigma": 0.0
      }
    }
  ],
  "parametros_analise_final": {
    "limiar_concavidade": 0.005,
    "max_dist": 80,
    "min_score": 0.9,
    "fator_detalhe": 14.0
  }
}
```

Cada etapa em `pipeline_filtros` é executada sequencialmente pelo `RecipeRunner`.

---

## Como usar o projeto

### 1. Instalar dependências

Recomenda-se utilizar um ambiente virtual:

```bash
python -m venv .venv
```

Ativação no Windows:

```bash
.venv\Scripts\activate
```

Ativação no Linux/macOS:

```bash
source .venv/bin/activate
```

Instalar dependências principais:

```bash
pip install numpy opencv-python opencv-contrib-python scipy matplotlib pillow
```

Observação: o projeto usa `cv2.ximgproc.niBlackThreshold`, portanto é recomendado instalar `opencv-contrib-python`, não apenas `opencv-python`.

---

### 2. Criar ou ajustar uma receita

Execute:

```bash
python gera_receita.py
```

Na interface:

1. carregue uma imagem;
2. aplique filtros;
3. ajuste parâmetros;
4. execute a segmentação;
5. visualize os resultados;
6. salve a receita.

---

### 3. Aplicar uma receita em uma imagem ou pasta

Execute:

```bash
python aplicador.py
```

Na interface:

1. carregue uma imagem ou selecione uma pasta;
2. carregue uma receita JSON;
3. aplique a receita;
4. salve os resultados.

---

### 4. Avaliar o método contra anotações manuais

Organize o dataset no formato:

```text
Dataset_TCC/
├── Mutantes/
│   ├── 2-1.tif
│   ├── 2-1.json
│   ├── 3-1.tif
│   └── 3-1.json
└── Selvagens/
    ├── 1-1.tif
    ├── 1-1.json
    └── ...
```

Cada imagem deve possuir um JSON do AnyLabeling com o mesmo nome base.

Depois execute:

```bash
python gera_resultados.py
```

Os resultados serão salvos em:

```text
output_process_final/
```

---

## Saídas geradas pelo avaliador

Para cada imagem processada, o sistema gera arquivos semelhantes a:

```text
<amostra>_1_original.jpg
<amostra>_2_pre_processada.jpg
<amostra>_3_ground_truth.jpg
<amostra>_3b_ground_truth_contorno.jpg
<amostra>_4_predicao_algoritmo.jpg
<amostra>_4_debug_segmentacao.jpg
<amostra>_4b_predicao_algoritmo_contorno.jpg
<amostra>_4c_debug_segmentacao_contorno.jpg
<amostra>_5_inspecao_overlay.jpg
```

Esses arquivos permitem inspecionar visualmente cada etapa do pipeline.

---

## Exemplo de execução completa

```bash
python gera_receita.py
```

Ajuste e salve uma receita.

Depois:

```bash
python gera_resultados.py
```

Ao final, consulte:

```text
output_process_final/relatorio_metricas.csv
```

---

## Principais métricas

### Dice/F1 de massa

Avalia a sobreposição entre a máscara predita e a máscara de referência em nível de pixel.

### F1-Score por instância

Avalia a correspondência entre objetos individuais preditos e objetos anotados manualmente.

### TP, FP e FN

- `TP`: núcleos corretamente detectados;
- `FP`: objetos preditos sem correspondência adequada na referência;
- `FN`: objetos da referência que não foram detectados pelo algoritmo.

---

## Observações importantes

- O desempenho do método depende fortemente da qualidade da máscara pré-processada.
- O filtro por elipse é usado para remover ruídos sem descartar indevidamente núcleos próximos da massa principal.
- O mecanismo de fallback ajuda em casos nos quais não há contorno interno confiável para pareamento.
- A calibração dos parâmetros pode melhorar significativamente a separação de instâncias em imagens mais complexas.
- O método é interpretável e permite inspeção visual das decisões geométricas.

---

## Licença

Defina aqui a licença do projeto, caso deseje disponibilizá-lo publicamente.

Exemplo:

```text
MIT License
```

---

## Autor

João Pedro de Oliveira Marques

Trabalho desenvolvido no contexto de segmentação geométrica de núcleos em imagens microscópicas de embriões de *Drosophila melanogaster*.
