> **Tarefa:** Classificação binária de radiografias torácicas (NORMAL vs. PNEUMONIA)  
> **Modelo:** EfficientNet-B0 com validação cruzada 5-fold  
> **Métrica principal:** AUC-ROC
> **Resultado:** 0.99975 ± 0.00012
---

## Como reproduzir no Google Colab

### Passo 1 - Abrir o notebook

Clique em **"Open in Colab"** ou faça upload manual do arquivo `.ipynb` no [Google Colab](https://colab.research.google.com/).

É melhor usar uma sessão com GPU

### Passo 2 - Organizar o dataset no Google Drive

**Baixe o dataset do Kaggle:** [ligia-compviz - Competition Data](https://www.kaggle.com/competitions/ligia-compviz/data)

Depois disso, extraia-o pro seu google drive da seguinte maneira:

```
/content/drive/MyDrive/
└── ligia-compviz/
    ├── train.csv
    ├── test.csv
    ├── train/
    │   └── train/
    │       ├── NORMAL/
    │       │   └── *.jpeg
    │       └── PNEUMONIA/
    │           └── *.jpeg
    └── test_images/
        └── test_images/
            └── *.jpeg
```

### Passo 3 - Instalar dependências

A primeira célula do notebook já instala o necessário automaticamente:

```python
!pip install -q timm albumentations==1.4.6 opencv-python-headless imagehash grad-cam
```
Mas, para instalar todas as dependências com versões exatas:

```bash
pip install -r requirements.txt
```

### Passo 4 - Executar o notebook

Execute as células em ordem, do início ao fim. O notebook está dividido nas seguintes seções:

| Seção | Descrição |
|---|---|
| 0. Setup | Instalação e imports |
| 1. Paths e CSVs | Leitura dos dados e montagem do Drive |
| 2. EDA | Análise exploratória e distribuição de classes |
| 3. Deduplicação | Remoção de duplicatas exatas (MD5) e visuais (pHash) |
| 4. StratifiedGroupKFold | Divisão dos folds com agrupamento por paciente |
| 5. Transformações | Augmentations de treino, validação e TTA |
| 6. Dataset CXR | Classe `CXRDataset` para carregamento das imagens |
| 7. Backbone | Definição do modelo EfficientNet-B0 |
| 8. Loop de treinamento | Funções de treino e validação por época |
| 9. Treinamento (5-fold) | Treinamento completo com salvamento dos `.pth` |
| 10–11. Resumo e curvas | Tabela de AUC por fold e curvas de loss/AUC |
| 12. Análise OOF | Curva ROC e Matriz de Confusão agregadas |
| 13. Análise de Erros | FP, FN e casos de baixa confiança |
| 14. Calibração | Brier Score e Reliability Diagram |
| 15. Grad-CAM | Interpretabilidade visual das predições |
| 16. Inferência com TTA | Predição no conjunto de teste com TTA |
| 17. Submissão | Geração do `submission.csv` |

### Passo 5 - Usando os checkpoints pré-treinados

Se você quiser pular o treinamento e usar diretamente os pesos já treinados:

1. Faça o upload da pasta `models/` para o seu Google Drive em `MyDrive/ligia-compviz/models/`.
2. Atualize o caminho `MODELS_DIR` na célula de configuração:
   ```python
   MODELS_DIR = Path('/content/drive/MyDrive/ligia-compviz/models')
   ```
3. Execute apenas as seções **0, 1, 5, 6, 7** e depois pule diretamente para a **seção 16 (Inferência com TTA)**.

---

## Resultados

| Fold | AUC |
|---|---|
| 0 | 0.99975 |
| 1 | 0.99984 |
| 2 | 0.99951 |
| 3 | 0.99978 |
| 4 | 0.99986 |
| **Média** | **0.99975 ± 0.00012** |

---

## Decisões técnicas principais

**Deduplicação (MD5 + pHash):** para garantir a ausência de data leakage entre amostras e detectar radiografias visualmente similares de um mesmo paciente.

**StratifiedGroupKFold:** para manter a proporção de classes em cada fold e impedir que imagens do mesmo grupo deem overlap entre treino e validação.

**EfficientNet-B0:** backbone pré-treinado em ImageNet com Noisy Student, escolhido pelo bom custo-benefício entre performance e tempo de treinamento.

**`pos_weight` no BCEWithLogitsLoss:** compensa o desbalanceamento entre as classes NORMAL e PNEUMONIA, penalizando mais os erros na classe minoritária.

**Augmentations:** rotações leves, flips horizontais e variações suaves de brilho/contraste, escolhas deliberadas para não distorcer padrões anatômicos clinicamente relevantes nas radiografias.

**TTA (Test-Time Augmentation):** médias de predições com múltiplas variações da imagem de teste para reduzir variância na inferência final.
