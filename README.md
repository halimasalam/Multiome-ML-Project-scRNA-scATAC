# ML-Based Immune Cell Classification from Multiome PBMC Data

##  Overview

This project combines **single-cell transcriptomics (scRNA-seq)** and **chromatin accessibility (scATAC-seq)** to predict immune cell types using **machine learning**.

Using multiome PBMC data from 10x Genomics, I trained classifiers to distinguish T cells, B cells, monocytes, NK cells, and others — based on both gene expression and chromatin accessibility.

---

##  Objective

- Classify immune cell types from single-cell omics data using ML.
- Compare transcriptomic vs epigenomic features for prediction.
- Integrate scRNA and scATAC data to explore regulatory biology.

---

##  Dataset

- **Source**: [10x Genomics Multiome PBMC 10k Dataset](https://www.10xgenomics.com/datasets/10-k-human-pbm-cs-multiome-v-1-0-chromium-x-1-standard-2-0-0)
- **Data types**:
  - scRNA-seq (5' expression matrix)
  - scATAC-seq (peak accessibility)
- ~10,000 human PBMCs

---

## ️ Methods

### scRNA-seq
- Quality filtering (n_genes, mitochondrial content)
- Normalization, log transformation
- PCA, UMAP, Leiden clustering
- Cell type annotation using canonical markers

### scATAC-seq
- Filtering low-quality cells
- LSI (Latent Semantic Indexing)
- UMAP & clustering
- Integration with RNA data using shared barcodes

###  Machine Learning
- **Classifier**: Random Forest / XGBoost
- **Input features**:
  - Top 1000 variable genes (RNA)
  - Top peaks or gene-linked ATAC accessibility
- **Output**: Cell type label
- **Evaluation**: Accuracy, precision, recall, confusion matrix

---

##  Results

- RNA-based classifier achieved **92% accuracy**
- ATAC-based classifier reached **85%**
- Top predictive genes: CD3D, MS4A1, LYZ
- ATAC peaks aligned with gene regulatory regions

![umap_rna](results/figures/umap_rna_clusters.png)  
![confusion_matrix](results/figures/confusion_matrix_rf.png)

---

## 📁 Project Structure
/data
/raw/
/processed/
/notebooks
rna_analysis.ipynb
atac_analysis.ipynb
ml_classifier.ipynb
/results
/figures/
/tables/
README.md
requirements.txt

## How to Run
```bash
git clone https://github.com/yourusername/multiome-ml-pbmc.git
cd multiome-ml-pbmc
pip install -r requirements.txt
jupyter notebook notebooks/rna_analysis.ipynb
```


## Future Work
- Use multi-modal deep learning (e.g., autoencoders)
- Extend to disease PBMCs (e.g., COVID, leukemia)
- Compare integration methods (WNN, Seurat v5, MuData)

## Acknowledgements
- Data from 10x Genomics
- Tools: Scanpy, scikit-learn, matplotlib, seaborn, Signac


