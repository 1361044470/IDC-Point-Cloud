# Iterative Mutation-Point Detection and Completion (IDC) 
**For High-Ratio Outlier Removal in 3D Point Clouds**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20046692.svg)](https://doi.org/10.5281/zenodo.20046692)

This repository contains the official MATLAB implementation of the **IDC framework**. The code is provided to enhance the transparency and reproducibility of our research.

## 📝 Related Publication
This repository contains the official implementation of the following paper:

**Title**: Iterative Mutation-Point Detection and Completion for High-Ratio Outlier Removal in 3D Point Clouds

**Journal**: *Multimedia Systems*

If you find this code or our framework helpful in your research, please consider citing our paper using the following format:

**Plain Text:**
Manuscript: Iterative Mutation-Point Detection and Completion for High-Ratio Outlier Removal in 3D Point Clouds
Status: Under review

**BibTeX:**
```bibtex
@misc{shen2026idc,
  title={Iterative Mutation-Point Detection and Completion for High-Ratio Outlier Removal in 3D Point Clouds},
  author={Shen, Xiuqing and Luo, Mingling and Wu, Qi and Xie, Shoulie and Chen, Bin and Wu, Shiqian},
  year={2026},
  note={Manuscript under review}
}
```

## 📁 Repository Structure
To facilitate easy verification of our algorithm, we provide a clean and lightweight repository structure:
- `IDC.m`: The main pipeline for the synthetic dataset (includes ground-truth evaluation metrics like ODR, Recall, Accuracy).
- `IDCWithoutLabels.m`: The robust pipeline for real scanned models and large-scale indoor scenes (without ground truth).
- `rlhh.m`: The core function for adaptive mutation-point detection.
- `Data/`: Contains representative samples used in our paper. *Note: Due to file size limits on GitHub, massive indoor scenes are not included, but the algorithm naturally scales to such datasets.*

## ⚙️ Requirements
- MATLAB (Tested on R2023a, but backward compatible with earlier versions).
- MATLAB Computer Vision Toolbox (for point cloud operations).

## 🚀 Quick Start
We have configured the scripts to be plug-and-play. All file paths are strictly relative.
1. Download or clone this repository to your local machine.
2. Open MATLAB and navigate to the `IDC-Point-Cloud` folder.
3. **To test on synthetic data:** Open and run `IDC.m`. It will automatically read the point cloud, perform outlier removal, generate the threshold selection curve, and output the quantitative metrics table.
4. **To test on real scanned data:** Open and run `IDCWithoutLabels.m`. It will process the real scanned point cloud and seamlessly export a high-resolution comparison figure.

## ✉️ Contact
For any questions regarding the code or the paper, please feel free to open an issue or contact the authors.
