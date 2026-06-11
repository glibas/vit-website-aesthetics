# Website aesthetics prediction with vision transformers

[![DOI](https://zenodo.org/badge/1265547995.svg)](https://doi.org/10.5281/zenodo.20639731)

Training and evaluation code for the paper:

> G. Briia, I. Turkin. Automated website aesthetic evaluation using Vision Transformers with progressive fine-tuning., 2026.

The paper compares three pre-training paradigms (supervised ViT/DeiT, self-supervised DINOv2, vision-language CLIP) for predicting human aesthetic ratings of website screenshots. Models are adapted with a three-phase fine-tuning protocol: the regression head is trained first, then the last four encoder blocks with layer-wise learning rate decay, then the full network at a reduced rate. CLIP-Base came out on top (Pearson r = 0.680 on the WDP test set, r = 0.720 in 5-fold CV on the R&G dataset).

## Notebooks

| Notebook | What it does | Paper |
|---|---|---|
| `VIT Benchmark - Models comparison.ipynb` | Fine-tunes nine ViT variants on a stratified 68/12/20 split of WDP and compares paradigms | Table 3, Figs. 1-2 |
| `VIT Benchmark - hyperparameters tuning.ipynb` | Random search (20 of 432 configurations) over learning rates, weight decay, LLRD factor and loss function for CLIP-Base | Section 3.2 |
| `VIT Benchmark - WDP dataset 5-fold CV.ipynb` | Stratified 5-fold cross-validation of CLIP-Base on WDP, with per-category results | Table 5, Fig. 3 |
| `VIT Benchmark - R&G dataset 5-fold CV.ipynb` | 5-fold cross-validation on the Reinecke & Gajos dataset | Tables 5, 9 |
| `VIT Benchmark - WAD20 dataset 5-fold CV.ipynb` | 5-fold cross-validation on WAD20 | Table 5 |
| `VIT Benchmark - WDP dataset 5-fold CV - Above fold.ipynb` | Input-scope ablation: WDP screenshots cropped to the above-the-fold region | Table 7 |
| `VIT Benchmark - cross dataset.ipynb` | Trains on one dataset, evaluates on the other two without fine-tuning | Table 6 |

Cell outputs from the actual runs are kept in the notebooks, so the logged numbers can be checked against the paper without re-running anything.

## Datasets

The datasets are not redistributed here. All three are published by their original authors:

| Dataset | Contents | Source |
|---|---|---|
| WDP | 3,136 full-page homepage screenshots (banks, fashion, homeware, universities) with mean aesthetics ratings | Miniukovich & Figl, *Data in Brief*, 2024. [doi:10.1016/j.dib.2023.109976](https://doi.org/10.1016/j.dib.2023.109976) |
| R&G | ~400 above-the-fold screenshots with visual appeal ratings from LabintheWild | Reinecke & Gajos, CHI 2014. [doi:10.1145/2556288.2557052](https://doi.org/10.1145/2556288.2557052) |
| WAD20 | 540 above-the-fold screenshots with aesthetics ratings | Eisbach, Kuss & Klichowicz, *ACM TOCHI*, 2023. [doi:10.1145/3569889](https://doi.org/10.1145/3569889) |

## Running

The notebooks were written for Google Colab and were run on a single NVIDIA Tesla T4 (16 GB). Each notebook mounts Google Drive and expects the dataset archives at:

```
MyDrive/datasets/webdesignprototypicality.zip
MyDrive/datasets/ReineckeGajos.zip
MyDrive/datasets/WAD20.zip
```

Results (logs, metrics JSON, prediction CSVs) are written to `MyDrive/results/vit_benchmark/`. Dependencies:

```
pip install transformers timm accelerate scipy pandas scikit-learn pillow tqdm matplotlib psutil
```

Outside Colab, drop the `google.colab` calls in each `Config.__post_init__` and point the path constants at local copies of the datasets.

## License

MIT, see [LICENSE](LICENSE).

## Citation

```bibtex
@article{briia2026vit,
  author  = {Briia, Glib and Turkin, Ihor},
  title   = {Automated website aesthetic evaluation using Vision Transformers with progressive fine-tuning},
  journal = {},
  year    = {2026}
}
```
