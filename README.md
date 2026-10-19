# Paranasal Sinus CT Segmentation with 2D U-Net

**Supplementary data repository** for the manuscript submitted to *Pediatric Radiology* (Springer).

> **"Automated Segmentation of Paranasal Sinuses in Pediatric CT Using 2D U-Net: Volume Estimation and Age/Gender Prediction"**

---

## Repository Structure

```
├── code/                        # Python source code
│   ├── asama1_veri_topla.py     # Step 1: Data collection & preprocessing
│   ├── asama2_model_egit.py     # Step 2: U-Net model training
│   ├── asama3_tam_pipeline_test.py          # Step 3: Full pipeline test
│   ├── asama3_tam_pipeline_test_toplu.py    # Batch testing (multiple patients)
│   └── asama3_tam_pipeline_test_tek_kisi.py # Single patient testing
│
├── results/
│   ├── toplu_test_sonuclari.xlsx       # Main results: 30 test patients (Dice, IoU, volumes, age/gender predictions)
│   └── segmentasyon_raporu_uzman.xlsx  # Detailed segmentation metrics vs. expert reference
│
├── stl/                         # 3D mesh files for all 30 test patients
│   ├── {patient_id}_Gercek.stl  # Expert (ground truth) segmentation
│   └── {patient_id}_Tahmin.stl  # AI model prediction
│
└── images/                      # Example segmentation visualizations
```

---

## Methods Summary

- **Model:** 2D U-Net (TensorFlow/Keras)
- **Input:** Pediatric paranasal sinus CT scans (NIfTI format)
- **Preprocessing:** HU windowing (WC=400, WW=2000), canonical reorientation
- **Training set:** 132 patients
- **Test set:** 30 patients (independent)
- **Segmentation target:** Maxillary sinuses (bilateral)
- **Secondary tasks:** Age regression (Random Forest), Gender classification (Random Forest)

## Key Results (30 test patients)

| Metric | Value |
|--------|-------|
| Dice coefficient | 0.87 ± 0.09 |
| IoU | 0.78 |
| Precision | 0.82 |
| Recall | 0.93 |
| Age prediction MAE | 2.42 years |
| Gender accuracy | 46.7% |

---

## STL Files

The `stl/` directory contains 3D surface meshes generated via marching cubes algorithm from the binary segmentation masks. Each patient has two files:
- `*_Gercek.stl` — Expert-annotated ground truth
- `*_Tahmin.stl` — AI model prediction

These can be viewed with any STL viewer (MeshLab, 3D Slicer, Windows 3D Viewer, etc.).

---

## Notes

- Raw CT scan data (NIfTI files) are **not included** in this repository due to patient privacy regulations (KVKK/GDPR).
- Results in `toplu_test_sonuclari.xlsx` correspond exactly to the 30-patient test cohort reported in the manuscript.
- Volume unit: cm³

---

## Contact

For questions regarding this work, please contact the corresponding author.
