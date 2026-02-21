# Physics-Informed Synthetic Geotechnical Data Generator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Data License: CC BY 4.0](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

This repository contains the complete code, generated datasets, and validation plots for the paper:

**"Bridging Data Scarcity in Geotechnics: A Physics‑Informed Approach to Generate Synthetic Soil Records for Machine Learning Classification"**  
by Oluwashina Oyeniran and Opeyemi Adetunji (2026).

---

## 📖 Overview

Geotechnical engineering often suffers from **data scarcity**—site investigations typically yield only a handful of samples (often fewer than 20), which is insufficient for training robust machine learning models. This work presents a **physics‑informed hierarchical generative framework** that creates **10,000 synthetic soil samples** from just **10 real borehole samples**. The generation process respects:

- Statistical distributions of the original measurements.
- Empirical relationships (e.g., friction angle vs. density and grain size).
- Physical constraints (e.g., D60 ∈ [0.85, 4.75] mm).
- USCS classification rules for sands (Cu ≥ 6 and 1 ≤ Cc ≤ 3 for well‑graded sand).

The final dataset is **class‑balanced** (1,774 samples each of Well‑graded sand SW and Poorly graded sand SP) and ready for machine learning applications.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18zMpWaguJQCteRXHckxFlH_Ha3Io-_et?usp=sharing)
---
