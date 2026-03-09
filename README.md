# From Raw to Clean: An End-to-End Pipeline for Time-Series Cleaning with Diffusion Models

> Hello, sweet world!

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch)
![Docker](Add Docker container and docker compose logo) 
![License] (Add Apache 2.0 logo)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)


## Overview
**WaveStitch+** is a refinement of WaveStitch[add ref], that combines RePaint, Hann weighted windowing, and DDIM sampling strategies with Expectation-Maximazition (EM) training loop to handle conditional time-series gaps in the test data, using models trained on data that may also contain missing data. In this work, we encapsulate the WaveStitch+ as a reusable and reproducible building block to facilitate an end-to-end data cleaning pipeline for time series.  

---

<!-- ##  Table of Contents

- [Features](#features)
- [Installation & Setup](#installation--setup)
- [Usage Examples](#usage-examples)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

--- -->

<!-- ## Features -->


--- 


## Installation & Setup

### Requirements

- Python 3.10.18
- PyTorch 2.8
- CUDA 12.8.90 *(required, for GPU acceleration and inference parallelism)*

### Clone & Build Images & Install Dependencies
#### Git Clone

```bash
git clone https://github.com/judiths1618/6G-Data-process.git
```

#### Build WaveStitchPlus Docker Image
```
cd 6G-Data-process/dockers/tools
bash build image.sh
```

#### Start Apache Airflow and SeaWeedFS services
```
cd 6G-Data-process/dockers
bash start.sh
```

#### (Optional) Install Dependencies

> find therequirements.txt path and run
```bash
pip install -r requirements.txt
```

### (Optional) Set Up a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate       # Linux/macOS
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

---
## Usage Examples

### Run Basic WaveStitch Pipeline
> WaveStitchPlus Pipeline - S3-backed data lake storage: load --> train (save models) + inference --> store curated data.

```
cd 6G-Data-process/dockers
bash test docker.sh
```


### Run the End-to-End Data Quality and Cleaning Pipeline
Log into http://localhost:8088 with Admin username and password
Find the pipeline named "data_quality_and_cleaning_pipeline" in DAGs and click "Run" in Actions


### Evaluate Repair Quality

```
cd 6G-Data-process/notebooks
```
Run Jupyter notebooks:
> generate diagnostic plots
```
notebooks/analysis of the ts imputation.ipynb 
```
> generate comparison plots
```
notebooks/comparisons.ipynb
```
> generate plots with all features grid
```
notebooks/visual.ipynb
```
---

## Project Structure

```
6G-DATA-PROCESS/
├── 6GDALI_Datasets/                   # Raw data (gitignored)
│   ├── EUR        
│   └── KUL
├── dockers/
│   ├── airflow         # Core pipeline orchestration
│   │   ├── dags
│   │   ├── logs
│   │   ├── plugins
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── tools/            # Individual diffusion-based repair modules 
│   │   ├── scripts/
│   │   ├── waveStitchPlus_app
│   │   └── build image.sh
│   ├── docker-compose.yml   
│   ├── fix errors.sh          
│   ├── start.sh               # start docker services by runing docker compose
│   └── test docker.sh         # Test the WaveStitchPlus-gpu:latest docker container
├── notebooks/            # Jupyter notebooks for data analytics

├── LICENSE               # Apache 2.0
├── .gitignore
└── README.md
```

---


## License

This project is licensed under the **Apache 2.0 License**. See [LICENSE](LICENSE) for details.

---

<!-- ## Citation

If you use WaveStitchPlus in your research, please cite:

```bibtex

}
``` -->