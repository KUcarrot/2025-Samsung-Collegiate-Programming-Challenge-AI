## 2025-Samsung-Collegiate-Programming-Challenge-AI-
Dacon : https://dacon.io/competitions/official/236500/overview/description

**3rd Prize**, Private 12위 (Accuracy: 0.7819)

## File Download Link (data & models)
[Google Drive](https://drive.google.com/drive/folders/1epH3ukBbnumAVUxVUQh3dgHLGEAtuhSH?usp=sharing)


## Environment
```
OS: Google Colab Pro (Ubuntu 22.04.4 LTS)
Python 버전: 3.11.13
주요 라이브러리: Transformers, bitsandbytes, PyTorch 등 (상세 버전은 `requirements.txt` 참고)
하드웨어: GPU (NVIDIA A100-SXM4-40GB)
```


## Directory
```
├── data/
│   ├── test.csv
│   └── test_input_images/
├── models/
│   ├── instructblip-flan-t5-xl/
│   ├── flan-t5-large/
│   ├── blip-image-captioning-large/
│   └── all-mpnet-base-v2/
├── README.md
├── requirements.txt
├── Execution.ipynb
├── utils.py
└── inference.py
```
## 실행 방법
**0. 파일 다운**

```
Google Drive에서 data.zip & models.zip 다운 후 압축 해제
```

**1. 경로 설정**

```bash
from google.colab import drive
drive.mount('/content/drive')

import os
# 본인의 구글 드라이브 경로에 맞게 수정.
project_path = "/content/drive/MyDrive/"
os.chdir(project_path)

# 현재 위치가 프로젝트 폴더로 잘 바뀌었는지 확인
!pwd
```

**2. 라이브러리 설치**


```bash
!pip install -r requirements.txt
```

**3. 추론파일 실행**

```bash
!python inference.py --data_dir ./data --model_dir ./models --output_dir ./submission
```

## Reference
```
@article{wu2023solution,
  title={Solution for smart-101 challenge of iccv multi-modal algorithmic reasoning task 2023},
  author={Wu, Xiangyu and Yang, Yang and Xu, Shengdong and Wu, Yifeng and Chen, Qingguo and Lu, Jianfeng},
  journal={arXiv preprint arXiv:2310.06440},
  year={2023}
}
