![Python](https://img.shields.io/badge/Python-3.8.15-3670A0?style=flat-square&logo=python&logoColor=ffdd54)
![PyCharm](https://img.shields.io/badge/PyCharm-143?style=flat-square&logo=pycharm&logoColor=black&color=black&labelColor=green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.6.0-%23FF6F00.svg?style=flat-square&logo=TensorFlow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-1.13.0-%23EE4C2C.svg?style=flat-square&logo=PyTorch&logoColor=white)

# sdp-term-project

[2022 Samsung AI Challenge (3D Metrology)](https://dacon.io/competitions/official/235954/overview/description)

- **전자현미경(SEM) 이미지로부터 깊이를 예측하는 AI 알고리즘 개발**
- 실제 Hole 단위 SEM 영상으로부터 추론한 Depth Map (.png, int)

- `Depth Map` - 실제 SEM Hole 영상의 픽셀 별 매칭되는 깊이 정보
- `Average Depth` - 전체 SEM 영상의 평균 깊이

- **Train Dataset (학습용 데이터셋, 학습 가능) - 총 60664개**
    - `SEM [폴더]` : 실제 SEM 영상을 Hole 단위로 분할한 영상 (8bit Gray 영상)
    - `average_depth.csv` : 전체 SEM 영상과 대응되는 평균 Depth
- **Simulation Dataset (학습용 데이터셋, 학습 가능) - 총 259956개**
    - `SEM [폴더]` : Simulator을 통해 생성한 Hole 단위 SEM 영상 (실제 SEM 영상과 유사하나, 대응 관계는 없음)
    - `Depth [폴더]` : Simulator을 통해 얻은 SEM 영상과 Pixel별로 대응되는 Depth Map
    - Depth 이미지 1개당 2개의 Simulator Hole 단위 SEM 영상이 Pair하게 매칭됩니다. (Name_itr0, Name_itr1)
- **Test Dataset (평가를 위한 테스트 데이터셋, 학습 불가능) - 총 25988개**
    - `SEM [폴더]` : 실제 SEM 영상을 Hole 단위로 분할한 영상 (8bit Gray 영상)
- **sample_submission.zip (제출 양식) - 총 25988개**
    - 실제 Hole 단위 SEM 영상으로부터 추론한 Depth Map (PNG 파일)

- **학습이 가능한 데이터**
    - Train Dataset의 **Real SEM Hole** 영상
        - Simulation SEM Hole 영상과 비슷하지만, 대응관계는 없음
        - Depth Map이 주어지지 않음
        - 전체 SEM 영상의 Depth 평균인 average_depth 정보만 주어짐
    - Simulation Dataset의 **Simulation SEM Hole** 영상
        - 매칭되는 Depth Map으로부터 Simulator를 통해 생성된 영상
        - 즉, Simulation 폴더 내 Depth Map과 매칭되는 Simulation SEM Hole 영상은 대응관계가 존재함
- 실제 추론은 Real SEM Hole 영상으로만 이루어짐
