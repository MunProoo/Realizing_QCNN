# Realizing_QCNN

## QCNN ?
Convolutional Neural Network(CNN)는 많은 가중치를 포함하여 계산 비용이 많이 들기 때문에 고성능 하드웨어가 요구된다.  
Quantized Convolution Neural Network(QCNN)은 CNN의 메모리 용량을 경량화 하면서도 이미지 분류의 정확도는 떨어뜨리지 않기 위해 가중치들을 양자화하는 것이다.  
Mnist 데이터셋에 대해 QCNN으로 학습시키고, 웹캠으로 찍은 손 글씨 이미지를 인식한 결과 95%의 정확도를 확인하였다.   
이미지 분류 정확도가 CNN과 거의 유사한 정확도를 유지함과 동시에 가중치를 저장하는 메모리의 용량은 10.7배 감소하였다.  

## Quntized Process
![image](https://github.com/MunProoo/Realizing_QCNN/assets/52486862/28509274-105c-46a7-bb8b-d525f8636aac)  
QCNN의 더 자세한 설명은 하단의 논문에서 확인하실 수 있습니다.


## 인식 사진
![image](https://github.com/MunProoo/Realizing_QCNN/assets/52486862/43446996-0d5c-41fd-b7f3-f64978de1b8f)  


---
저자 : 
- 충북대학교 정보통신공학부 학사 문주영
- 정보통신공학부 학사 유준상
- 정보통신공학부 석사 이재흠
- 충북대학교 조경록 교수
  
[충북대학교 컴퓨터 정보통신 연구소](https://ricic.cbnu.ac.kr/ricic/journal_collection/37217)
