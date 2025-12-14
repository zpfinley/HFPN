# Hierarchical Fusion and Prediction Network (HFPN) with Multi-level Cross-modality Relation Learning Framework for Audio-visual Event Localization

- ### 1 Download data
    The AVE dataset is available [here](https://drive.google.com/file/d/1FjKwe79e0u96vdjIVwfRQ1V6SoDHe7kK/view).

    The VGGSound-AVEL100k dataset is available [here](https://drive.google.com/drive/folders/1en1dks1GYiGaDS9Ar-QtJmmyoOdzEsQj?usp=sharing).

- ### 2 Create environment and install package
  ```
  conda env create -f environment.yml
  ```
  
- ### 3 Extract features
  ```
  // extract CNN14 feature
  1. cd feature_extraction_tutorial/CNN14
  2. sh run.sh
  // extract SwinT feature
  1. cd feature_extraction_tutorial/SwinTrans
  2. python extract_swin_feat.py
  ```

- ### 4 Training
  ```
  sh supv_train.sh
  ```
- ### 5 Test
  ```
  sh supv_test.sh
  ```
