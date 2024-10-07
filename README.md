# 客戶流失預測模型

這個專案實現了一個機器學習模型集成，用於預測客戶流失的可能性。

## 功能概述

1. 數據預處理：加載和處理訓練和測試數據集。
2. 模型訓練：訓練多個機器學習模型，包括 Logistics Regression、XGBoost 和 LightGBM。
3. 模型集成：使用平均法結合多個模型的預測結果。
4. 模型評估：計算並顯示 Classification Report，包含各分類的及加權的 f1-score。
5. 視覺化：生成 Feature Importance, Confusion Matrix。

## 環境需求

- Python 3.9
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- xgboost
- lightgbm

您可以使用以下命令安裝所需的套件：

```
pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm
```

## 使用方法

1. 確保 `train.csv` 和 `test.csv` 檔案與程式碼在同一目錄下。
2. 運行程式：

```
python main.py
```

## 輸出說明

程式執行後將生成以下輸出：

1. **predictions.csv**：包含每個客戶的 CustomerID 和 Predicted_exit（0 或 1）。

2. **控制台輸出**：
   - Classification Report
      - 分類0: precision, recall, f1
      - 分類1: precision, recall, f1
      - 全樣本 accuracy
      - macro avg: 簡單平均
      - weighted avg: 加權平均

3. **feature_importance.png**：顯示 XGBoost 和 LightGBM 模型的特徵重要性。

4. **confusion_matrix.png**：集成模型的混淆矩陣視覺化。

## 程式結構

- `load_and_preprocess_data()`: 加載和預處理數據
- `train_models()`: 訓練多個機器學習模型
- `evaluate_model()`: 評估模型性能
- `ensemble_predict()`: 使用集成方法進行預測
- `plot_feature_importance()`: 繪製特徵重要性圖
- `plot_confusion_matrix()`: 繪製混淆矩陣圖
- `main()`: 主函數，協調整個流程

## 單模式的 Classification Report
   ### Random Forest (基礎模式，最後沒加入 Ensemble Model)
```
              precision    recall  f1-score   support

           0       0.89      0.95      0.92     26052
           1       0.73      0.54      0.62      6955

    accuracy                           0.86     33007
   macro avg       0.81      0.74      0.77     33007
weighted avg       0.85      0.86      0.85     33007
```

   ### Gradient Boosting (基礎模式，最後沒加入 Ensemble Model)
```
              precision    recall  f1-score   support

           0       0.89      0.95      0.92     26052
           1       0.75      0.55      0.63      6955

    accuracy                           0.87     33007
   macro avg       0.82      0.75      0.77     33007
weighted avg       0.86      0.87      0.86     33007
```
   ### Logistics Regression (基礎模式，最後有加入 Ensemble Model)
```
              precision    recall  f1-score   support

           0       0.85      0.95      0.90     26052
           1       0.70      0.39      0.50      6955

    accuracy                           0.84     33007
   macro avg       0.78      0.67      0.70     33007
weighted avg       0.82      0.84      0.82     33007
```

   ### XGBoost
```
              precision    recall  f1-score   support

           0       0.89      0.95      0.92     26052
           1       0.74      0.56      0.64      6955

    accuracy                           0.87     33007
   macro avg       0.81      0.75      0.78     33007
weighted avg       0.86      0.87      0.86     33007
```

   ### LightGBM

   ```
                precision    recall  f1-score   support

           0       0.89      0.95      0.92     26052
           1       0.75      0.57      0.64      6955

    accuracy                           0.87     33007
   macro avg       0.82      0.76      0.78     33007
weighted avg       0.86      0.87      0.86     33007
```

## Ensemble Model 最終將最好的3個模式加入

### Logistics + XGBoost + LightGBM
```
           0       0.88      0.95      0.92     26052
           1       0.76      0.53      0.63      6955

    accuracy                           0.87     33007
   macro avg       0.82      0.74      0.77     33007
weighted avg       0.86      0.87      0.86     33007
```

最終模式的結果 accuracy 為 0.87，沒退出的類別 f1 = 0.92，有退出的類別 f1 = 0.63。