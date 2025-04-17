## 版本記錄（2025/04/17）

### ✅ 版本一：使用邏輯斯回歸（Logistic Regression）
- 模型：Logistic Regression
- 特徵：
  - 使用 FFT 頻域分析
  - 範例程式碼為 FFT 頻譜特徵提取參考

---

### ✅ 版本二：使用 GBDT + CatBoost，評估指標 Macro AUC (以去除k-fold)
- 模型：GBDT + CatBoost
- 參數與特徵加入：
  - 📌 三軸合成向量：
    - `AccMag`：加速度合成向量
    - `GyroMag`：陀螺儀合成向量
  - 📌 時間窗設定：
    - `seg_id`：片段編號
    - `segment_length`：片段長度
  - 📌 FFT 頻率能量分布：
    - `fft_peak1`：最大能量峰值佔比
    - `fft_peak2`：次大峰值佔比
    - `fft_peak3`：第三大峰值佔比

---
