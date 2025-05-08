# -*- coding: utf-8 -*-
"""
Created on 2025/5/8
0.81172903	

@author: User
"""
# 記得改輸出檔名
import os
import zipfile
import math
import numpy as np
import pandas as pd
from scipy.stats import entropy, kurtosis, skew
#from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


# ==== 資料解壓與讀檔（請依照實際路徑修改）====
train_zip = r"C:\Users\User\Downloads\39_Training_Dataset.zip"
test_zip = r"C:\Users\User\Downloads\39_Test_Dataset.zip"
extract_root = "aicup_data"
sample_path = r"C:\Users\User\Downloads\sample_submission.csv"

train_dir = os.path.join(extract_root, "train")
test_dir = os.path.join(extract_root, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

with zipfile.ZipFile(train_zip, 'r') as zip_ref:
    zip_ref.extractall(train_dir)
with zipfile.ZipFile(test_zip, 'r') as zip_ref:
    zip_ref.extractall(test_dir)

train_inner = os.path.join(train_dir, "39_Training_Dataset")
test_inner = os.path.join(test_dir, "39_Test_Dataset")

train_info = pd.read_csv(os.path.join(train_inner, "train_info.csv"))
test_info = pd.read_csv(os.path.join(test_inner, "test_info.csv"))


# =============================================================================
# FFT 與頻域特徵相關函式（整合至主程式流程）
# =============================================================================
def FFT(xreal, ximag):
    """
    手動實作 Cooley-Tukey FFT
    參數:
      xreal, ximag: 輸入信號的實部與虛部（list 或 array），長度至少為 2 的冪次
    回傳:
      n, xreal, ximag：n 為實際採用的 FFT 長度，xreal 與 ximag 為 FFT 轉換後的結果
    """
    # 找到不超過 len(xreal) 的最大 2 的冪次
    n = 2
    while n * 2 <= len(xreal):
        n *= 2
    p = int(math.log(n, 2))
    
    # Bit reversal 排序
    for i in range(n):
        a = i
        b = 0
        for j in range(p):
            b = b * 2 + (a % 2)
            a = a // 2
        if b > i:
            xreal[i], xreal[b] = xreal[b], xreal[i]
            ximag[i], ximag[b] = ximag[b], ximag[i]
    
    # 準備 twiddle factors
    wreal = [1.0]
    wimag = [0.0]
    arg = -2 * math.pi / n
    treal = math.cos(arg)
    timag = math.sin(arg)
    for j in range(1, n // 2):
        wreal.append(wreal[-1] * treal - wimag[-1] * timag)
        wimag.append(wreal[-1] * timag + wimag[-1] * treal)
    
    m = 2
    while m <= n:
        for k in range(0, n, m):
            for j in range(m // 2):
                index1 = k + j
                index2 = index1 + m // 2
                t = int(n * j / m)
                treal_temp = wreal[t] * xreal[index2] - wimag[t] * ximag[index2]
                timag_temp = wreal[t] * ximag[index2] + wimag[t] * xreal[index2]
                ureal = xreal[index1]
                uimag = ximag[index1]
                xreal[index1] = ureal + treal_temp
                ximag[index1] = uimag + timag_temp
                xreal[index2] = ureal - treal_temp
                ximag[index2] = uimag - timag_temp
        m *= 2

    return n, xreal, ximag

def FFT_data(input_data, swinging_times):
    """
    依據各擺動區段時間點計算區段內的加速度與陀螺儀向量均值
    參數:
      input_data    : 2D 數據，每列為 [Ax, Ay, Az, Gx, Gy, Gz]
      swinging_times: 擺動區段起始 index 列表，例如 [start0, start1, ..., end]
    回傳:
      a_mean, g_mean: 每一區段的加速度與陀螺儀向量均值（依序排列）
    """
    num_segments = len(swinging_times) - 1
    a_mean = [0] * num_segments
    g_mean = [0] * num_segments
    
    for i in range(num_segments):
        a_vals = []
        g_vals = []
        for j in range(swinging_times[i], swinging_times[i+1]):
            a_val = math.sqrt(input_data[j][0]**2 + input_data[j][1]**2 + input_data[j][2]**2)
            g_val = math.sqrt(input_data[j][3]**2 + input_data[j][4]**2 + input_data[j][5]**2)
            a_vals.append(a_val)
            g_vals.append(g_val)
        a_mean[i] = sum(a_vals) / len(a_vals) if a_vals else 0
        g_mean[i] = sum(g_vals) / len(g_vals) if g_vals else 0
    return a_mean, g_mean

def extract_fft_features(signal):
    """
    以手動 FFT 計算信號的頻域特徵，並攤平成：
      - fft_entropy, fft_kurtosis, fft_skewness, fft_power
      - fft_peak1, fft_peak2, fft_peak3 (前三大頻率峰值佔總能量比)
    """
    L = len(signal)
    # 補零至 2 的冪次
    n_fft = 1
    while n_fft < L:
        n_fft *= 2
    sig = list(signal) + [0] * (n_fft - L)
    xreal = sig.copy()
    ximag = [0] * n_fft
    n, fft_real, fft_imag = FFT(xreal, ximag)

    # 計算功率譜密度 (PSD)
    psd = np.array([fft_real[i]**2 + fft_imag[i]**2 for i in range(n)])
    total = psd.sum() + 1e-9
    psd_norm = psd / total

    feats = {
        'fft_entropy':  entropy(psd_norm),
        'fft_kurtosis': kurtosis(psd),
        'fft_skewness': skew(psd),
        'fft_power':    psd.mean(),
    }

    # 攤平成三大峰值比
    idxs = np.argsort(psd)[-3:][::-1]
    for i, idx in enumerate(idxs, start=1):
        feats[f'fft_peak{i}'] = psd[idx] / total
    
    psd = np.array([fft_real[i]**2 + fft_imag[i]**2 for i in range(n)])
    freqs = np.arange(n) / n
    total = psd.sum() + 1e-9
    psd_norm = psd / total
    # 新增特徵
    centroid = np.sum(freqs * psd_norm)
    spread = np.sqrt(np.sum(((freqs - centroid) ** 2) * psd_norm))
    feats.update({
        'fft_centroid': centroid,
        'fft_spread': spread
    })
    return feats


def extract_segment_features(seg, seg_id):
    """
    計算單一段 (seg) 的時域 & 頻域特徵，並加入：
      - 三軸合成向量 AccMag/GyroMag
      - seg_id, segment_length
    回傳一個扁平的 dict，所有特徵都已攤平成欄位。
    """
    features = {}

    # 1. 計算 magnitude
    acc_mag  = np.linalg.norm(seg[:, :3], axis=1)
    gyro_mag = np.linalg.norm(seg[:, 3:], axis=1)

    # 2. 通道清單
    names  = ["Ax","Ay","Az","Gx","Gy","Gz","AccMag","GyroMag"]
    arrays = [*seg.T, acc_mag, gyro_mag]

    # 3. 時域 + 頻域 特徵
    for name, arr in zip(names, arrays):
        # 時域
        features[f"{name}_mean"] = arr.mean()
        features[f"{name}_std"]  = arr.std()
        features[f"{name}_rms"]  = np.sqrt((arr**2).mean())
        features[f"{name}_min"]  = arr.min()
        features[f"{name}_max"]  = arr.max()
        features[f"{name}_zcr"] = np.mean(np.diff(np.sign(arr)) != 0)
        features[f"{name}_kurtosis"] = kurtosis(arr)
        features[f"{name}_skewness"] = skew(arr)
        # 頻域
        fft_feats = extract_fft_features(arr)
        for k, v in fft_feats.items():
            features[f"{name}_{k}"] = v

    # 4. 段落資訊
    features["seg_id"]         = seg_id
    features["segment_length"] = len(seg)

    return features


# =============================================================================
# 原始資料處理與特徵擷取流程（直接用 FFT 特徵）
# =============================================================================

def analyze_segment_statistics(info_df, data_dir):
    lengths, rms_list, p2p_list = [], [], []
    for _, row in info_df.iterrows():
        uid = row["unique_id"]
        txt_path = os.path.join(data_dir, f"{uid}.txt")
        try:
            raw = np.loadtxt(txt_path)
            cut = [int(x) for x in row["cut_point"].strip("[]").split()]
            cut = [0] + cut + [len(raw)]
            segments = [raw[cut[i]:cut[i+1]] for i in range(min(27, len(cut)-1))]
            for seg in segments:
                if len(seg) < 3:
                    continue
                acc = np.linalg.norm(seg[:, :3], axis=1)
                rms = np.sqrt(np.mean(acc**2))
                p2p = np.ptp(acc)
                lengths.append(len(seg))
                rms_list.append(rms)
                p2p_list.append(p2p)
        except:
            continue

    def plot_and_summary(data, name):
        # 這裡是畫圖和印統計的程式碼（不變）
        q25, q75 = np.percentile(data, [25, 75])
        iqr = q75 - q25
        lower = q25 - 1.5 * iqr
        upper = q75 + 1.5 * iqr
        print(f"{name} 建議篩選區間: [{lower:.1f}, {upper:.1f}]")
        return lower, upper  # 回傳上下限

    # 計算每個指標的區間
    length_lower, length_upper = plot_and_summary(lengths, "Segment Length")
    rms_lower, rms_upper = plot_and_summary(rms_list, "RMS Acceleration")
    p2p_lower, p2p_upper = plot_and_summary(p2p_list, "Peak-to-Peak Acceleration")

    # 把區間存進字典
    thresholds = {
        'length': [length_lower, length_upper],
        'rms': [rms_lower, rms_upper],
        'p2p': [p2p_lower, p2p_upper]
    }
    return thresholds  # 回傳字典


def is_valid_segment(seg, thresholds):
    length = len(seg)
    # 檢查長度是否在區間內
    if length < thresholds['length'][0] or length > thresholds['length'][1]:
        return False
    acc = np.linalg.norm(seg[:, :3], axis=1)
    rms = np.sqrt(np.mean(acc**2))
    p2p = np.ptp(acc)
    # 檢查 RMS 和 峰值差是否在區間內
    if rms < thresholds['rms'][0] or rms > thresholds['rms'][1]:
        return False
    if p2p < thresholds['p2p'][0] or p2p > thresholds['p2p'][1]:
        return False
    return True

def create_feature_dataframe(info_df, data_dir, thresholds, with_labels=True):
    rows = []
    for _, row in info_df.iterrows():
        uid = row["unique_id"]
        txt_path = os.path.join(data_dir, f"{uid}.txt")
        try:
            raw_data = np.loadtxt(txt_path)
            cut_points = [int(x) for x in row["cut_point"].strip("[]").split()]
            cut_points = [0] + cut_points + [len(raw_data)]
            segments = [raw_data[cut_points[i]:cut_points[i+1]] for i in range(min(27, len(cut_points)-1))]
            for i, seg in enumerate(segments):
                if not is_valid_segment(seg, thresholds):  # 傳入 thresholds
                    continue
                feat = extract_segment_features(seg, seg_id=i)
                feat["unique_id"] = uid
                if with_labels:
                    feat["gender"] = 1 if row["gender"] == 1 else 0
                    feat["hand"] = 1 if row["hold racket handed"] == 1 else 0
                    feat["play"] = row["play years"]
                    feat["level"] = row["level"]
                rows.append(feat)
        except Exception as e:
            continue
    return pd.DataFrame(rows)

# =============================================================================
# 模型訓練與驗證 ( GBDT + CatBoost + Macro AUC)
# =============================================================================
#from sklearn.ensemble import GradientBoostingClassifier


# 在這裡加入：獲取閾值
thresholds = analyze_segment_statistics(train_info, os.path.join(train_inner, "train_data"))

# 1. 建立訓練資料（傳入 thresholds）
train_df = create_feature_dataframe(train_info, os.path.join(train_inner, "train_data"), thresholds, with_labels=True)
print(train_df.columns.tolist())

# 2. 先以 unique_id 做一次 80/20 split
#unique_ids = train_df["unique_id"].unique()
#train_ids, val_ids = train_test_split(
#    unique_ids,
#    test_size=0.2,
#    random_state=42,
#    shuffle=True
#)
#train_part = train_df[train_df["unique_id"].isin(train_ids)].reset_index(drop=True)
#val_part   = train_df[train_df["unique_id"].isin(val_ids)].reset_index(drop=True)

from sklearn.model_selection import GroupShuffleSplit

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(splitter.split(train_df, groups=train_df["unique_id"]))

train_part = train_df.iloc[train_idx].reset_index(drop=True)
val_part   = train_df.iloc[val_idx].reset_index(drop=True)

# 3. 標準化（移除標籤欄位）
X_train = train_part.drop(columns=["unique_id", "gender", "hand", "play", "level"])
X_val   = val_part.drop(columns=["unique_id", "gender", "hand", "play", "level"])
scaler  = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_val_scaled   = scaler.transform(X_val)



from flaml import AutoML

# 確保標籤型別為 int
y_gender = train_part["gender"].astype(int)
y_hand   = train_part["hand"].astype(int)
y_play   = train_part["play"].astype(int)
y_level  = train_part["level"].astype(int)

# 訓練 gender（二分類）
automl_gender = AutoML()
automl_gender.fit(
    X_train_scaled, y_gender,
    task="classification",
    metric="roc_auc",  # 二分類用 AUC
    time_budget=3600,    # 每個模型時間上限（秒）
    verbose=1
)

# 訓練 hand（二分類）
automl_hand = AutoML()
automl_hand.fit(
    X_train_scaled, y_hand,
    task="classification",
    metric="roc_auc",
    time_budget=3600,
    verbose=1
)

# 訓練 play（多分類）
automl_play = AutoML()
automl_play.fit(
    X_train_scaled, y_play,
    task="classification",
    metric="log_loss",  # 多分類建議用 log_loss（也可自定）
    time_budget=3600,
    verbose=1
)

# 訓練 level（多分類）
automl_level = AutoML()
automl_level.fit(
    X_train_scaled, y_level,
    task="classification",
    metric="log_loss",
    time_budget=3600,
    verbose=1
)


# 6. 驗證集評估 (Macro AUC)
import pandas as pd

def evaluate_on_val_set(val_df):
    print("\n🔍 評估驗證集...")
    preds, truths = [], []

    for uid, group in val_df.groupby("unique_id"):
        Xg = scaler.transform(group.drop(columns=["unique_id", "gender", "hand", "play", "level"]))
        y = group.iloc[0]  # 該 ID 的標籤
        
        # FLAML 預測
        gender_probs = automl_gender.predict_proba(Xg)[:, 1]
        hand_probs   = automl_hand.predict_proba(Xg)[:, 1]
        play_probs   = automl_play.predict_proba(Xg)
        level_probs  = automl_level.predict_proba(Xg)

        # top-k 平均（可自訂 k）
        def top_k_mean(probs, k=3):
            scores = np.abs(probs - 0.5)
            topk_idx = np.argsort(scores)[-k:]
            return probs[topk_idx].mean()

        pg = top_k_mean(gender_probs)
        ph = top_k_mean(hand_probs)
        pp = play_probs[np.argsort(np.abs(play_probs - 1/3).max(axis=1))[-3:]].mean(axis=0)
        pl = level_probs[np.argsort(np.abs(level_probs - 0.25).max(axis=1))[-3:]].mean(axis=0)

        preds.append({
            "gender": pg,
            "hand":   ph,
            **{f"play_{i}": pp[i] for i in range(len(pp))},
            **{f"level_{i+2}": pl[i] for i in range(len(pl))}
        })
        truths.append({
            "gender": y["gender"],
            "hand":   y["hand"],
            "play":   y["play"],
            "level":  y["level"]
        })

    pred_df  = pd.DataFrame(preds)
    truth_df = pd.DataFrame(truths)

    # 計算 AUC（binary: macro ROC AUC，multi: OVR ROC AUC）
    auc_gender = roc_auc_score(truth_df["gender"], pred_df["gender"], average="macro")
    auc_hand   = roc_auc_score(truth_df["hand"],   pred_df["hand"],   average="macro")
    y_play     = pd.get_dummies(truth_df["play"])
    y_level    = pd.get_dummies(truth_df["level"])
    auc_play   = roc_auc_score(y_play,  pred_df.filter(like="play_"),  multi_class='ovr', average="macro")
    auc_level  = roc_auc_score(y_level, pred_df.filter(like="level_"), multi_class='ovr', average="macro")
    
    final_score = 0.25 * (auc_gender + auc_hand + auc_play + auc_level)

    print(f"Gender AUC : {auc_gender:.4f}")
    print(f"Hand   AUC : {auc_hand:.4f}")
    print(f"Play   AUC : {auc_play:.4f}")
    print(f"Level  AUC : {auc_level:.4f}")
    print(f"Final AUC : {final_score:.4f}\n")

    return final_score


# 執行評估
evaluate_on_val_set(val_part)

# =============================================================================
# Test set 預測並輸出結果
# =============================================================================
print("開始 Test set 預測...")
all_results = []
fallback_count = 0
total_count = 0

for uid in test_info["unique_id"]:
    total_count += 1
    txt_path = os.path.join(test_inner, "test_data", f"{uid}.txt")
    try:
        raw_data = np.loadtxt(txt_path)
        row = test_info[test_info["unique_id"] == uid].iloc[0]
        cut_points = [int(x) for x in row["cut_point"].strip("[]").split()]
        cut_points = [0] + cut_points + [len(raw_data)]
        segments = [
            raw_data[cut_points[i]:cut_points[i+1]]
            for i in range(min(27, len(cut_points)-1))
        ]

        # 使用 thresholds 篩選片段
        valid_feats = []
        for i, seg in enumerate(segments):
            if not is_valid_segment(seg, thresholds):
                continue
            valid_feats.append(extract_segment_features(seg, seg_id=i))

        # fallback
        if not valid_feats:
            fallback_count += 1
            fallback_seg = raw_data[:100] if len(raw_data) >= 100 else raw_data
            valid_feats = [extract_segment_features(fallback_seg, seg_id=0)]

        # 預測（用 FLAML 模型）
        # 預測
        df_feat   = pd.DataFrame(valid_feats)
        X_scaled  = scaler.transform(df_feat)
        
        gender_probs = automl_gender.predict_proba(X_scaled)[:, 1]
        hand_probs   = automl_hand.predict_proba(X_scaled)[:, 1]
        play_probs   = automl_play.predict_proba(X_scaled)
        level_probs  = automl_level.predict_proba(X_scaled)
        
        # top-k mean：提高信心片段的權重
        def top_k_mean(probs, k=3):
            scores = np.abs(probs - 0.5)
            topk_idx = np.argsort(scores)[-k:]
            return probs[topk_idx].mean()
        
        # 二分類 => 拿出正類機率
        gender_p = top_k_mean(gender_probs)
        hand_p   = top_k_mean(hand_probs)
        
        # 多分類 => 計算 top-k 的平均機率分布
        play_p  = play_probs[np.argsort(np.abs(play_probs - 1/3).max(axis=1))[-3:]].mean(axis=0)
        level_p = level_probs[np.argsort(np.abs(level_probs - 0.25).max(axis=1))[-3:]].mean(axis=0)


        all_results.append({
            "unique_id": uid,
            "gender": round(gender_p, 4),
            "hold racket handed": round(hand_p, 4),
            "play years_0": round(play_p[0], 4),
            "play years_1": round(play_p[1], 4),
            "play years_2": round(play_p[2], 4),
            "level_2": round(level_p[0], 4),
            "level_3": round(level_p[1], 4),
            "level_4": round(level_p[2], 4),
            "level_5": round(level_p[3], 4)
        })
    except Exception as e:
        print(f"❌ 無法處理 {uid}: {e}")
        continue

# ✅ 額外印出 fallback 使用比例
print(f"\n⚠️ fallback 使用次數：{fallback_count} / {total_count} 筆 ({(fallback_count/total_count)*100:.2f}%)")

# 後處理與輸出檔案照原本流程不變
sub = pd.DataFrame(all_results)
sub = sub[["unique_id", "gender", "hold racket handed",
           "play years_0", "play years_1", "play years_2",
           "level_2", "level_3", "level_4", "level_5"]]

if os.path.exists(sample_path):
    sample = pd.read_csv(sample_path)
    missing_ids = set(sample["unique_id"]) - set(sub["unique_id"])
    extra_ids = set(sub["unique_id"]) - set(sample["unique_id"])
    print(f"檢查 ID 數量：預期 {len(sample)}，目前產出 {len(sub)}")
    if missing_ids:
        print(f"❌ 缺少 ID: {sorted(list(missing_ids))[:5]} ... 共 {len(missing_ids)} 筆")
    if extra_ids:
        print(f"⚠️ 多出 ID: {sorted(list(extra_ids))[:5]} ... 共 {len(extra_ids)} 筆")
    if not missing_ids and not extra_ids:
        print("✅ 所有 ID 與 sample_submission.csv 一致！")
else:
    print(f"⚠️ 找不到 sample_submission.csv：{sample_path}")

sub.to_csv("sample_submission_predtry3.csv", index=False)
print("✅ 結果已輸出至 sample_submission_predtry3.csv")
