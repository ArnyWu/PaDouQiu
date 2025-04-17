import os
import zipfile
import math
import numpy as np
import pandas as pd
from scipy.stats import entropy, kurtosis, skew
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# 存檔記得要改檔名不然會被覆蓋掉(在最下面)
# ==== 資料解壓與讀檔（請依照實際路徑修改）====
train_zip = r"C:\Users\User\Downloads\39_Training_Dataset.zip"
test_zip = r"C:\Users\User\Downloads\39_Test_Dataset.zip"
sample_path = r"C:\Users\User\Downloads\sample_submission.csv"  #更改成他提供的範例路徑(用來比對欄位與筆數是否相同)
extract_root = "aicup_data"

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
    以手動 FFT 計算信號的頻域特徵
    參數:
      signal: 一維數據，代表單一軸的數值序列
    回傳:
      字典形式的頻域特徵，包括光譜熵、峰度、偏度與均值
    """
    L = len(signal)
    # 補零至 2 的冪次
    n_fft = 1
    while n_fft < L:
        n_fft *= 2
    sig_list = list(signal)
    sig_list += [0] * (n_fft - L)
    xreal = sig_list.copy()
    ximag = [0] * n_fft
    n, fft_real, fft_imag = FFT(xreal, ximag)
    
    # 計算功率譜密度（PSD）
    psd = [fft_real[i]**2 + fft_imag[i]**2 for i in range(n)]
    total_power = sum(psd)
    if total_power > 0:
        psd_norm = [p / total_power for p in psd]
    else:
        psd_norm = [0] * len(psd)
    fft_entropy_val = entropy(psd_norm) if total_power > 0 else 0
    fft_kurtosis_val = kurtosis(psd)
    fft_skewness_val = skew(psd)
    fft_power_val = np.mean(psd)
    
    return {
        'fft_entropy': fft_entropy_val,
        'fft_kurtosis': fft_kurtosis_val,
        'fft_skewness': fft_skewness_val,
        'fft_power': fft_power_val
    }

def extract_segment_features(seg):
    """
    針對單一段 (seg) 計算所有軸的時間域統計特徵與使用手動 FFT 取得的頻域特徵
    """
    features = {}
    axis_names = ["Ax", "Ay", "Az", "Gx", "Gy", "Gz"]
    for i, name in enumerate(axis_names):
        axis_data = seg[:, i]
        features[f"{name}_mean"] = np.mean(axis_data)
        features[f"{name}_std"] = np.std(axis_data)
        features[f"{name}_rms"] = np.sqrt(np.mean(axis_data**2))
        features[f"{name}_min"] = np.min(axis_data)
        features[f"{name}_max"] = np.max(axis_data)
        fft_feat = extract_fft_features(axis_data)
        for k, v in fft_feat.items():
            features[f"{name}_{k}"] = v
    # 額外計算三軸加速度與陀螺儀向量
    acc = np.linalg.norm(seg[:, :3], axis=1)
    gyro = np.linalg.norm(seg[:, 3:], axis=1)
    features["acc_mean"] = np.mean(acc)
    features["gyro_mean"] = np.mean(gyro)
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
        plt.figure(figsize=(8, 4))
        sns.histplot(data, bins=40, kde=True)
        plt.title(f"{name} Distribution")
        plt.xlabel(name)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        mean_val = np.mean(data)
        std_val = np.std(data)
        q25, q75 = np.percentile(data, [25, 75])
        print(f"\n📊 {name} Summary:")
        print(f"  Mean: {mean_val:.3f}, Std: {std_val:.3f}, Q1: {q25:.2f}, Q3: {q75:.2f}, IQR: {q75 - q25:.2f}")
        lower = q25 - 1.5 * (q75 - q25)
        upper = q75 + 1.5 * (q75 - q25)
        print(f"  建議篩選區間: [{lower:.1f}, {upper:.1f}]\n")

    plot_and_summary(lengths, "Segment Length")
    plot_and_summary(rms_list, "RMS Acceleration")
    plot_and_summary(p2p_list, "Peak-to-Peak Acceleration")

def is_valid_segment(seg):
    length = len(seg)
    if length < 57 or length > 129:
        return False
    acc = np.linalg.norm(seg[:, :3], axis=1)
    rms = np.sqrt(np.mean(acc**2))
    p2p = np.ptp(acc)
    if rms < 2500 or p2p < 3000:
        return False
    return True

def create_feature_dataframe(info_df, data_dir, with_labels=True):
    rows = []
    for _, row in info_df.iterrows():
        uid = row["unique_id"]
        txt_path = os.path.join(data_dir, f"{uid}.txt")
        try:
            raw_data = np.loadtxt(txt_path)
            cut_points = [int(x) for x in row["cut_point"].strip("[]").split()]
            cut_points = [0] + cut_points + [len(raw_data)]
            segments = [raw_data[cut_points[i]:cut_points[i+1]] for i in range(min(27, len(cut_points)-1))]
            for seg in segments:
                if not is_valid_segment(seg):
                    continue
                feat = extract_segment_features(seg)
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
# 模型訓練與驗證
# =============================================================================
# 建立訓練資料與驗證集
train_df = create_feature_dataframe(train_info, os.path.join(train_inner, "train_data"), with_labels=True)
train_ids, val_ids = train_test_split(train_df["unique_id"].unique(), test_size=0.2, random_state=42)
train_part = train_df[train_df["unique_id"].isin(train_ids)]
val_part   = train_df[train_df["unique_id"].isin(val_ids)]

# 標準化（移除標籤欄位）
X_train = train_part.drop(columns=["unique_id", "gender", "hand", "play", "level"])
X_val   = val_part.drop(columns=["unique_id", "gender", "hand", "play", "level"])
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)

# 模型訓練（以 LogisticRegression 為例）
gender_model = LogisticRegression(solver="lbfgs", max_iter=1000)
hand_model   = LogisticRegression(solver="lbfgs", max_iter=1000)
play_model   = LogisticRegression(multi_class="multinomial", solver="saga", max_iter=1000)
level_model  = LogisticRegression(multi_class="multinomial", solver="saga", max_iter=1000)

gender_model.fit(X_train_scaled, train_part["gender"])
hand_model.fit(X_train_scaled, train_part["hand"])
play_model.fit(X_train_scaled, train_part["play"])
level_model.fit(X_train_scaled, train_part["level"])

def evaluate_on_val_set(val_df, gender_model, hand_model, play_model, level_model, scaler):
    print("\n🔍 評估驗證集...")
    grouped = val_df.groupby("unique_id")
    preds = []
    truths = []
    for uid, group in grouped:
        X = group.drop(columns=["unique_id", "gender", "hand", "play", "level"])
        y = group.iloc[0]  # 同一 uid 標籤相同，只取第一筆即可
        X_scaled = scaler.transform(X)
        gender_p = gender_model.predict_proba(X_scaled)[:, 1].mean()
        hand_p   = hand_model.predict_proba(X_scaled)[:, 1].mean()
        play_p   = play_model.predict_proba(X_scaled).mean(axis=0)
        level_p  = level_model.predict_proba(X_scaled).mean(axis=0)
        preds.append({
            "unique_id": uid,
            "gender": gender_p,
            "hand": hand_p,
            "play_0": play_p[0], "play_1": play_p[1], "play_2": play_p[2],
            "level_2": level_p[0], "level_3": level_p[1],
            "level_4": level_p[2], "level_5": level_p[3],
        })
        truths.append({
            "gender": y["gender"],
            "hand": y["hand"],
            "play": y["play"],
            "level": y["level"]
        })
    pred_df = pd.DataFrame(preds)
    truth_df = pd.DataFrame(truths)
    auc_gender = roc_auc_score(truth_df["gender"], pred_df["gender"])
    auc_hand   = roc_auc_score(truth_df["hand"], pred_df["hand"])
    y_play = pd.get_dummies(truth_df["play"])
    y_level = pd.get_dummies(truth_df["level"])
    auc_play  = roc_auc_score(y_play, pred_df[["play_0", "play_1", "play_2"]], multi_class='ovr', average="micro")
    auc_level = roc_auc_score(y_level, pred_df[["level_2", "level_3", "level_4", "level_5"]], multi_class='ovr', average="micro")
    final_score = 0.25 * (auc_gender + auc_hand + auc_play + auc_level)
    print("\n🎯 驗證集 AUC:")
    print(f"Gender : {auc_gender:.4f}")
    print(f"Hand   : {auc_hand:.4f}")
    print(f"Play   : {auc_play:.4f}")
    print(f"Level  : {auc_level:.4f}")
    print(f"Final Score: {final_score:.4f}\n")
    return final_score

evaluate_on_val_set(val_part, gender_model, hand_model, play_model, level_model, scaler)


# =============================================================================
# Test set 預測並輸出結果
# =============================================================================
print("開始 Test set 預測...")
all_results = []
for uid in test_info["unique_id"]:
    txt_path = os.path.join(test_inner, "test_data", f"{uid}.txt")
    try:
        raw_data = np.loadtxt(txt_path)
        row = test_info[test_info["unique_id"] == uid].iloc[0]
        cut_points = [int(x) for x in row["cut_point"].strip("[]").split()]
        cut_points = [0] + cut_points + [len(raw_data)]
        segments = [raw_data[cut_points[i]:cut_points[i+1]] for i in range(min(27, len(cut_points)-1))]
        valid_segments = [extract_segment_features(seg) for seg in segments if is_valid_segment(seg)]
        # 若無 segment 符合條件，取前 100 筆做 fallback
        if len(valid_segments) == 0:
            fallback_seg = raw_data[:100] if len(raw_data) >= 100 else raw_data
            valid_segments = [extract_segment_features(fallback_seg)]
        df_feat = pd.DataFrame(valid_segments)
        X_scaled = scaler.transform(df_feat)
        gender_prob = gender_model.predict_proba(X_scaled)[:, 1].mean()
        hand_prob   = hand_model.predict_proba(X_scaled)[:, 1].mean()
        play_prob   = play_model.predict_proba(X_scaled).mean(axis=0)
        level_prob  = level_model.predict_proba(X_scaled).mean(axis=0)
        all_results.append({
            "unique_id": uid,
            "gender": round(gender_prob, 4),
            "hold racket handed": round(hand_prob, 4),
            "play years_0": round(play_prob[0], 4),
            "play years_1": round(play_prob[1], 4),
            "play years_2": round(play_prob[2], 4),
            "level_2": round(level_prob[0], 4),
            "level_3": round(level_prob[1], 4),
            "level_4": round(level_prob[2], 4),
            "level_5": round(level_prob[3], 4)
        })
    except Exception as e:
        print(f"❌ 無法處理 {uid}: {e}")
        continue

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

sub.to_csv("sample_submission_predtry.csv", index=False)
print("✅ 結果已輸出至 sample_submission_predtry.csv")
