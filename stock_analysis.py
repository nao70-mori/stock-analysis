import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


mac_path = '/Users/naoya/Documents/python/自主制作'

# Docker 側のフォルダパス
docker_path = '/app'

# 実際に存在する方を使う
if os.path.exists(mac_path):
    folder_path = mac_path
else:
    folder_path = docker_path

# Docker 内で実行




csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

# --- 全ファイルをまとめて読み込み ---
dfs = []
for file in csv_files:
    df = pd.read_csv(file)
    if df.empty or len(df) < 2:
        continue
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    dfs.append(df)

# --- すべて結合 ---
df = pd.concat(dfs, ignore_index=True)

min_value = df.min()
max_value = df.max()

print(min_value)
print(max_value)

# 翌日の終値との差を計算
df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

# 最終行を削除 & インデックスを振り直す
# Target だけを dropna する
df = df.dropna(subset=["Target"]).reset_index(drop=True)

# 5日移動平均
df["MA5"] = df["Close"].rolling(window=5).mean()

# 25日移動平均
df["MA25"] = df["Close"].rolling(window=25).mean()

#標準偏差
df["Volatility_5"] = df["Close"].rolling(window=5).std()

# 曜日特徴量（0=月曜, 4=金曜）
df["DayOfWeek"] = df.index % 5  # 日付データがあれば dt.dayofweek を使える

# 特徴量と目的変数
features = ["Open", "High", "Low", "Close", "Volume", "MA5", "MA25", "Volatility_5", "DayOfWeek"]
X = df[features]
y = df["Target"]
X = X.fillna(X.mean())






# 時系列を保ったまま分割
train_size = int(len(df) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

models = {
    "RandomForest": RandomForestClassifier(random_state=15),
    "DecisionTree": DecisionTreeClassifier(random_state=15),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=15)
}


for name, model in models.items():
    print(f"\n===== {name} =====")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    true_counts = Counter(y_test)
    pred_counts = Counter(y_pred)
    print("正解率:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.show()

    latest_data = X.iloc[[-1]]  # 最新の行
    prediction = model.predict(latest_data)[0]
    print("明日は上がる" if prediction == 1 else "明日は下がる")

    print("F1スコア:", f1_score(y_test, y_pred))

    y_prob = model.predict_proba(X_test)[:, 1]  # 上がる確率だけ
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))