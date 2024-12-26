import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# 数据读取
print("加载训练数据...")
train_df = pd.read_csv('train_set.csv', sep='\t', nrows=15000)
print(f"数据加载完成，包含 {train_df.shape[0]} 行和 {train_df.shape[1]} 列。")

# 数据分析：句子长度分布
train_df['text_len'] = train_df['text'].apply(lambda x: len(x.split(' ')))
print("\n句子长度统计信息：")
print(train_df['text_len'].describe())

plt.figure(figsize=(10, 5))
plt.hist(train_df['text_len'], bins=50, color='skyblue')
plt.title('Sentence Length Distribution')
plt.xlabel('Sentence Length')
plt.ylabel('Frequency')
plt.show()

# 数据分析：类别分布
print("\n类别分布：")
print(train_df['label'].value_counts())

plt.figure(figsize=(10, 5))
train_df['label'].value_counts().plot(kind='bar', color='orange')
plt.title('Label Distribution')
plt.xlabel('Label')
plt.ylabel('Frequency')
plt.show()

# TF-IDF 特征提取
print("\n开始进行 TF-IDF 特征提取...")
tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=5000)
X = tfidf.fit_transform(train_df['text'])
y = train_df['label']
print(f"TF-IDF 特征提取完成，特征维度为：{X.shape[1]}")

# 划分训练集和验证集
print("\n划分训练集和验证集...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"训练集样本数：{X_train.shape[0]}，验证集样本数：{X_val.shape[0]}")

# 模型训练
print("\n训练 RidgeClassifier 模型...")
clf = RidgeClassifier()
clf.fit(X_train, y_train)
print("模型训练完成！")

# 验证集评估
print("\n评估模型...")
val_pred = clf.predict(X_val)
f1 = f1_score(y_val, val_pred, average='macro')
print(f"验证集 F1 Score: {f1:.4f}")

# 测试集预测
print("\n加载测试数据...")
test_df = pd.read_csv('test_a.csv', sep='\t', nrows=50000)
test = tfidf.transform(test_df['text'])
print("测试数据加载完成！")

print("\n进行预测...")
test_pred = clf.predict(test)

# 预测结果保存
predictions_df = pd.DataFrame(test_pred, columns=['label'])
predictions_df.to_csv('test.csv', index=False)
print("预测结果已保存至 test.csv 文件。")
