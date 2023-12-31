import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from sklearn import linear_model
import sklearn.model_selection

os.chdir('../csv/Part2/')

"""
ノック31:データを読み込んで確認する
"""
# use_logとcustomer_joinを読み込む
## 欠損値の確認
uselog = pd.read_csv('use_log.csv')
print(uselog.isnull().sum())
print()

customer = pd.read_csv('customer_join.csv')
print(customer.isnull().sum())
print()

'''
ノック32:クラスタリングで顧客をグループ化
'''
# クラスタリングに必要な変数に絞り込む
## mean, median, max, min, membership_period
customer_clustering = customer[["mean", "median", "max", "min", "membership_period"]]
print(customer_clustering.head())
print()

# K-means法によるクラスタリング
## 標準化を行う(membership_periodの値のみ大きく異なることから、クラスタリングに影響が出てしまうため)
sc = StandardScaler()
customer_clustering_sc = sc.fit_transform(customer_clustering)

## K-meansのモデル構築
kmeans = KMeans(n_clusters = 4, random_state = 0)
clusters = kmeans.fit(customer_clustering_sc)
customer_clustering = customer_clustering.assign(cluster = clusters.labels_)

print(customer_clustering["cluster"].unique())
print(customer_clustering.head())
print()

'''
ノック33:クラスタリング結果を分析する
'''
# データ件数を把握する
customer_clustering.columns = ["月内平均値", "月内中央値", "月内最大値", "月内最小値", "会員期間", "cluster"]
print(customer_clustering.groupby("cluster").count())
print()

# グループ毎の平均値を集計
print(customer_clustering.groupby("cluster").mean())
print()

"""
ノック34:クラスタリング結果を可視化する
"""
# 次元削除を用いて5つの変数を二次元上にプロットする
## 主成分分析で次元削除
X = customer_clustering_sc
pca = PCA(n_components = 2)
pca.fit(X)
x_pca = pca.transform(X)
pca_df = pd.DataFrame(x_pca)
pca_df["cluster"] = customer_clustering["cluster"]

# 散布図のプロット
for i in customer_clustering["cluster"].unique():
    tmp = pca_df.loc[pca_df["cluster"] == i]
    plt.scatter(tmp[0], tmp[1])
plt.show()

''''
ノック35:クラスタリング結果をもとに退会顧客の傾向を把握する
'''
# グループごとの退会/継続顧客の集計
customer_clustering = pd.concat([customer_clustering, customer], axis = 1)
print(customer_clustering.groupby(["cluster", "is_deleted"], as_index = False).count()[["cluster", "is_deleted", "customer_id"]])
print()

# グループ/定期利用flg毎の集計
print(customer_clustering.groupby(["cluster", "routine_flg"], as_index = False).count()[["cluster", "routine_flg", "customer_id"]])
print()

"""
ノック36:翌月の利用回数予測を行うための準備
"""
# 2018/11の利用回数を予測(当月:2018/10)
## 2018/05~2018/10の6か月の利用データと2018/11の顧客データを教師データとして利用
## uselogデータを用いて、顧客ごとに集計
uselog["usedate"] = pd.to_datetime(uselog["usedate"])
uselog["年月"] = uselog["usedate"].dt.strftime("%Y%m")
uselog_months = uselog.groupby(["年月", "customer_id"], as_index = False).count()
uselog_months.rename(columns = {"log_id":"count"}, inplace = True)
del uselog_months["usedate"]
print(uselog_months.head())
print()

## uselogデータを用いて、年月ごとに集計
year_months = list(uselog_months["年月"].unique())
predict_data = pd.DataFrame()
for i in range(6, len(year_months)):
    tmp = uselog_months.loc[uselog_months["年月"] == year_months[i]].copy()
    tmp.rename(columns = {"count":"count_pred"}, inplace = True)
    for j in range(1, 7):
        tmp_before = uselog_months.loc[uselog_months["年月"] == year_months[i-j]].copy()
        del tmp_before["年月"]
        tmp_before.rename(columns = {"count":"count_{}".format(j-1)}, inplace = True)
        tmp = pd.merge(tmp, tmp_before, on = "customer_id", how = "left")
    predict_data = pd.concat([predict_data, tmp], ignore_index = True)
print(predict_data.head())
print()

## 欠損値の除去
predict_data = predict_data.dropna()
predict_data = predict_data.reset_index(drop = True)
print(predict_data.head())
print()

'''
ノック37:特徴となる変数を付与する
'''
# customerのstart_date列をpredict_dataに結合
predict_data = pd.merge(predict_data, customer[["customer_id", "start_date"]], on = "customer_id", how = "left")
print(predict_data.head())
print()

# 会員期間を月単位で作成
predict_data["now_date"] = pd.to_datetime(predict_data["年月"], format = "%Y%m")
predict_data["start_date"] = pd.to_datetime(predict_data["start_date"])
predict_data["period"] = None
for i in range(len(predict_data)):
    delta = relativedelta(predict_data.loc[i, "now_date"], predict_data.loc[i, "start_date"])
    predict_data.loc[i, "period"] = delta.years * 12 + delta.months
print(predict_data.head())
print()

'''
ノック38:来月の利用回数予測モデルを作成
'''
# 2018/04以降に新規に入った顧客に絞ってモデル作成を行う
## 線形回帰モデル
predict_data = predict_data.loc[predict_data["start_date"] >= pd.to_datetime("20180401")]
model = linear_model.LinearRegression()
X = predict_data[["count_0", "count_1","count_2","count_3","count_4","count_5","period"]]
y = predict_data["count_pred"]
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state = 0)
model.fit(X_train, y_train)
print("学習用データ, 評価用データの比較")
print(model.score(X_train, y_train))
print(model.score(X_test, y_test))
print()

"""
ノック39:モデルに寄与している変数を確認する
"""
# 説明変数ごとに、寄与している変数の係数を出力する
coef = pd.DataFrame({"feature_names":X.columns, "coefficient":model.coef_})
print(coef)
print()

"""
ノック40:来月の利用回数を予測する
"""
# 2人の顧客の利用データを作成
x1 = [3, 4, 4, 6, 8, 7, 8]
x2 = [2, 2, 3, 3, 4, 6, 8]
x_pred = pd.DataFrame(data = [x1, x2], columns = ["count_0", "count_1","count_2","count_3","count_4","count_5","period"])
print(model.predict(x_pred))

# uselog_monthsのデータをcsv出力
os.chdir("../../out/")
uselog_months.to_csv("use_log_months.csv", index = False)