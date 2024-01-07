import os
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.tree import DecisionTreeClassifier
import sklearn.model_selection
from sklearn import tree
import matplotlib.pyplot as plt
import japanize_matplotlib

os.chdir("../csv/Part2/")

"""
ノック41:データを読み込んで利用データを整形する
"""
print("Knock 41")
print("==========================================================================")
# csvの読み込み
customer = pd.read_csv('customer_join.csv')
uselog_months = pd.read_csv('use_log_months.csv')

# 機械学習用に利用データを加工
year_months = list(uselog_months["年月"].unique())
uselog = pd.DataFrame()
for i in range(1, len(year_months)):
    tmp = uselog_months.loc[uselog_months["年月"] == year_months[i]].copy()
    tmp.rename(columns = {"count":"count_0"}, inplace = True)
    tmp_before = uselog_months.loc[uselog_months["年月"] == year_months[i-1]].copy()
    del tmp_before["年月"]
    tmp_before.rename(columns = {"count":"count_1"}, inplace = True)
    tmp = pd.merge(tmp, tmp_before, on = "customer_id", how = "left")
    uselog = pd.concat([uselog, tmp], ignore_index = True)
print(uselog.head())

print("==========================================================================")

"""
ノック42:退会前月の退会顧客データを作成する
"""
print("Knock 42")
print("==========================================================================")

# end_date列の1ヵ月前の年月を取得
# uselogとcustomer_id, 年月をキーにして結合
exit_customer = customer.loc[customer["is_deleted"] == 1].copy()
exit_customer["exit_date"] = None
exit_customer["end_date"] = pd.to_datetime(exit_customer["end_date"])
for i in exit_customer.index:
    exit_customer.loc[i, "exit_date"] = exit_customer.loc[i, "end_date"] - relativedelta(months = 1)
exit_customer["exit_date"] = pd.to_datetime(exit_customer["exit_date"])
exit_customer["年月"] = exit_customer["exit_date"].dt.strftime("%Y%m")
uselog["年月"] = uselog["年月"].astype(str)
exit_uselog = pd.merge(uselog, exit_customer, on = ["customer_id", "年月"], how = "left")
print(len(uselog))
print(exit_uselog.head())

# 欠損値の除去
exit_uselog = exit_uselog.dropna(subset = ["name"])
print(len(exit_uselog))
print(len(exit_uselog["customer_id"].unique()))
print(exit_uselog.head())

print("==========================================================================")

'''
ノック43:継続顧客のデータを作成する
'''
print("Knock 43")
print("==========================================================================")

# uselogデータに継続顧客データを結合する
conti_customer = customer.loc[customer["is_deleted"] == 0]
conti_uselog = pd.merge(uselog, conti_customer, on = ["customer_id"], how = "left")
print(len(conti_uselog))
conti_uselog = conti_uselog.dropna(subset = ["name"])
print(len(conti_uselog))

# 継続顧客データをシャッフル、重複を削除
conti_uselog = conti_uselog.sample(frac = 1, random_state = 0).reset_index(drop = True)
conti_uselog = conti_uselog.drop_duplicates(subset = "customer_id")
print(len(conti_uselog))
print(conti_uselog.head())

# 継続顧客データと、退会顧客データを縦に結合
predict_data = pd.concat([conti_uselog, exit_uselog], ignore_index = True)
print(len(predict_data))
print(predict_data.head())

print("==========================================================================")

'''
ノック44:予測する月の在籍期間を作成する
'''
print("Knock 44")
print("==========================================================================")

# 在籍期間の列を追加する
predict_data["period"] = 0
predict_data["now_date"] = pd.to_datetime(predict_data["年月"], format = "%Y%m")
predict_data["start_date"] = pd.to_datetime(predict_data["start_date"])
for i in range(len(predict_data)):
    delta = relativedelta(predict_data.loc[i, "now_date"], predict_data.loc[i, "start_date"])
    predict_data.loc[i, "period"] = int(delta.years * 12 + delta.months)
print(predict_data.head())

print("==========================================================================")

'''
ノック45:欠損値を除去
'''
print("Knock 45")
print("==========================================================================")

# 欠損値の数を把握
print(predict_data.isna().sum())
print()

# 欠損値の除外
predict_data = predict_data.dropna(subset = ["count_1"])
print(predict_data.isna().sum())
print()
print("==========================================================================")

'''
ノック46:文字列型の変数を処理できるように整形する
'''
print("Knock 46")
print("==========================================================================")
# 予測に用いるデータに絞り込む
## 説明変数: (カテゴリー変数:campaign_name, class_name, gender), routine_flg, period
## 目的変数: is_deleted
target_col = ["campaign_name", "class_name", "gender", "count_1", "routine_flg", "period", "is_deleted"]
predict_data = predict_data[target_col]
print(predict_data.head())
print()

# ダミー変数の作成
predict_data = pd.get_dummies(predict_data)
print(predict_data.head())

# campaign_name_通常, class_name_ナイト, gender_M列を削除
del predict_data["campaign_name_通常"]
del predict_data["class_name_ナイト"]
del predict_data["gender_M"]
print(predict_data.head())
print("==========================================================================")

'''
ノック47:決定木を用いて退会予測モデルを作成してみよう
'''
print("Knock 47")
print("==========================================================================")

# 退会、継続データの件数を揃える(各1104件, 50:50)
exit = predict_data.loc[predict_data["is_deleted"] == 1]
conti = predict_data.loc[predict_data["is_deleted"] == 0].sample(len(exit), random_state = 0)

# データの結合
X = pd.concat([exit, conti], ignore_index = True)

# is_deleted列を目的変数y, 残りのデータを説明変数Xにする
y = X["is_deleted"]
del X["is_deleted"]

# 学習用データと評価用データの分割
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state = 0)

# モデル定義
model = DecisionTreeClassifier(random_state = 0)

# 学習用データを指定
model.fit(X_train, y_train)

# 評価データの予測（1:退会、0:継続）
y_test_pred = model.predict(X_test)
print(y_test_pred)
print()

# 正解との比較(一致していれば正解)
results_test = pd.DataFrame({"y_test":y_test, "y_pred":y_test_pred})
print(results_test.head())
print("==========================================================================")

'''
ノック48:決定木を用いて退会予測モデルを作成してみよう
'''
print("Knock 48")
print("==========================================================================")

# 正解率を出力
correct = len(results_test.loc[results_test["y_test"] == results_test["y_pred"]])
data_count = len(results_test)
score_test = correct / data_count
print(score_test)
print()

# 評価用データ, 学習用データの精度を確認
print("評価用")
print(model.score(X_test, y_test))
print("学習用")
print(model.score(X_train, y_train))
print()

# モデルのパラメータを変更して、学習用データの過学習傾向を解決する
## 木構造の深さを浅くする
X = pd.concat([exit, conti], ignore_index = True)
y = X["is_deleted"]
del X["is_deleted"]
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state = 0)

model = DecisionTreeClassifier(random_state = 0, max_depth = 5)
model.fit(X_train, y_train)
print("評価用")
print(model.score(X_test, y_test))
print("学習用")
print(model.score(X_train, y_train))
print()
print("==========================================================================")

'''
ノック49:モデルに寄与している変数を確認する
'''
print("Knock 49")
print("==========================================================================")
# データフレームに格納
importance = pd.DataFrame({"feature_names":X.columns, "coefficient":model.feature_importances_})
print(importance)

# 木構造の可視化
plt.figure(figsize = (20, 8))
tree.plot_tree(model, feature_names = X.columns, fontsize = 8)
plt.show()
print("==========================================================================")

'''
ノック50:顧客の退会を予測する
'''
print("Knock 50")
print("==========================================================================")

# 説明変数の定義
## 1か月前の利用回数、定期利用者、在籍期間、キャンペーン区間、会員区分、性別
count_1 = 3
routine_flg = 1
period = 10
campaign_name = "入会費無料"
class_name = "オールタイム"
gender = "M"

# データ加工
if campaign_name == "入会費半額":
    campaign_name_list = [1, 0]
elif campaign_name == "入会費無料":
    campaign_name_list = [0, 1]
elif campaign_name == "通常":
    campaign_name_list = [0, 0]

if class_name == "オールタイム":
    class_name_list = [1, 0]
elif class_name == "デイタイム":
    class_name_list = [0, 1]
elif class_name == "ナイト":
    class_name_list = [0, 0]

if gender == "F":
    gender_list = [1]
elif gender == "M":
    gender_list = [0]

input_data = [count_1, routine_flg, period]
input_data.extend(campaign_name_list)
input_data.extend(class_name_list)
input_data.extend(gender_list)
input_data = pd.DataFrame(data = [input_data], columns = X.columns)

# データをもとに予測する
print(model.predict(input_data))
print(model.predict_proba(input_data))
print("==========================================================================")