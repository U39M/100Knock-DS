import os
import pandas as pd
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
os.chdir('../csv/Part2/')

'''
ノック21:データを読み込んで把握する
'''
# 4種類のデータを読み込む
## ジムの利用履歴データ(2018/04 - 2019/03)
uselog = pd.read_csv('use_log.csv')
print(len(uselog))
print(uselog.head())
print()

## 会員データ(2019/03末時点での会員データ)
customer = pd.read_csv('customer_master.csv')
print(len(customer))
print(customer.head())
print()

## 会員区分データ(オールタイム、デイタイム、ナイト)
class_master = pd.read_csv('class_master.csv')
print(len(class_master))
print(class_master.head())
print()

## キャンペーン区分データ(通常、入会費半額、入会費無料)
campaign_master = pd.read_csv('campaign_master.csv')
print(len(campaign_master))
print(campaign_master.head())
print()

'''
ノック22:顧客データを整形する
'''
# customerにclass_masterとcampaign_masterを結合する
customer_join = pd.merge(customer, class_master, on = "class", how = "left")
customer_join = pd.merge(customer_join, campaign_master, on = "campaign_id", how = "left")
print(customer_join.head())
print("customer_length:" + str(len(customer)))
print("join_length:" + str(len(customer_join)))
print()

# 欠損値の確認
print(customer_join.isnull().sum())
print()

'''
ノック23:顧客データの基礎集計をする
'''
# 会員区分別に集計
print(customer_join.groupby("class_name").count()["customer_id"])
print()

# キャンペーン区分別で集計
print(customer_join.groupby("campaign_name").count()["customer_id"])
print()

# 性別で集計
print(customer_join.groupby("gender").count()["customer_id"])
print()

# 退会済みかどうか毎に集計
print(customer_join.groupby("is_deleted").count()["customer_id"])
print()

# start_date(入会日のデータ)をdatetime型に変換
customer_join["start_date"] = pd.to_datetime(customer_join["start_date"])

# customer_startに該当ユーザーのデータを格納
customer_start = customer_join.loc[customer_join["start_date"]>pd.to_datetime("20180401")]
print(len(customer_start))
print()

"""
ノック24:最新顧客データの基礎集計をする
"""
# 最新月のデータを調査
## 最終日(2019年3月31日)に退会したユーザー もしくは、最新月に在籍しているユーザーに絞り込む
customer_join["end_date"] = pd.to_datetime(customer_join["end_date"])
customer_newer = customer_join.loc[(customer_join["end_date"] >= pd.to_datetime("20190331")) | (customer_join["end_date"].isna())]
print(len(customer_newer))
print(customer_newer["end_date"].unique())
print()

# 会員区分毎に全体の数を把握する
print(customer_newer.groupby("class_name").count()["customer_id"])
print()

# キャンペーン区分毎に全体の数を把握する
print(customer_newer.groupby("campaign_name").count()["customer_id"])
print()

# 性別毎に全体の数を把握する
print(customer_newer.groupby("gender").count()["customer_id"])
print()

"""
ノック25:利用履歴データを集計する
"""
# 顧客ごとの月利用回数を集計したデータを作成
uselog["usedate"] = pd.to_datetime(uselog["usedate"])
uselog["年月"] = uselog["usedate"].dt.strftime("%Y%m")
uselog_months = uselog.groupby(["年月", "customer_id"], as_index = False).count()
uselog_months.rename(columns = {"log_id":"count"}, inplace = True)
del uselog_months["usedate"]
print(uselog_months.head())
print()

# 月内の利用回数の平均値, 中央値, 最大値, 最小値の集計
uselog_customer = uselog_months.groupby("customer_id").agg(["mean", "median", "max", "min"])["count"]
uselog_customer = uselog_customer.reset_index(drop = False)
print(uselog_customer.head())
print()

'''
ノック26:利用履歴データから定期利用フラグを作成する
'''
# 顧客毎に月/曜日別に集計し、最大値が4以上の曜日が1ヶ月でもあったユーザーをフラグ1とする
## 顧客毎に月/曜日別に集計する（月曜:0 ~ 日曜:6)
uselog["weekday"] = uselog["usedate"].dt.weekday
uselog_weekday = uselog.groupby(["customer_id", "年月", "weekday"], as_index = False).count()[["customer_id", "年月", "weekday", "log_id"]]
uselog_weekday.rename(columns = {"log_id":"count"}, inplace = True)
print(uselog_weekday.head())
print()

## 顧客毎の各月の最大値を取得、4以上の場合はフラグを立てる
uselog_weekday = uselog_weekday.groupby("customer_id", as_index = False).max()[["customer_id", "count"]]
uselog_weekday["routine_flg"] = 0
uselog_weekday["routine_flg"] = uselog_weekday["routine_flg"].where(uselog_weekday["count"] < 4, 1)
print(uselog_weekday.head())
print()

"""
ノック27:顧客データと利用履歴データを結合しよう
"""
# uselog_customer, uselog_weekdayをcustomer_joinと結合する
customer_join = pd.merge(customer_join, uselog_customer, on = "customer_id", how = "left")
customer_join = pd.merge(customer_join, uselog_weekday[["customer_id", "routine_flg"]], on = "customer_id", how = "left")
print(customer_join.head())
print()

# 欠損値の確認
print(customer_join.isnull().sum())
print()

"""
ノック28:会員期間を計算する
"""
# 会員期間を表した列を追加(2019/03までに退会していないユーザーは欠損値となるため、2019/04/30に置換する)
customer_join["calc_date"] = customer_join["end_date"]
customer_join["calc_date"] = customer_join["calc_date"].fillna(pd.to_datetime("20190430"))
customer_join["membership_period"] = 0
for i in range(len(customer_join)):
    delta = relativedelta(customer_join["calc_date"].iloc[i], customer_join["start_date"].iloc[i])
    customer_join.loc[i, "membership_period"] = delta.years*12 + delta.months
print(customer_join.head())
print()

"""
ノック29:顧客行動の各種統計量を把握する
"""
# flg毎に顧客数を集計する(列のmean:顧客の月内平均利用回数, 行のmean:顧客の月内平均利用回数の平均)
print(customer_join[["mean", "median", "max", "min"]].describe())
print()

# フラグの集計
print(customer_join.groupby("routine_flg").count()["customer_id"])
print()

# 会員期間の分布を確認する
plt.hist(customer_join["membership_period"])
plt.show()

"""
ノック30:退会ユーザーと継続ユーザーの違いを把握する
"""
# 退会ユーザーと継続ユーザーを分けて比較する
customer_end = customer_join.loc[customer_join["is_deleted"] == 1]
customer_stay = customer_join.loc[customer_join["is_deleted"] == 0]
print("退会ユーザー")
print(customer_end.describe())
print()
print("継続ユーザー")
print(customer_stay.describe())
print()

# customer_joinをcsv出力
os.chdir("../../out/")
customer_join.to_csv("customer_join.csv", index = False)