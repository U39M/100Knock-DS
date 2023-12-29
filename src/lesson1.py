import pandas as pd
import os
import matplotlib.pyplot as plt

os.chdir('../csv/Part1/')

'''
ノック1 : データの読み込み
'''
# 顧客データ（名前、性別等）が入ったcsvの読み込み
customer_master = pd.read_csv('customer_master.csv')
print(customer_master.head())
print()

# 商品データが入ったcsvの読み込み
item_master = pd.read_csv('item_master.csv')
print(item_master.head())
print()

# 購入明細データ1が入ったcsvの読み込み
transaction_1 = pd.read_csv('transaction_1.csv')
print(transaction_1.head())
print()

# 購入明細の詳細データ1が入ったcsvの読み込み
transaction_detail_1 = pd.read_csv('transaction_detail_1.csv')
print(transaction_detail_1.head())
print()

'''
ノック2 : データをユニオン（縦に結合）する
'''
# 購入明細データ2が入ったcsvの読み込み
transaction_2 = pd.read_csv('transaction_2.csv')

# 購入明細データをユニオンする
transaction = pd.concat([transaction_1, transaction_2], ignore_index=True)

# 結合後のデータの確認
print(transaction)
print("transaction_1 = " + str(len(transaction_1)))
print("transaction_2 = " + str(len(transaction_2)))
print("transaction = " + str(len(transaction)))
print()

# 詳細データ2のcsvの読み込み
transaction_detail_2 = pd.read_csv('transaction_detail_2.csv')

# 詳細データのユニオン
transaction_detail = pd.concat([transaction_detail_1, transaction_detail_2], ignore_index=True)
print(transaction_detail.head())
print()

'''
ノック3 : 売上データ同士を結合（ジョイン）する
'''
# 売上データを結合（ジョイン）する
join_data = pd.merge(transaction_detail, transaction[["transaction_id", "payment_date", "customer_id"]], on = "transaction_id", how = "left")
print(join_data.head())
print()

# ジョイン後のデータ確認
print("transaction_detail = " + str(len(transaction_detail)))
print("transaction = " + str(len(transaction)))
print("join_data = " + str(len(join_data)))
print()

'''
ノック4 : マスターデータをジョインする
'''
# customer_master, item_masterのデータを追加
join_data = pd.merge(join_data, customer_master, on = "customer_id", how = "left")
join_data = pd.merge(join_data, item_master, on = "item_id", how = "left")
print(join_data.head())
print()

'''
ノック5 : 必要なデータ列の作成
'''
# データフレーム型の掛け算
join_data["price"] = join_data["item_price"] * join_data["quantity"]
print(join_data[["quantity", "item_price", "price"]].head())
print()

'''
ノック6 : データ検算
'''
# データ加工前のpriceと加工後の計算で作成したpriceの値を比較
print("join_data = " + str(join_data["price"].sum()))
print("transaction = " + str(transaction["price"].sum()))
print(join_data["price"].sum() == transaction["price"].sum())
print()

'''
ノック7 : 各種統計量を把握する
'''
# 欠損値の確認
print(join_data.isnull().sum())
print()

# 各種統計量の確認
print(join_data.describe())
print()

# データの範囲を確認
print(join_data["payment_date"].min())
print(join_data["payment_date"].max())
print()

'''
ノック8 : 月別でデータを集計
'''
# データ型の確認
print(join_data.dtypes)
print()

# payment_dateをobject型 → datetime型に変更"
join_data["payment_date"] = pd.to_datetime(join_data["payment_date"])
join_data["payment_month"] = join_data["payment_date"].dt.strftime("%Y%m")
print(join_data[["payment_date", "payment_month"]].head())
print()

# 月別データの集計
print(join_data.groupby("payment_month").sum()["price"])
print()

'''
ノック9 : 月別、商品別でデータを集計
'''
# 月別、商品別でデータを集計
print(join_data.groupby(["payment_month", "item_name"]).sum()[["price", "quantity"]])
print()

# ピボットテーブルで表示
print(pd.pivot_table(join_data, index = 'item_name', columns = 'payment_month', values = ['price', 'quantity'], aggfunc = 'sum'))
print()

'''
ノック10 : 商品別の売上推移を可視化する
'''
# グラフ用データの作成
graph_data = pd.pivot_table(join_data, index = 'payment_month', columns = 'item_name', values = 'price', aggfunc = 'sum')
print(graph_data.head())
print()

# グラフの可視化
plt.plot(list(graph_data.index), graph_data["PC-A"], label = 'PC-A')
plt.plot(list(graph_data.index), graph_data["PC-B"], label=  'PC-B')
plt.plot(list(graph_data.index), graph_data["PC-C"], label = 'PC-C')
plt.plot(list(graph_data.index), graph_data["PC-D"], label = 'PC-D')
plt.plot(list(graph_data.index), graph_data["PC-E"], label = 'PC-E')
plt.legend()
plt.show()