import pandas as pd
import os

'''
ノック11 : データを読み取る
'''
# 売上データの読み取り
os.chdir("../csv/Part1/")
uriage_data = pd.read_csv('uriage.csv')
print(uriage_data.head())
print()

# 顧客台帳データの読み取り
os.chdir("../../xlsx")
kokyaku_data = pd.read_excel('kokyaku_daicho.xlsx')
print(kokyaku_data.head())
print()

'''
ノック12 : データの揺れを見る
'''
# 売上履歴のitem_nameからデータの揺れを確認
print(uriage_data["item_name"].head())
print()

# item_priceのデータの揺れを確認
print(uriage_data["item_price"].head())
print()

'''
ノック13 : データに揺れがあるまま集計してみる
'''
# 売上履歴から商品ごとの月売上合計を集計
uriage_data["purchase_date"] = pd.to_datetime(uriage_data["purchase_date"])
uriage_data["purchase_month"] = uriage_data["purchase_date"].dt.strftime("%Y%m")

# データ補正前の集計結果（商品毎）
res = uriage_data.pivot_table(index = "purchase_month", columns = "item_name", aggfunc = "size", fill_value = 0)
print(res)
print()

# データ補正前の集計結果（金額）
res = uriage_data.pivot_table(index = "purchase_month", columns = "item_name", values = "item_price", aggfunc = "sum", fill_value = 0)
print(res)
print()

'''
ノック14 : 商品名の揺れを補正する
'''
# 現状の把握
print(len(pd.unique(uriage_data["item_name"])))
print()

# データの揺れを解消
## 小文字を大文字に変換
uriage_data["item_name"] = uriage_data["item_name"].str.upper()

## 全角・半角スペースを除去
uriage_data["item_name"] = uriage_data["item_name"].str.replace("　", "")
uriage_data["item_name"] = uriage_data["item_name"].str.replace(" ", "")

## データitem_name順にソート
print(uriage_data.sort_values(by = ["item_name"], ascending = True))
print()

# 補正できたか確認する
print(pd.unique(uriage_data["item_name"]))
print(len(pd.unique(uriage_data["item_name"])))
print()

'''
ノック15 : 金額欠損値の補完をする
'''
# 欠損値の確認
print(uriage_data.isnull().any(axis = 0))
print()

# 欠損値の補完
fig_is_null = uriage_data["item_price"].isnull()
for trg in list(uriage_data.loc[fig_is_null, "item_name"].unique()):
    price = uriage_data.loc[(~fig_is_null) & (uriage_data["item_name"] == trg), "item_price"].max()
    uriage_data.loc[(fig_is_null) & (uriage_data["item_name"] == trg), "item_price"] = price
print(uriage_data.head())
print()

# 補完後の欠損値の確認
print(uriage_data.isnull().any(axis = 0))
print()

# 補完金額の検証(最大値と最小値が一致しているか確認)
for trg in list(uriage_data["item_name"].sort_values().unique()):
    print(trg + "の最大値:" + str(uriage_data.loc[uriage_data["item_name"] == trg]["item_price"].max()) + "の最小額:" + str(uriage_data.loc[uriage_data["item_name"] == trg]["item_price"].min(skipna = False)))
print()

'''
ノック16 : 顧客名の揺れを補正する
'''
# データの確認
print(kokyaku_data["顧客名"].head())
print()

print(uriage_data["customer_name"].head())
print()

# スペースの除去
kokyaku_data["顧客名"] = kokyaku_data["顧客名"].str.replace(" ", "")
kokyaku_data["顧客名"] = kokyaku_data["顧客名"].str.replace("　", "")
print(kokyaku_data["顧客名"].head())
print()

"""
ノック17 : 日付の揺れを補正する
"""
# 数値となっている箇所の特定
fig_is_serial = kokyaku_data["登録日"].astype("str").str.isdigit()
print(fig_is_serial.sum())
print()

# 数値から日付に変換
fromSerial = pd.to_timedelta(kokyaku_data.loc[fig_is_serial, "登録日"].astype("float") - 2, unit = "D") + pd.to_datetime('1900/1/1')
print(fromSerial)
print()

# 日付として取り込まれているデータの書式変更
fromString = pd.to_datetime(kokyaku_data.loc[~fig_is_serial, "登録日"])
print(fromString)
print()

# 数値 → 日付に補正したデータと、書式変更をしたデータの結合
kokyaku_data["登録日"] = pd.concat([fromSerial, fromString])
print(kokyaku_data["登録日"])
print()

# 登録月の集計
kokyaku_data["登録年月"] = kokyaku_data["登録日"].dt.strftime("%Y%m")
rslt = kokyaku_data.groupby("登録年月").count()["顧客名"]
print(rslt)
print(len(kokyaku_data))
print()

# 数値項目の有無を確認
fig_is_serial = kokyaku_data["登録日"].astype("str").str.isdigit()
print(fig_is_serial.sum())
print()

'''
ノック18 : 顧客名をキーに2つのデータをジョインする
'''
# データの結合
join_data = pd.merge(uriage_data, kokyaku_data, left_on = "customer_name", right_on = "顧客名", how = "left")
join_data = join_data.drop("customer_name", axis = 1)
print(join_data)
print()

'''
ノック19 : クレンジングしたデータをダンプ(出力)する
'''
# データの整形（列の並び替え）
dump_data = join_data[["purchase_date", "purchase_month", "item_name", "item_price", "顧客名", "かな", "地域", "メールアドレス", "登録日"]]
print(dump_data)
print()

# csvファイル出力
os.chdir("../out")
dump_data.to_csv("dump_data.csv", index = False)

'''
ノック20 : データを集計する
'''
# ダンプファイルの読み込み
import_data = pd.read_csv("dump_data.csv")
print(import_data)

# 購入年月、商品の集計
byItem = import_data.pivot_table(index = "purchase_month", columns = "item_name", aggfunc = "size", fill_value = 0)
print(byItem)
print()

# 購入年月、売上金額の集計
byPrice = import_data.pivot_table(index = "purchase_month", columns = "item_name", values = "item_price", aggfunc = "sum", fill_value = 0)
print(byPrice)
print()

# 購入年月、各顧客の購入数の集計
byCustomer = import_data.pivot_table(index = "purchase_month", columns = "顧客名", aggfunc = "size", fill_value = 0)
print(byCustomer)
print()

# 購入年月、地域における販売数の集計
byRegion = import_data.pivot_table(index = "purchase_month", columns = "地域", aggfunc = "size", fill_value = 0)
print(byRegion)
print()

# 集計期間内での離脱顧客（期間内に購入していないユーザーのチェック）
away_data = pd.merge(uriage_data, kokyaku_data, left_on = "customer_name", right_on = "顧客名", how = "right")
print(away_data[away_data["purchase_date"].isnull()][["顧客名", "メールアドレス", "登録日"]])
print()