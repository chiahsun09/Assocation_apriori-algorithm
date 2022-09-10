#https://www.twblogs.net/a/5d675eecbd9eee5327fed1fc
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 16:40:36 2019

@author: 橘貓吃不胖

購物籃分析實例
"""

import pandas as pd

# 引入訂單數據
df = pd.read_csv('mall_basket_series.csv')

# 轉換成指定的格式
#只要準備訂單編號和購物明細
basket = (df.groupby(['ORDER_NO', 'PROD_NAME'])['ORDER_NO']
          .count().unstack()
          .fillna(0))



def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

# 將數據填充爲0、1的格式
basket_sets = basket.applymap(encode_units)   



from mlxtend.frequent_patterns import apriori
# 生成頻繁項目集，指定最小支持度爲0.5(指購物明細至少有佔20%以上，熱銷品)
frequent_itemsets=apriori(basket_sets, min_support=0.2,use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets.to_csv('mall_basket_series_support.csv',index=False)


from mlxtend.frequent_patterns import association_rules
# 生成關聯規則，最小置信度爲0.01
rules=association_rules(frequent_itemsets, metric="confidence", min_threshold=0.01)
rules["antecedents_len"]=rules["antecedents"].apply(lambda x: len(x))


#rules[rules["antecedents"]=={'aaa','fff','ddd'}]
#想要促銷指定貨品，來作綑綁式銷售，因此需要指明 antecedents


rules.rename(columns = {'support':'support支持度', 'confidence':'confidence置信度','lift':'lift提升度'}, inplace = True)


rules.to_csv('mall_basket_series_bi.csv',index=False)