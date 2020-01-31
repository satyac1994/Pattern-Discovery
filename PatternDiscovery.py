# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 12:09:34 2019

@author: chodiss
"""

import pandas as pd
#from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
df_ir=pd.read_csv("italy_retail.csv")
# Dropped the unnamed column as we are not sure what the column describes
df_ir.drop(['Unnamed: 0'],axis=1,inplace=True)
df_ir_temp=df_ir[['InvoiceNo','Description','Quantity']]
df_irg=df_ir_temp.groupby(['InvoiceNo','Description'])['Quantity'].sum()

temp=df_irg.unstack().reset_index().fillna(0).set_index('InvoiceNo')
print(temp)
def encoding(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
temp1 = temp.applymap(encoding)
print(temp1)

fi= apriori(temp1, min_support=0.1, use_colnames=True)
rules = association_rules(fi, metric="confidence", min_threshold=1)
for i in rules:
    print(rules[i])
#print(rules)

rules = rules.sort_values(['confidence'], ascending =[False]) 
#print(rules[["antecedents","consequents"]].head(6)) 
print(rules)