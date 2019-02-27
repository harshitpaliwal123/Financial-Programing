# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 15:09:41 2018

@author: rdoyen
"""

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

#Setting up the working directory of this file
os.chdir('C:/Users/rdoyen/Desktop/Financial Prof/Project/data_berka')

#Work on the account table
account = pd.read_csv('account.asc', sep=';')
account.rename(index=str, columns={"date":"Opening_Date"})

#Creating of column for the year, month and day. 
account['date']=pd.to_datetime(account['date'], format='%y%m%d')
account['Year'] = pd.DatetimeIndex(account['date']).year
account['Month'] = pd.DatetimeIndex(account['date']).month
account['Day'] = pd.DatetimeIndex(account['date']).day
#renaming the frequency variable
account['frequency'] = np.where(account['frequency']=='POPLATEK MESICNE', 'Monthly', np.where(account['frequency']=='POPLATEK TYDNE', 'Weekly','Immediate' ))
#removing district id of the branch. 
del(account['district_id'])


#time serie evolution of account creation per month
x=[]

for i in range(1,12):
    PerMonth=np.sum(account['Month']==i)
    x.append(PerMonth)

print(x)

#time serie evolution of account creation per month and per year
accountperyear = account.pivot_table(index='Year', columns='Month', values='account_id',
                               aggfunc='count')

print(accountperyear)

accountperyear['Total'] = accountperyear.sum(axis=1)
print(accountperyear)

#Plot of the number of account created per month. (time series analysis)
fig, ax = plt.subplots()
fig.set_size_inches(10, 5, forward=True)

plt.bar(accountperyear.index, height=accountperyear[1],width=0.07, label='January')
plt.bar(accountperyear.index+(1*0.083333), height=accountperyear[2],width=0.07, label='February')
plt.bar(accountperyear.index+(2*0.083333), height=accountperyear[3],width=0.07, label='March')
plt.bar(accountperyear.index+(3*0.083333), height=accountperyear[4],width=0.07, label='April')
plt.bar(accountperyear.index+(4*0.083333), height=accountperyear[5],width=0.07,label='May')
plt.bar(accountperyear.index+(5*0.083333), height=accountperyear[6],width=0.07, label='June')
plt.bar(accountperyear.index+(6*0.083333), height=accountperyear[7],width=0.07, label='July')
plt.bar(accountperyear.index+(7*0.083333), height=accountperyear[8],width=0.07, label='August')
plt.bar(accountperyear.index+(8*0.083333), height=accountperyear[9],width=0.07, label='September')
plt.bar(accountperyear.index+(9*0.083333), height=accountperyear[10],width=0.07, label='October')
plt.bar(accountperyear.index+(10*0.083333), height=accountperyear[11],width=0.07, label='November', color='lawngreen')
plt.bar(accountperyear.index+(11*0.083333), height=accountperyear[12],width=0.07, label='December', color='yellow')
# Fix the tick label of x axis
plt.xticks(accountperyear.index+0.5, accountperyear.index)

plt.show()



#Working on the client table. 
client = pd.read_csv('client.asc', sep=';')
client.head()

client['birth_number'] = client['birth_number'].astype(str)

client['birth_year']= client['birth_number'].str[0:2]
client['birth_month'] = client['birth_number'].str[2:4]
client['birth_day'] = client['birth_number'].str[4:6]

client['birth_year'] = client['birth_number'].transform(lambda bn: int('19' + str(bn)[:2]))

client['birth_month'] = client['birth_month'].astype(int)

client['Gender'] = np.where(client['birth_month']>12, 'Female', 'Male')

#Creating the correct birth_month of the male and female
client['birth_month'] = np.where(client['birth_month']>12, client['birth_month']-50, client['birth_month'] )
client['DOB'] = client.birth_day.str.cat(client.birth_month.astype(str), sep='/')

#To compute the age of the customer, we considered that the year of 'today' was equal to 1999.
#since the database date of 1999
client['age'] = 1999 - client['birth_year']

client['birth_day'] = client['birth_number'].astype(str).str[-2:].astype(int)
#Removing columns
client_clean =client.drop(columns=['birth_number', 'birth_year','birth_month','birth_day','DOB'])


# Each record relates together a client with an account i.e. this relation describes the rights
# of clients to operate accounts
disposition = pd.read_csv('disp.asc', sep=';')

#Merging disposition and client table on client_id
client_disp = pd.merge(client_clean,disposition, on = 'client_id')
client_disp = client_disp[['account_id','client_id','district_id','disp_id','Gender','age','type']]


# Each record describes a credit card issued to an account
card = pd.read_csv('card.asc', sep=';')

#Merge the card table with client_disp on client_id
client_disp_card = pd.merge(client_disp,card, on = 'disp_id', how= 'left').fillna('not availed')

client_disp_card =client_disp_card.drop(columns=['disp_id', 'card_id','issued'])

client_disp_card = client_disp_card.rename(columns={'type_x': 'disposition_type', 'type_y': 'card_type'})

index1 = client_disp_card['disposition_type'] == 'OWNER'

client_disp_card_owner= client_disp_card[index1]
client_disp_card_owner.info()

# Each record describes characteristics of a payment order
order = pd.read_csv('order.asc', sep=';')
#Changing the name of k_symbol for more clarity
order['k_symbol'] = np.where(order['k_symbol']=='SIPO', 'Household', 
                               np.where(order['k_symbol']== 'POJISTNE', 'Insurrance',
                                        np.where(order['k_symbol']== 'LEASING', 'Leasing',
                                                np.where(order['k_symbol']== 'UVER', 'Loan','Other'))))

del(order['bank_to'])
del(order['account_to'])


#Creating a table with the count of loan per account_id for each of the k_symbol
countorder = order.pivot_table(index='account_id', columns='k_symbol', values='amount',
                               aggfunc='count')
#Replacing NaN by 0
#from numpy import *
#where_are_NaNs = isnan(countorder)
#countorder[where_are_NaNs] = 0


#Creating a table with the sum of loan per account_id for each of the k_symbol
sumorder = order.pivot_table(index='account_id', columns='k_symbol', values='amount',
                               aggfunc='sum')
#Replacing NaN by 0
#where_are_NaNs = isnan(sumorder)
#sumorder[where_are_NaNs] = 0

#Merging these two tables together
mergesumcount =pd.merge(countorder, sumorder, left_on='account_id', right_on='account_id').reset_index()


mergesumcount = mergesumcount.rename(index=str, columns={"Household_x": "Count_Order_Household", "Insurrance_x": "Count_Order_Insurrance",
                                                        "Leasing_x":"Count_Order_Leasing","Loan_x":"Count_Order_Loan","Other_x":"Count_Order_Other",
                                                        "Household_y": "Sum_Order_Household", "Insurrance_y": "Sum_Order_Insurrance",
                                                        "Leasing_y":"Sum_Order_Leasing","Loan_y":"Sum_Order_Loan","Other_y":"Sum_Order_Other"})

orderfinal = mergesumcount
orderfinal = orderfinal.set_index('account_id')

# Each record describes one transaction on an account
trans = pd.read_csv('trans.asc', sep=';', low_memory=False)
trans = trans.fillna('OTHERS')

## Pivoting type to get count and sum
trans_type_count = pd.pivot_table(trans, values ="amount", index="account_id", columns="type", aggfunc ='count', fill_value =0)
trans_type_count.rename(columns={'PRIJEM': 'Credit_count', 'VYBER': 'CashWithd_count','VYDAJ': 'Withdraw_count'}, inplace=True)

trans_type_sum = pd.pivot_table(trans, values ="amount", index="account_id", columns="type", aggfunc ='sum', fill_value =0)
trans_type_sum.rename(columns={'PRIJEM': 'Credit_sum', 'VYBER': 'CashWithd_sum','VYDAJ': 'Withdraw_sum'}, inplace=True)

## Transaction Type table by merging trans_type_count & trans_type_sum
trans_type = pd.merge(trans_type_count, trans_type_sum, on = 'account_id', how='left')

#Pivoting OP to get count and sum
trans_op_count = pd.pivot_table(trans, values ="amount", index="account_id", columns="operation", aggfunc ='count', fill_value =0)
trans_op_count.rename(columns={'OTHERS': 'op_Others_count', 'PREVOD NA UCET': 'op_RemittanceBank_count', 'PREVOD Z UCTU': 'op_CollectionBank_count','VKLAD': 'op_CashCredit_count','VYBER': 'op_CashWithdra_count','VYBER KARTOU': 'op_CredCardWithdra_count'}, inplace=True)
                               
trans_op_sum = pd.pivot_table(trans, values ="amount", index="account_id", columns="operation", aggfunc ='sum', fill_value =0)                              
trans_op_sum.rename(columns={'OTHERS': 'op_Others_sum','PREVOD NA UCET': 'op_RemittanceBank_sum', 'PREVOD Z UCTU': 'op_CollectionBank_sum','VKLAD': 'op_CashCredit_sum','VYBER': 'op_CashWithdra_sum','VYBER KARTOU': 'op_CredCardWithdra_sum'}, inplace=True)

## Transaction Operation table by merging trans_op_count & trans_op_sum
trans_op = pd.merge(trans_op_count, trans_op_sum, on = 'account_id', how='left')

## Pivoting k_symbol to get count and sum
trans_k_count = pd.pivot_table(trans, values ="amount", index="account_id", columns="k_symbol", aggfunc ='count', fill_value =0)
trans_k_count.rename(columns={'DUCHOD': 'k_Pension_count','OTHERS': 'k_Others_count', 'POJISTNE': 'k_Insurance_count','SANKC. UROK': 'k_NegFee_count','SIPO': 'k_Household_count','SLUZBY': 'k_Statement_count','UROK': 'k_CreditedInt_count','UVER': 'k_Loan_count'}, inplace=True)

trans_k_sum = pd.pivot_table(trans, values ="amount", index="account_id", columns="k_symbol", aggfunc ='sum', fill_value =0)
trans_k_sum.rename(columns={'DUCHOD': 'k_Pension_sum','OTHERS': 'k_Others_sum', 'POJISTNE': 'k_Insurance_sum','SANKC. UROK': 'k_NegFee_sum','SIPO': 'k_Household_sum','SLUZBY': 'k_Statement_sum','UROK': 'k_CreditedInt_sum','UVER': 'k_Loan_sum'}, inplace=True)
#print(trans_k_sum.head())


## Transaction k_symbol table 
trans_k = pd.merge(trans_k_count, trans_k_sum, on = 'account_id', how='left')

trans_k = trans_k.drop(columns=[' _x',' _y'])
                   
trans_balance = trans.groupby('account_id')['balance'].agg(['mean']).round()
trans_balance.rename(columns={'mean': 'average_balance'}, inplace=True)
print(trans_balance.head())

### merging everything

#df1.merge(df2,on='name').merge(df3,on='name')

trans_merged = trans_type.merge(trans_op,on='account_id').merge(trans_k, on='account_id').merge(trans_balance, on='account_id')
print(trans_merged.head(5))
print(len(trans_merged))

## to add new column
# trans2=trans_merged.assign(Cred_With = Credit_count + Withdraw_count)            

# Each record describes a loan granted for a given account
loan = pd.read_csv('loan.asc', sep=';')
print(loan)

#Merging loan and account on account id
testmerge = pd.merge(loan, account, how='right',on='account_id')

#Merging orderfinal and testmerge on account_id
testmerge2 = pd.merge(orderfinal, testmerge, how='right', on='account_id')


# Each record describes demographic characteristics of a district.
dist = pd.read_csv('district.asc', sep=';')

# Merge trans_merged and TestMerge2
testmerge2 = pd.merge(orderfinal, testmerge, how='right', on='account_id')

merge3 = pd.merge(trans_merged, testmerge2, how='right', on='account_id')

print(merge3.info())

#Creating two variable for the sum and the count of withdrawal
merge3['WithdrawalCount']=merge3['CashWithd_count']+merge3['Withdraw_count']
merge3['WithdrawalSum']=merge3['CashWithd_sum']+merge3['Withdraw_sum']

merge3 = merge3.drop(columns=['CashWithd_count','Withdraw_count','CashWithd_sum','Withdraw_sum'])
merge3.info()

#Rearranging the columns of our basetable for clarity
merge3 = merge3[['account_id','Year','Month','Day', 'frequency','Credit_count','Credit_sum','WithdrawalCount','WithdrawalSum','average_balance',
                 'op_CashCredit_count','op_CollectionBank_count','op_Others_count','op_CashWithdra_count','op_CredCardWithdra_count','op_RemittanceBank_count',
                 'op_CashCredit_sum','op_CollectionBank_sum','op_Others_sum','op_CashWithdra_sum','op_CredCardWithdra_sum','op_RemittanceBank_sum',
                 'k_Statement_count','k_Statement_sum','k_CreditedInt_count','k_CreditedInt_sum','k_NegFee_count','k_NegFee_sum','k_Pension_count','k_Pension_sum',
                 'Count_Order_Leasing','Sum_Order_Leasing',
                 'k_Insurance_count','k_Insurance_sum','Count_Order_Insurrance','Sum_Order_Insurrance',
                 'k_Household_count','k_Household_sum','Count_Order_Household','Sum_Order_Household',
                 'k_Others_count','k_Others_sum','Count_Order_Other','Sum_Order_Other',
                 'k_Loan_count','k_Loan_sum','Count_Order_Loan','Sum_Order_Loan','amount','loan_id','duration','payments','date_x','status'
                 ]]

#Chaning the variable amount to avoid round up
merge3['amount']=merge3['Sum_Order_Loan']*merge3['duration']
merge3 = merge3.drop(columns=['payments','loan_id','date_x'])

#Renaming the columns for clarity
merge3.rename(columns={'Year':'Year_DOC','Month':'Month_DOC','Day':'Day_DOC', 'frequency':'Frequency_issuance','Credit_count':'Nbr_Credit_Trans.','Credit_sum':'Amount_Credit_Trans.','WithdrawalCount':'Nbr_Debit_Trans.','WithdrawalSum':'Amount_Debit_Trans.',
                 'op_CashCredit_count':'Nbr_Credit_Cash','op_CollectionBank_count':'Nbr_Credit_Bank','op_Others_count':'Nbr_Credit_Others','op_CashWithdra_count':'Nbr_Debit_Cash','op_CredCardWithdra_count':'Nbr_Debit_Card','op_RemittanceBank_count':'Nbr_Debit_Bank',
                 'op_CashCredit_sum':'Amount_Credit_Cash','op_CollectionBank_sum':'Amount_Credit_Bank','op_Others_sum':'Amount_Credit_Others','op_CashWithdra_sum':'Amount_Debit_Cash','op_CredCardWithdra_sum':'Amount_Debit_Card','op_RemittanceBank_sum':'Amount_Debit_Bank',
                 'k_Statement_count':'Nbr_Statement_Trans.','k_Statement_sum':'Amount_Statement_Trans','k_CreditedInt_count':'Nbr_Interest_Credited_Trans','k_CreditedInt_sum':'Amount_Interest_Credited_Trans','k_NegFee_count':'Nbr_Sanction_Interest_Trans','k_NegFee_sum':'Amount_Sanction_Interest_Trans','k_Pension_count':'Nbr_Pension_Trans','k_Pension_sum':'Amount_Pension_Trans',
                 'Count_Order_Leasing':'Nbr_PO_Leasing','Sum_Order_Leasing':'Monthly_Amount_PO_Leasing',
                 'k_Insurance_count':'Nbr_Insurance_Trans','k_Insurance_sum':'Amount_Insurance_Trans','Count_Order_Insurrance':'Nbr_PO_Insurance','Sum_Order_Insurrance':'Monthly_Amount_PO_Insurance',
                 'k_Household_count':'Nbr_Household_Trans','k_Household_sum':'Amount_Household_Trans','Count_Order_Household':'Nbr_PO_Household','Sum_Order_Household':'Monthly_Amount_PO_Household',
                 'k_Others_count':'Nbr_Other_Trans','k_Others_sum':'Amount_Other_Trans','Count_Order_Other':'Nbr_PO_Other','Sum_Order_Other':'Monthly_Amount_PO_Other',
                 'k_Loan_count':'Nbr_Loan_Trans','k_Loan_sum':'Amount_Loan_Trans','Count_Order_Loan':'Nbr_PO_Loan','Sum_Order_Loan':'Monthly_Amount_PO_Loan','amount':'Theoritical_Amount_Loan_Due','duration':'Loan_Duration','status':'Status_Loan_Owner'}, inplace=True)

#Merging client_disp_card_owner & merge3
Final_Basetable = pd.merge(client_disp_card_owner,merge3, on='account_id')

Final_Basetable.info()


## Adding some extra demographical information in the final basetable ###
dist = dist.rename(columns={'A1': 'district_id'})
dist['A12'] = dist['A12'].apply(pd.to_numeric, errors='coerce')
dist = dist.drop(columns=['A2','A4','A5','A6','A7','A8','A9','A10','A14','A15','A16'])
dist = dist.rename(columns={'A3':'Region','A11': 'Average_salary_district','A12':'Unemployment_rate_95','A13':'Unemployment_rate_96'})

Final_Basetable2 = pd.merge(Final_Basetable,dist, on = 'district_id')
Final_Basetable2['Average_Unemployment_rate']= (Final_Basetable2.Unemployment_rate_95 + Final_Basetable2.Unemployment_rate_96)/2
Final_Basetable2 = Final_Basetable2.drop(columns=['Unemployment_rate_95', 'Unemployment_rate_96'])
Final_Basetable2 = Final_Basetable2.rename(columns={'Average_Unemployment_rate': 'Average_Unemployment_rate_district'})

#Final Basetable finished
Final_Basetable=Final_Basetable2 
print(Final_Basetable.info())


#Graph Average amount of Household Permanent Order per age and gender
moyenneAmountparAgeGender = Final_Basetable.pivot_table(index='age', columns='Gender',
                                                  values='Amount_Household_Trans',aggfunc='mean')

plt.subplot(3,1,1)

bars1 = moyenneAmountparAgeGender['Female']
bars2 = moyenneAmountparAgeGender['Male']

plt.plot(moyenneAmountparAgeGender.index,bars1, color='red', label='Female')
plt.plot(moyenneAmountparAgeGender.index,bars2, color='blue', label='Male' )
plt.xlim(17,81)
plt.legend()
plt.xlabel('Age')
plt.ylabel('Avg amount of Household PO')
plt.title('Average amount of Household Permanent Order per age and gender')

plt.subplot(3,1,3)

moyenneAmountparAgeGender2 = Final_Basetable.pivot_table(index='age', columns='Gender',
                                                  values='Monthly_Amount_PO_Household',aggfunc='mean')

bars1 = moyenneAmountparAgeGender2['Female']
bars2 = moyenneAmountparAgeGender2['Male']

plt.plot(moyenneAmountparAgeGender2.index,bars1, color='red', label='Female')
plt.plot(moyenneAmountparAgeGender2.index,bars2, color='blue', label='Male' )
plt.xlim(17,81)
plt.legend()
plt.xlabel('Age')
plt.ylabel('Avg monthly amount of Household PO')
plt.title('Average Monthly Amount of Household Permanent Order per age and gender')

plt.show()

#Graph Average amount of Insurrance Permanent Order per age and gender
plt.subplot(3,1,1)

moyenneAmountInsurranceparAgeGender = Final_Basetable.pivot_table(index='age', columns='Gender',
                                                  values='Amount_Insurance_Trans',aggfunc='mean')

bars1 = moyenneAmountInsurranceparAgeGender['Female']
bars2 = moyenneAmountInsurranceparAgeGender['Male']

plt.plot(moyenneAmountInsurranceparAgeGender.index,bars1, color='red', label='Female')
plt.plot(moyenneAmountInsurranceparAgeGender.index,bars2, color='blue', label='Male' )
plt.xlim(17,81)
plt.xlabel('Age')
plt.ylabel('Avg amount of Insurrance PO')
plt.legend()
plt.title('Average amount of Insurrance Permanent Order per age and gender')


plt.subplot(3,1,3)
moyenneAmountInsurranceparAgeGender2 = Final_Basetable.pivot_table(index='age', columns='Gender',
                                                  values='Monthly_Amount_PO_Insurance',aggfunc='mean')

bars1 = moyenneAmountInsurranceparAgeGender2['Female']
bars2 = moyenneAmountInsurranceparAgeGender2['Male']

plt.plot(moyenneAmountInsurranceparAgeGender2.index,bars1, color='red', label='Female')
plt.plot(moyenneAmountInsurranceparAgeGender2.index,bars2, color='blue', label='Male' )
plt.xlim(17,81)
plt.legend()
plt.xlabel('Age')
plt.ylabel('Avg monthly amount of Insurrance PO')
plt.title('Average Monthly Amount of Insurrance Permanent Order per age and gender')
plt.show()

#Creating of graph of the Average Balance per card type and per gender

avgBalanceperCardType = Final_Basetable.pivot_table(index='card_type', columns='Gender',
                                                  values='average_balance',aggfunc='mean')

bars1=avgBalanceperCardType['Male']
bars2=avgBalanceperCardType['Female']
r1 = np.arange(len(bars1))
r2 = [x + 0.3 for x in r1]
plt.bar(r1, bars1, width = 0.3, color = 'blue', edgecolor = 'black',label='Male')
plt.bar(r2, bars2, width = 0.3, color = 'red', edgecolor = 'black',label='Female')
plt.xticks([r + 0.3 for r in range(len(bars1))], ['classic', 'gold', 'junior','not availed'])
plt.ylim(30000,65000)
plt.xlabel('Card type')
plt.ylabel('Average Balance')
plt.title('Average Balance per card type and per gender')
plt.legend()
plt.show()

#changed 19/12
#Boxplot Average balance perd card type
fig, ax = plt.subplots()
fig.set_size_inches(20, 5, forward=True)

plt.subplot(1,2,1)
plt.boxplot([
                Final_Basetable.average_balance[Final_Basetable.card_type == 'gold'].values,
                Final_Basetable.average_balance[Final_Basetable.card_type == 'classic'].values, 
                Final_Basetable.average_balance[Final_Basetable.card_type == 'junior'].values,
                Final_Basetable.average_balance[Final_Basetable.card_type == 'not availed'].values
            ])

plt.xticks([1, 2, 3,4], ['gold', 'classic', 'junior','not availed'])
plt.title('Average balance perd card type')
plt.xlabel('Card type')
plt.ylabel('Average balance')
plt.show()
#changed 19/12

#Creating the graph of the Count of user per card type and per age
avgBalanceperCardType = Final_Basetable.pivot_table(index='age', columns='card_type',
                                                  values='average_balance',aggfunc='count')

plt.plot(avgBalanceperCardType.index,avgBalanceperCardType['classic'], label='classic')
plt.plot(avgBalanceperCardType.index,avgBalanceperCardType['junior'], label='junior')
plt.plot(avgBalanceperCardType.index,avgBalanceperCardType['gold'], label='gold')
plt.plot(avgBalanceperCardType.index,avgBalanceperCardType['not availed'], label='not availed')
plt.legend()
plt.xlabel('Card type')
plt.ylabel('Count of user')
plt.title('Count of user per card type and per age')
plt.show()


### Boxplot with the Theoritical Amount for each Status of Loan owner

fig, ax = plt.subplots()
fig.set_size_inches(20, 5, forward=True)

plt.subplot(1,2,1)
plt.boxplot([
                Final_Basetable.Theoritical_Amount_Loan_Due[Final_Basetable.Status_Loan_Owner == 'A'].values,
                Final_Basetable.Theoritical_Amount_Loan_Due[Final_Basetable.Status_Loan_Owner == 'B'].values, 
                Final_Basetable.Theoritical_Amount_Loan_Due[Final_Basetable.Status_Loan_Owner == 'C'].values,
                Final_Basetable.Theoritical_Amount_Loan_Due[Final_Basetable.Status_Loan_Owner == 'D'].values
            ])

plt.xticks([1, 2, 3,4], ['A', 'B', 'C','D'])
plt.ylim(0,600000)
plt.title('Theoritical Amount due for Loans for Each Group Status')
plt.xlabel('Status of Loan Owner')
plt.ylabel('Total Amount of the Loan')

### Boxplot with the Actual Amount paid for each Status of Loan owner
plt.subplot(1,2,2)
plt.boxplot([
                Final_Basetable.Amount_Loan_Trans[Final_Basetable.Status_Loan_Owner == 'A'].values,
                Final_Basetable.Amount_Loan_Trans[Final_Basetable.Status_Loan_Owner == 'B'].values, 
                Final_Basetable.Amount_Loan_Trans[Final_Basetable.Status_Loan_Owner == 'C'].values,
                Final_Basetable.Amount_Loan_Trans[Final_Basetable.Status_Loan_Owner == 'D'].values
            ])

plt.xticks([1, 2, 3,4], ['A', 'B', 'C','D'])
plt.ylim(0,600000)
plt.title('Actual Amount paid for Loans for Each Group Status')
plt.xlabel('Status of Loan Owner')
plt.ylabel('Actual Amount paid for the Loan')
plt.show()


#### Boxplot with the Monthly Payment loans per client status

plt.boxplot([
                Final_Basetable.Monthly_Amount_PO_Loan[Final_Basetable.Status_Loan_Owner == 'A'].values,
                Final_Basetable.Monthly_Amount_PO_Loan[Final_Basetable.Status_Loan_Owner == 'B'].values, 
                Final_Basetable.Monthly_Amount_PO_Loan[Final_Basetable.Status_Loan_Owner == 'C'].values,
                Final_Basetable.Monthly_Amount_PO_Loan[Final_Basetable.Status_Loan_Owner == 'D'].values
            ])

plt.xticks([1, 2, 3,4], ['A', 'B', 'C','D'])
plt.title('Monthly Amount due for Loans for Each Group Status')
plt.xlabel('Status of Loan Owner')
plt.ylabel('Monthly Amount due for the Loan')
plt.show()



#### chart for duration with average amount spent per status

table1 = Final_Basetable.pivot_table(index='Loan_Duration', columns = 'Status_Loan_Owner', values= 'Theoritical_Amount_Loan_Due', aggfunc='mean')
print(table1)
table1.info()

fig, ax = plt.subplots()
fig.set_size_inches(10, 5, forward=True)

plt.plot(table1.index, table1['A'],color='orange')
plt.plot(table1.index, table1['B'],color='red')
plt.plot(table1.index, table1['C'],color='blue')
plt.plot(table1.index, table1['D'],color='green')
plt.title('Average Amount of Loans per Duration of Loans for each Group Status')
plt.xlabel('Loan Total Duration')
plt.ylabel('Average Amount of the Loans')
plt.legend()
plt.show()


### Distribution of clients in status A,B,C,D

loan_status = Final_Basetable.pivot_table(index='Status_Loan_Owner', columns='Gender', values='client_id',
                               aggfunc='count')
loan_status
barWidth = 0.3
r1 = np.arange(len(loan_status['Female']))
r2 = [x + barWidth for x in r1]


plt.bar(r1, height=loan_status['Female'], width=barWidth, color='red', label='Female')
plt.bar(r2, height=loan_status['Male'], width=barWidth, color='blue', label='Male')

plt.xticks([r + barWidth for r in range(len(loan_status['Female']))], ['A', 'B', 'C','D'])
plt.title('Distribution of Clients for each Loan Group Status per Gender')
plt.xlabel('Status of Loan Owner')
plt.ylabel('Number of Clients')
plt.legend()
plt.show()

Final_Basetable.info()


# Distribution by age for bad Loan Clients
loans_B_D = Final_Basetable.pivot_table(index='age', columns='Status', values='client_id',
                               aggfunc='count')

loans_B_D

fig, ax = plt.subplots()
fig.set_size_inches(10, 5, forward=True)

plt.bar(loans_B_D.index, height=loans_B_D['Bad Client'], width=0.1, color='red', label='Bad Clients')
plt.show()

#### Overall Overview of clients ###

age_pivot = Final_Basetable.pivot_table(index='Gender',columns='age',values='account_id',
                               aggfunc='count')

card_pivot = Final_Basetable.pivot_table(columns='card_type',values='account_id',
                               aggfunc='count')

#### Card types and distribution analysis #####
card_group = Final_Basetable.groupby('card_type').count()

x= card_group.index
y = card_group['account_id'].values

plt.bar(x,y)
plt.show()


#### Gender distribution analysis #####
age_group = Final_Basetable.groupby('Gender').count()

x2= age_group.index
y2 = age_group['account_id'].values

plt.bar(x2,y2,)
plt.show()

### District_account_id distribution analysis #####
dist_group = Final_Basetable.groupby('district_id').count()

x3= dist_group.index
y3 = dist_group['account_id'].values

plt.bar(x3,y3)
plt.show()

##### Distribution of clients per age
age_graph = Final_Basetable.pivot_table(index='age', values='client_id',
                               aggfunc='count')
fig, ax = plt.subplots()
fig.set_size_inches(15, 5, forward=True)

plt.bar(age_graph.index, height=age_graph['client_id'], width=0.8)
plt.title('Distribution of Clients per Age')
plt.xlabel('Age')
plt.ylabel('Number of Clients')
plt.xticks(np.arange(15, 90, step=5))
plt.show()


##### Distribution of clients per region
region_graph = Final_Basetable.pivot_table(index='Region', values='client_id',
                               aggfunc='count')


fig, ax = plt.subplots()
fig.set_size_inches(15, 5, forward=True)

plt.bar(region_graph.index, height=region_graph['client_id'], width=0.8)
plt.title('Distribution of Clients per Region')
plt.xlabel('Region')
plt.ylabel('Number of Clients')
plt.show()

##### debit/credit transaction per region
debit_region = Final_Basetable.pivot_table(index='Region', values=['Amount_Debit_Trans.', 'Amount_Credit_Trans.','Average_salary_district'],
                               aggfunc='mean').round()

fig, ax = plt.subplots()
fig.set_size_inches(15, 5, forward=True)
debit_region
barWidth = 0.3
rC = np.arange(len(debit_region['Amount_Credit_Trans.']))
rD = [x + barWidth for x in rC]

plt.bar(rC, height=debit_region['Amount_Credit_Trans.'], width=barWidth, color='orange', label='Credit')
plt.bar(rD, height=debit_region['Amount_Debit_Trans.'], width=barWidth, color='green', label='Debit')
plt.xticks([r + barWidth for r in range(len(debit_region['Amount_Credit_Trans.']))], ['Prague', 'central Bohemia', 'east Bohemia','north Bohemia','north Moravia','south Bohemia','south Moravia','west Bohemia'])
plt.title('Average Amount of Debits/Credits per Region')
plt.xlabel('Region')
plt.ylabel('Average Amount Debit/Credit')
plt.legend()
plt.ylim(600000, 750000)
plt.show()

fig, ax = plt.subplots()
fig.set_size_inches(15, 5, forward=True)

plt.plot(debit_region.index, debit_region['Average_salary_district'])
plt.title('Average Salary per Region')
plt.xlabel('Region')
plt.ylabel('Average Salary per month')
plt.show()


