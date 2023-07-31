import pandas as pd
df = pd.read_table(r'C:\B_data\output.txt',encoding='cp949')
df = df[1:]
df.columns = ['State']

res = [2,3]

for i in range(1, df.shape[0]):
    if res[0]+(3*i) >= df.shape[0]  or res[1]+(3*i) >= df.shape[0] :
        break
    res.append(res[0]+(3*i))
    res.append(res[1]+(3*i))

df.drop(res,axis=0,inplace=True)
df.reset_index(drop=True, inplace=True)

df

df2 = pd.DataFrame(df['State'].str.split('/').tolist(),columns=['상태','정확도','LED','시간'])

df2['상태']=df2['상태'].str.slice(5,11)
df2['정확도']=df2['정확도'].str.slice(7,14)
df2['LED']=df2['LED'].str.slice(7,10)

df2.dropna(subset=['시간'],inplace=True)
df2['시간']=df2['시간'].str.slice(6,20)

df2['날짜'] = df2['시간'].str.slice(0,6)
df2['시간']=df2['시간'].str.slice(5,14)

df2 = df2[['날짜','시간','상태','정확도','LED']]

df2['ID'] = '동대문2-12'
df2['장소'] = '경동시장  '

df2 = df2[['ID','장소','날짜','시간','상태','정확도','LED']]

df2.reset_index(drop=True, inplace=True)


df2.to_csv(r'C:\B_data\우회전 경고.csv', index=False, encoding='utf-8-sig')
df2.to_csv(r'C:\B_data\우회전 경고.txt', index=False, encoding='utf-8-sig', sep = '\t')

df2

df3 = pd.DataFrame(columns=['ID','장소','날짜','시간','상태','LED'])
num = df3.shape[0]
df3.loc[num+1] = [df2['ID'].unique()[0], df2['장소'].unique()[0],  df2['날짜'].unique()[0] , str(list(df2['시간'])[0] + ' ~' + list(df2['시간'])[-1])
              , df2['상태'].unique()[0], df2['LED'].unique()[0]]


df3.to_csv(r'C:\B_data\res.txt', index=False, encoding='utf-8-sig',  sep = '\t')

with open(r'C:\B_data\res.txt', "r",encoding='utf-8') as f:
    lines = f.readlines()
with open(r'C:\B_data\res.txt', "w",encoding='utf-8') as f:
    for line in lines:
        if line.strip("\n") != '\ufeffID\t장소\t날짜\t시간\t상태\tLED':     # <= 이 문자열만 골라서 삭제
            f.write(line)
        if line == '동대문2-12\t경동시장\t06-21\t15:48:20 ~ 15:49:20\t우회전 경고\tON \n' :
            line = '동대문2-12\t경동시장\t06-21 \t15:48:20 ~ 15:49:20\t우회전 경고\tON \n'
            f.write(line)
