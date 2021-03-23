### Analysis of Suicides in India

Here we are analysing the Suicide dataset given above, the dataset consists of the data, from all over India, of the number of people who have committed suicide during the period between 2001 and 2012, the State that they are from, the year, the cause, their Education qualification, their Social Status, their Profession, their age and the means that they adopted to commmit suicide.

The dataset is fairly huge having around 237,520 rows/values and 7 columns, good for analysis. 

Here we will be trying to answer some basic questions and also try to draw a few other conclusions that is hidden in the data.

The modules that we would be using for this purpose are :
1. pandas - for reading the dataset and for it's inbuilt functions that help us manupulate the data in the dataset.
2. numpy - for performing all the mathematical computations on the data.
3. seaborn - for the beautiful bargraphs and plots that it has, which can be used to visualize the data, helpful for analysis
4. matplotlib - for drawing the piechart's, also used to aid our analysis.

Incase you do not have any of the above packages you can install them using the command -pip install <library_name> in Anaconda prompt or the command prompt.

Incase you want to try this on your own, download the dataset and the relevant format of the code whether '.ipnyb' or '.py', put both the dataset and the code in the same directory and then run it locally in your machine.

Try to do it on your own for better understanding and to learn the concepts.

Let's begin, Happy Learning.


```python
# importing the required libraries 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```


```python
# Read the Suicide data '.csv' and check if it is has been read properly

df = pd.read_csv('Suicides in India 2001-2012.csv')
df.head()
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>Year</th>
      <th>Type_code</th>
      <th>Type</th>
      <th>Gender</th>
      <th>Age_group</th>
      <th>Total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A &amp; N Islands</td>
      <td>2001</td>
      <td>Causes</td>
      <td>Illness (Aids/STD)</td>
      <td>Female</td>
      <td>0-14</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A &amp; N Islands</td>
      <td>2001</td>
      <td>Causes</td>
      <td>Bankruptcy or Sudden change in Economic</td>
      <td>Female</td>
      <td>0-14</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A &amp; N Islands</td>
      <td>2001</td>
      <td>Causes</td>
      <td>Cancellation/Non-Settlement of Marriage</td>
      <td>Female</td>
      <td>0-14</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A &amp; N Islands</td>
      <td>2001</td>
      <td>Causes</td>
      <td>Physical Abuse (Rape/Incest Etc.)</td>
      <td>Female</td>
      <td>0-14</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A &amp; N Islands</td>
      <td>2001</td>
      <td>Causes</td>
      <td>Dowry Dispute</td>
      <td>Female</td>
      <td>0-14</td>
      <td>0</td>
    </tr>
  </tbody>
</table>





```python
# See the common statistics of the data

df.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>237519.000000</td>
      <td>237519.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2006.500448</td>
      <td>55.034477</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.452240</td>
      <td>792.749038</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2001.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2004.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2007.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2010.000000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2012.000000</td>
      <td>63343.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# See the basic description of the variables, their type and value count

df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 237519 entries, 0 to 237518
    Data columns (total 7 columns):
     #   Column     Non-Null Count   Dtype 
    ---  ------     --------------   ----- 
     0   State      237519 non-null  object
     1   Year       237519 non-null  int64 
     2   Type_code  237519 non-null  object
     3   Type       237519 non-null  object
     4   Gender     237519 non-null  object
     5   Age_group  237519 non-null  object
     6   Total      237519 non-null  int64 
    dtypes: int64(2), object(5)
    memory usage: 12.7+ MB
    


```python
# print the all the states available in the dataset

print(df['State'].unique())
```

    ['A & N Islands' 'Andhra Pradesh' 'Arunachal Pradesh' 'Assam' 'Bihar'
     'Chandigarh' 'Chhattisgarh' 'D & N Haveli' 'Daman & Diu' 'Delhi (Ut)'
     'Goa' 'Gujarat' 'Haryana' 'Himachal Pradesh' 'Jammu & Kashmir'
     'Jharkhand' 'Karnataka' 'Kerala' 'Lakshadweep' 'Madhya Pradesh'
     'Maharashtra' 'Manipur' 'Meghalaya' 'Mizoram' 'Nagaland' 'Odisha'
     'Puducherry' 'Punjab' 'Rajasthan' 'Sikkim' 'Tamil Nadu'
     'Total (All India)' 'Total (States)' 'Total (Uts)' 'Tripura'
     'Uttar Pradesh' 'Uttarakhand' 'West Bengal']
    


```python
# replace all the states specified as Union Territories into normal states since that distinction does not matter to us here.

df.replace('A & N Islands (Ut)', 'A & N Islands', inplace = True)
df.replace('Chandigarh (Ut)', 'Chandigarh', inplace = True)
df.replace('D & N Haveli (Ut)', 'D & N Haveli', inplace = True)
df.replace('Daman & Diu (Ut)', 'Daman & Diu', inplace = True)
df.replace('Lakshadweep (Ut)', 'Lakshadweep', inplace = True)
df.replace('Delhi (Ut)', 'Delhi', inplace = True)

# replace 'Bankruptcy or Sudden change in Economic' to 'Bankruptcy or Sudden change in Economic Status' to ensure uniformity
# since both are the same but a few values have the prior and some the former.

df.replace('Bankruptcy or Sudden change in Economic', 'Bankruptcy or Sudden change in Economic Status', inplace = True)

# Here too we are trying to ensure uniformity by replacing the values properly

df.replace('Others (Please Specify)', 'By Other means', inplace = True)
df.replace('Not having Children(Barrenness/Impotency', 'Not having Children (Barrenness/Impotency)', inplace = True)
```


```python
# Here we remove all the values whose exact causes are not known, this is to ensure that the data whose cause is not known does not
# end up affecting our analysis.

df = df.drop(df[(df.State == 'Total (Uts)') | (df.State == 'Total (All India)') | (df.State =='Total (States)')].index)
df = df.drop(df[(df.Type == 'By Other means') | (df.Type =='Other Causes (Please Specity)') | (df.Type =='Other Causes (Please Specify)') | (df.Type =='Causes Not known') | (df.Type =='By Other means (please specify)')].index)
df = df.drop(df[df['Total'] == 0].index)
```


```python
#Plot a bargraph based on the gender to see which gender people commit the most suicides

sns.set(rc={'figure.figsize':(10,8)})
sns.barplot(x=df['Gender'], y=df['Total'])
```




    <AxesSubplot:xlabel='Gender', ylabel='Total'>




    
![output_9_1](https://user-images.githubusercontent.com/50414959/112197340-fc5a6200-8c31-11eb-826c-74e1b1d4e474.png)
    



```python
# Draw the same bargraph as a pie chart to see the percentage stats

df_gender = df.groupby('Gender')['Total'].sum()
df_gender_type = pd.DataFrame(df_gender).reset_index().sort_values('Total')
labels = df_gender_type['Gender']
exp = (0,0.1)
plt.pie(df_gender_type['Total'], labels=labels, autopct='%1.1f%%', explode = exp, shadow=True)
plt.show()
```


    
![output_10_0](https://user-images.githubusercontent.com/50414959/112197361-04b29d00-8c32-11eb-9f1b-4c3659949fb7.png)

    


From the above bargraph and pie chart we can see that more number of Males commit suicide than females, reasons why this happens is not yet known from the analysis performed until now.

But we can now establish the fact that more number of males commit suicide than females, causes based on the number of people who have committed suicide from the period of 2001 to 2012 62.8% are men and rest 37.2% women.


```python
# Here we are spliting the data into different age groups

gp_age = df.groupby('Age_group')['Total'].sum()
df_age_type = pd.DataFrame(gp_age).reset_index().sort_values('Age_group')
df_age_type = df_age_type.drop([0])
df_age_type
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age_group</th>
      <th>Total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0-14</td>
      <td>59358</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15-29</td>
      <td>1197721</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30-44</td>
      <td>1113501</td>
    </tr>
    <tr>
      <th>4</th>
      <td>45-59</td>
      <td>652452</td>
    </tr>
    <tr>
      <th>5</th>
      <td>60+</td>
      <td>246480</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Draw a bargraph categorising the people into different age groups

sns.barplot(x = df_age_type['Age_group'], y = df_age_type['Total'])
```




    <AxesSubplot:xlabel='Age_group', ylabel='Total'>




    
![output_13_1](https://user-images.githubusercontent.com/50414959/112197391-0c724180-8c32-11eb-8895-9350b6d0b20c.png)

    



```python
# Draw the same bargraph as a pie chart to see the percentage stats

labels = df_age_type['Age_group']
exp = (0,0.1,0.1,0,0)
plt.pie(df_age_type['Total'], labels=labels, autopct='%1.1f%%', explode = exp, shadow=True)
plt.show()
```


    
![output_14_0](https://user-images.githubusercontent.com/50414959/112197410-1136f580-8c32-11eb-9423-e0faecd4934a.png)

    


From the above charts we can see that more number of Suicides are committed by the people belonging to the age groups 15-29 yrs (36.6%) closely followed by those in the age group 30-44 yrs (34.1%).

This gives rise to the conculsion that Suicides are more common among teens, teenagers and the working population then among others, thereby helping us to focus, the limited resources for Suicide prevention campaigns, for people in those age groups.


```python
# Here we are categorising the people into different age groups and proceed to futher 
# split them on the basis of their gender

df_age = df.drop(df[(df['Age_group'] == '0-100+')].index)
df_age = df_age.groupby(['Age_group', 'Gender'])['Total'].sum()
df_age_gender = pd.DataFrame(df_age).reset_index().sort_values('Age_group')
df_age_gender
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age_group</th>
      <th>Gender</th>
      <th>Total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0-14</td>
      <td>Female</td>
      <td>31213</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0-14</td>
      <td>Male</td>
      <td>28145</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15-29</td>
      <td>Female</td>
      <td>582173</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15-29</td>
      <td>Male</td>
      <td>615548</td>
    </tr>
    <tr>
      <th>4</th>
      <td>30-44</td>
      <td>Female</td>
      <td>385019</td>
    </tr>
    <tr>
      <th>5</th>
      <td>30-44</td>
      <td>Male</td>
      <td>728482</td>
    </tr>
    <tr>
      <th>6</th>
      <td>45-59</td>
      <td>Female</td>
      <td>179904</td>
    </tr>
    <tr>
      <th>7</th>
      <td>45-59</td>
      <td>Male</td>
      <td>472548</td>
    </tr>
    <tr>
      <th>8</th>
      <td>60+</td>
      <td>Female</td>
      <td>73470</td>
    </tr>
    <tr>
      <th>9</th>
      <td>60+</td>
      <td>Male</td>
      <td>173010</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Draw a bargraph for the above data that we had split

sns.barplot(x = "Age_group", y = "Total", hue = "Gender", data=df_age_gender)
```




    <AxesSubplot:xlabel='Age_group', ylabel='Total'>




    
![output_17_1](https://user-images.githubusercontent.com/50414959/112197437-17c56d00-8c32-11eb-9610-69df9b618385.png)
    


From the above bargraph we can see that, in the age group 30-44 (Working class) more number of males commit suicide than females, and in the age group of 15-29 (teenage) the number is almost the same with men excceding the women by not so huge a margin. 
The reasons for this may be many but we cannot draw a concrete conclusion on the reason given the fact that we either do not have enough data or we would have to analyse further.


```python
# Here we are grouping the data statewise so as to see which state has the most cases.

group = df.groupby('State')['Total'].sum()
tot_suicides = pd.DataFrame(group).reset_index().sort_values('Total',ascending=False)
```


```python
# Draw a bargraph categorising the data into statewise form, which is obtained from above

fig, ax = plt.subplots(figsize=(18,6))
plot = sns.barplot(x = 'State', y = 'Total', data = tot_suicides)
plot.set_xticklabels(plot.get_xticklabels(), rotation = 45)
plt.show()
```


![output_20_0](https://user-images.githubusercontent.com/50414959/112197451-1bf18a80-8c32-11eb-98b8-d92f3793c5f0.png)

    


From the above bargraph we can see that Maharastra has registered the most number of Suicides, followed by West Bengal, Andhra Pradesh and Tamil Nadu. West Bengal, Andhra Pradesh and Tamil Nadu have almost the same number of cases while Maharastra exceeds them by a wide margin. 

Towards the end you can see that the North-eastern states, and the union territories have the least number of Suicides recorded.


```python
# Here we are grouping the data yearwise so as to see which year has the most cases.

group_yr = df.groupby('Year')['Total'].sum()
yr = pd.DataFrame(group_yr).reset_index().sort_values('Year',ascending=False)
print(group_yr)
```

    Year
    2001    467928
    2002    476738
    2003    482322
    2004    486323
    2005    486115
    2006    512676
    2007    522233
    2008    531216
    2009    539470
    2010    564083
    2011    564376
    2012    547894
    Name: Total, dtype: int64
    


```python
# Draw a bargraph categorising the data into yearwise form, which is obtained from above

plot = sns.barplot(x = 'Year', y = 'Total', data = yr, palette = 'flare')
```


    
![output_23_0](https://user-images.githubusercontent.com/50414959/112197469-23189880-8c32-11eb-9723-1462b83c0043.png)

    


From the above graph we can conclude that the number of Suicide cases has been raising year by year with 2010 and 2011 recording the highest number of cases in 12 years. We can see that the number of cases shows a clear decline in 2012 from its steady rise in the previous years. 

Let's take the analysis further by analying all the individual factors like Social status, the means used, the causes, their Educational qualification and Professional Profile to gather more insights on why people commit suicide. This way we can gather some important conclusions which can help us in reducing the number of cases by helping us target the major reasons thereby giving us significant results than could be obtained if we were to attempt to reduce the cases without the any sort of analysis.


```python
# Here we are grouping the data based on the Type code which consists of all the factors 
# that we wish to analyse, as mentioned above.

group_count = df['Type_code'].value_counts()
group_count
```




    Causes                  33134
    Means_adopted           26709
    Professional_Profile    17782
    Education_Status         5602
    Social_Status            3349
    Name: Type_code, dtype: int64




```python
# Draw a bargraph of the data that we have grouped above

sns.set(rc={'figure.figsize':(12,9)})
sns.countplot(df['Type_code'])
```

    D:\External Application\PythonIDE\lib\site-packages\seaborn\_decorators.py:43: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      FutureWarning
    




    <AxesSubplot:xlabel='Type_code', ylabel='count'>




    
![output_27_2](https://user-images.githubusercontent.com/50414959/112197492-290e7980-8c32-11eb-8aa2-2ea8fbd4e97a.png)

    


From the above graph we can see how much of data is available for each of the catergory mentioned.

Now let's analyse each of the catergory in more detail and see which type in each category contribues to more number of Suicides.

Let's begin our analysis with the Social Status of the people


```python
# Segregating the data where Type_code is Social_Status 

df_bycode = df[df['Type_code'] == 'Social_Status']
df_bycode['Type'].unique()
```




    array(['Married', 'Never Married', 'Divorcee', 'Widowed/Widower',
           'Seperated'], dtype=object)




```python
# Grouping the data into its each of its constituent types

df_social = df_bycode.groupby('Type')['Total'].sum()
df_social_type = pd.DataFrame(df_social).reset_index().sort_values('Total')
print(f'People who have mentioned their Social Status {df_bycode.shape[0]}')
```

    People who have mentioned their Social Status 3349
    


```python
# Drawing a pie chart of the data that we have grouped above

labels = df_social_type['Type']
exp = (0,0,0,0,0.1)
plt.pie(df_social_type['Total'], labels=labels, autopct='%1.1f%%', explode = exp, shadow=True)
plt.show()
```


    
![output_33_0](https://user-images.githubusercontent.com/50414959/112197512-2e6bc400-8c32-11eb-9934-b84b5eea7621.png)

    



```python
#Drawing the same pie chart as a bargraph to see the absolute values

sns.barplot(x = df_social_type['Total'], y=df_social_type['Type'])
```




    <AxesSubplot:xlabel='Total', ylabel='Type'>




    
![output_34_1](https://user-images.githubusercontent.com/50414959/112197529-3297e180-8c32-11eb-9571-c9acaeb0a2db.png)

    


From the above pie chart and bargraph we can conclude that more number of Suicides occurs among married people followed by unmarried people. The least number of cases can be seen for divorced people. We can draw a conclusion here that the chance a person might likely commit suicide increases if he/she is married, i.e married couple have a higher chance of committing Suicide which means that many married people are more likely unhappy with their lives.


```python
# Segregating the data where Type_code is Professional_Profile

df_bycode = df[df['Type_code'] == 'Professional_Profile']
df_bycode['Type'].unique()
```




    array(['Student', 'House Wife', 'Service (Private)',
           'Public Sector Undertaking', 'Service (Government)',
           'Farming/Agriculture Activity', 'Retired Person',
           'Self-employed (Business activity)', 'Unemployed',
           'Professional Activity'], dtype=object)




```python
# Grouping the data into its each of its above constituent types

df_byProfession = df_bycode.groupby('Type')['Total'].sum()
df_Profession_Type = pd.DataFrame(df_byProfession).reset_index().sort_values('Total')
print(f'People who mentioned their Professions {df_bycode.shape[0]}')
```

    People who mentioned their Professions 17782
    


```python
# Drawing a pie chart of the data that we have grouped above

labels = df_Profession_Type['Type']
exp = (0,0,0,0,0,0,0,0.1,0.1,0.1)
plt.pie(df_Profession_Type['Total'], labels=labels, autopct='%1.1f%%', explode = exp, shadow=True)
plt.show()
```


    
![output_38_0](https://user-images.githubusercontent.com/50414959/112197548-388dc280-8c32-11eb-9d7d-9c4434c8cb81.png)

    



```python
#Drawing the same pie chart as a bargraph to see the absolute values

sns.barplot(x = df_Profession_Type['Total'], y=df_Profession_Type['Type'])
```




    <AxesSubplot:xlabel='Total', ylabel='Type'>




    
![output_39_1](https://user-images.githubusercontent.com/50414959/112197560-3af01c80-8c32-11eb-8049-69d8925a64f2.png)

    


From the above pie chart and bargraph we can conclude that more number of Suicides occur amongst house wives followed by those involved in Agricultural activities, Private service, people who are unemployed and the rest. Lowest Suicides are found among people who are in Government service and those retired. A conclusion that we might be able to draw her is that house wives are the most stressed people, and it makes sense since most of the house wives are not even acknowledged for their work, while some are even insulted and criticised. Only if house wives are given respect will the number of house wives committing suicide reduce. 

Farmers case is more delicate, farming is considered as a low key profession in India with farmers not having education, not being supported and in most cases are cheated. They work hard to feed the nation while we turn our backs upon them. Farmers committing suicide will not change unless governement and the people intervine to help the farmers and instead of looking down on the profession we must look up to the profession


```python
# Segregating the data where Type_code is Means_adopted

df_bycode = df[df['Type_code'] == 'Means_adopted']
df_bycode['Type'].unique()
```




    array(['By Hanging', 'By Fire/Self Immolation',
           'By Consuming Other Poison', 'By Drowning',
           'By Consuming Insecticides', 'By touching electric wires',
           'By Fire-Arms', 'By Overdose of sleeping pills',
           'By coming under running vehicles/trains',
           'By Jumping off Moving Vehicles/Trains',
           'By Jumping from (Other sites)', 'By Over Alcoholism',
           'By Jumping from (Building)', 'By Self Infliction of injury',
           'By Machine'], dtype=object)




```python
# Grouping the data into its each of its above constituent types

df_bymeans = df_bycode.groupby('Type')['Total'].sum()
df_means = pd.DataFrame(df_bymeans).reset_index().sort_values('Total')
print(f'Total number of People whose means of suicide is mentioned is {df_bycode.shape[0]}')
```

    Total number of People whose means of suicide is mentioned is 26709
    


```python
# Drawing a pie chart of the data that we have grouped above

labels = df_means['Type']
exp = [0]*(len(labels.unique())-3) + [0.1,0.1,0.1]
plt.pie(df_means['Total'], labels=labels, autopct='%1.1f%%', explode = exp, shadow=True)
plt.show()
```


    
![output_43_0](https://user-images.githubusercontent.com/50414959/112197593-43e0ee00-8c32-11eb-9c7a-c6916c2a1df3.png)

    



```python
#Drawing the same pie chart as a bargraph to see the absolute values

sns.barplot(x = df_means['Total'], y=df_means['Type'])
```




    <AxesSubplot:xlabel='Total', ylabel='Type'>




    
![output_44_1](https://user-images.githubusercontent.com/50414959/112197613-493e3880-8c32-11eb-94af-776a7a9e4c89.png)

    


We can see that most of the Suicides happen with the victim choosing hanging as the means of death, followed by consuming insecticides or other poison, it is then followed by fire/immolation, drowning with the least number of victims using Machines. 

Now we can see the hidden facts that the data has to offer.

Since most number of victims are house wives followed by those in Agricultural activities, the means are also outh to make sense. Hanging, consuming poison or burning is the easy way out for house wives and farmers since they are easily accessible to them given their profession. 


```python
# Segregating the data where Type_code is Education_Status

df_bycode = df[df['Type_code'] == 'Education_Status']
df_bycode['Type'].unique()
```




    array(['No Education', 'Middle', 'Primary', 'Matriculate/Secondary',
           'Graduate', 'Hr. Secondary/Intermediate/Pre-Universit',
           'Post Graduate and Above', 'Diploma'], dtype=object)




```python
# Grouping the data into its each of its above constituent types

df_byeducation = df_bycode.groupby('Type')['Total'].sum()
df_education = pd.DataFrame(df_byeducation).reset_index().sort_values('Total')
print(f'Total number of People whose Educational Status is mentioned is {df_bycode.shape[0]}')
```

    Total number of People whose Educational Status is mentioned is 5602
    


```python
# Drawing a pie chart of the data that we have grouped above

labels = df_education['Type']
plt.pie(df_education['Total'], labels=labels, autopct='%1.1f%%', explode = [0,0,0,0,0,0.1,0.1,0.1])
plt.show()
```


    
![output_48_0](https://user-images.githubusercontent.com/50414959/112197637-4e02ec80-8c32-11eb-89cb-54b4112fd6d0.png)

    



```python
#Drawing the same pie chart as a bargraph to see the absolute values

sns.barplot(x = df_education['Total'], y=df_education['Type'])
```




    <AxesSubplot:xlabel='Total', ylabel='Type'>




    
![output_49_1](https://user-images.githubusercontent.com/50414959/112197656-51967380-8c32-11eb-9e6b-c2c51f5a4438.png)

    


Most number of Suicides are seen among the people who have not completed their primary education, followed by those who have only completed till Secondary and the least cases are seen amongst postgraduates. This establishes a very important fact that education is vital in the society today.

Education has the potential to change peoples lives, helping them lead good lives.

Thus to reduce the number of Suicides the governement must focus on educating the people until their Secondary level.


```python
# Segregating the data where Type_code is Causes

df_bycode = df[df['Type_code'] == 'Causes']
df_bycode['Type'].unique()
```




    array(['Love Affairs', 'Other Prolonged Illness',
           'Failure in Examination', 'Family Problems',
           'Insanity/Mental Illness', 'Death of Dear Person', 'Unemployment',
           'Fall in Social Reputation', 'Suspected/Illicit Relation',
           'Cancellation/Non-Settlement of Marriage',
           'Not having Children (Barrenness/Impotency)', 'Poverty',
           'Professional/Career Problem', 'Paralysis',
           'Bankruptcy or Sudden change in Economic Status', 'Divorce',
           'Cancer', 'Property Dispute', 'Illness (Aids/STD)',
           'Physical Abuse (Rape/Incest Etc.)', 'Drug Abuse/Addiction',
           'Ideological Causes/Hero Worshipping', 'Dowry Dispute',
           'Illegitimate Pregnancy',
           'Not having Children (Barrenness/Impotency'], dtype=object)




```python
# Grouping the data into its each of its above constituent types

df_bycauses = df_bycode.groupby('Type')['Total'].sum()
df_causes = pd.DataFrame(df_bycauses).reset_index().sort_values('Total')
print(f'Total causes that have led to Suicide {df_bycode.shape[0]}')
```

    Total causes that have led to Suicide 33134
    


```python
# Drawing a pie chart of the data that we have grouped above

labels = df_causes['Type']
exp = [0]*(len(labels.unique())-3) + [0.1,0.1,0.1]
plt.pie(df_causes['Total'], labels=labels, autopct='%1.1f%%', explode = exp)
plt.show()
```


    
![output_53_0](https://user-images.githubusercontent.com/50414959/112197672-56f3be00-8c32-11eb-984f-0f86f2706b54.png)

    



```python
#Drawing the same pie chart as a bargraph to see the absolute values

sns.barplot(x = df_causes['Total'], y=df_causes['Type'])
```




    <AxesSubplot:xlabel='Total', ylabel='Type'>




    
![output_54_1](https://user-images.githubusercontent.com/50414959/112197703-5e1acc00-8c32-11eb-9fb6-e0c667621875.png)

    


Family problems is the singlest most important cause that forces people to commit suicide. Family is the support and strength for any individual, if they is not there then what do the people have to live and work for. 

We were able to arrive at a conclusion above that Housewives are the people who commit the most number of suicides and here we are able to conclude that most causes are Family Problems. So, being unhappy at house is the singlest huge factor that contributes to more number of cases among people. 

This can be solved by giving equal power to women in the house and by treating them with respect. Their contribution has to be acknowleged and appreciated for a health and happy family. This way we can also reduce the number of cases significantly.

#### Now lets analyse further by considering the state of Maharashtra which has the most number of suicides recorded.


```python
# Segregating the data of Maharashtra.

df_Maharashtra = df[df['State']=='Maharashtra']
df_Maharashtra
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>Year</th>
      <th>Type_code</th>
      <th>Type</th>
      <th>Gender</th>
      <th>Age_group</th>
      <th>Total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>135193</th>
      <td>Maharashtra</td>
      <td>2001</td>
      <td>Causes</td>
      <td>Insanity/Mental Illness</td>
      <td>Female</td>
      <td>0-14</td>
      <td>12</td>
    </tr>
    <tr>
      <th>135195</th>
      <td>Maharashtra</td>
      <td>2001</td>
      <td>Causes</td>
      <td>Failure in Examination</td>
      <td>Female</td>
      <td>0-14</td>
      <td>17</td>
    </tr>
    <tr>
      <th>135198</th>
      <td>Maharashtra</td>
      <td>2001</td>
      <td>Causes</td>
      <td>Family Problems</td>
      <td>Female</td>
      <td>0-14</td>
      <td>44</td>
    </tr>
    <tr>
      <th>135202</th>
      <td>Maharashtra</td>
      <td>2001</td>
      <td>Causes</td>
      <td>Death of Dear Person</td>
      <td>Female</td>
      <td>0-14</td>
      <td>1</td>
    </tr>
    <tr>
      <th>135203</th>
      <td>Maharashtra</td>
      <td>2001</td>
      <td>Causes</td>
      <td>Other Prolonged Illness</td>
      <td>Female</td>
      <td>0-14</td>
      <td>17</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>141977</th>
      <td>Maharashtra</td>
      <td>2012</td>
      <td>Social_Status</td>
      <td>Seperated</td>
      <td>Male</td>
      <td>0-100+</td>
      <td>30</td>
    </tr>
    <tr>
      <th>141978</th>
      <td>Maharashtra</td>
      <td>2012</td>
      <td>Social_Status</td>
      <td>Never Married</td>
      <td>Male</td>
      <td>0-100+</td>
      <td>2261</td>
    </tr>
    <tr>
      <th>141979</th>
      <td>Maharashtra</td>
      <td>2012</td>
      <td>Social_Status</td>
      <td>Divorcee</td>
      <td>Male</td>
      <td>0-100+</td>
      <td>29</td>
    </tr>
    <tr>
      <th>141980</th>
      <td>Maharashtra</td>
      <td>2012</td>
      <td>Social_Status</td>
      <td>Married</td>
      <td>Male</td>
      <td>0-100+</td>
      <td>8756</td>
    </tr>
    <tr>
      <th>141981</th>
      <td>Maharashtra</td>
      <td>2012</td>
      <td>Social_Status</td>
      <td>Widowed/Widower</td>
      <td>Male</td>
      <td>0-100+</td>
      <td>228</td>
    </tr>
  </tbody>
</table>
<p>4435 rows × 7 columns</p>
</div>




```python
# Segregating the data where Type_code is Social_Status

df_Maharashtra_Social = df_Maharashtra[df_Maharashtra['Type_code'] == 'Social_Status']
df_Maharashtra_Social.groupby('Type')['Total'].sum()
```




    Type
    Divorcee              892
    Married            134843
    Never Married       37539
    Seperated             958
    Widowed/Widower      6157
    Name: Total, dtype: int64




```python
# Grouping the data into its each of its above constituent types

df_Maharashtra_bySocial = df_Maharashtra_Social.groupby('Type')['Total'].sum()
df_Maharashtra_Social_Type = pd.DataFrame(df_Maharashtra_bySocial).reset_index().sort_values('Total')
print(f'Total people with Social Status mentioned in Maharashtra {df_Maharashtra_Social.shape[0]}')
```

    Total people with Social Status mentioned in Maharashtra 120
    


```python
# Drawing a bar graph of the data that we have grouped above

sns.barplot(x = df_Maharashtra_Social_Type['Total'], y=df_Maharashtra_Social_Type['Type'])
```




    <AxesSubplot:xlabel='Total', ylabel='Type'>




    
![output_60_1](https://user-images.githubusercontent.com/50414959/112197726-66730700-8c32-11eb-9de4-9d0b5e7e1917.png)

    


We can see that Married people are the ones who commit more suicide, reasons may be family problems and the fact that house wives commit the most suicides that adds up. But from what we see we can observe that the number of married people who commit suicide far outweigh the other factors thus establishing the fact that most of the married couples are generally unhappy for each other.


```python
# Segregating the data where Type_code is Professional_Profile

df_Maharashtra_Professional = df_Maharashtra[df_Maharashtra['Type_code'] == 'Professional_Profile']
df_Maharashtra_Professional.groupby('Type')['Total'].sum()
```




    Type
    Farming/Agriculture Activity         44769
    House Wife                           42059
    Professional Activity                 6665
    Public Sector Undertaking             1771
    Retired Person                        1727
    Self-employed (Business activity)     5932
    Service (Government)                  2913
    Service (Private)                    19754
    Student                              10441
    Unemployed                           11389
    Name: Total, dtype: int64




```python
# Grouping the data into its each of its above constituent types

df_Maharashtra_byProfessional = df_Maharashtra_Professional.groupby('Type')['Total'].sum()
df_Maharashtra_Professional_Type = pd.DataFrame(df_Maharashtra_byProfessional).reset_index().sort_values('Total')
print(f'Total people with Profession mentioned in Maharashtra {df_Maharashtra_Professional.shape[0]}')
```

    Total people with Profession mentioned in Maharashtra 822
    


```python
# Drawing a bar graph of the data that we have grouped above

sns.barplot(x = df_Maharashtra_Professional_Type['Total'], y=df_Maharashtra_Professional_Type['Type'])
```




    <AxesSubplot:xlabel='Total', ylabel='Type'>




    
![output_64_1](https://user-images.githubusercontent.com/50414959/112197757-6e32ab80-8c32-11eb-82d4-9b1c97757a36.png)

    


We can see that more people involved in Farming/Agriculture Activity end up committing suicide in the state of Maharashtra. Maharashtra is primarily a farming state with many people in the state involved directly or indirectly on agriculture. We have also concluded before that farmers are the most affected because farming is not seen as a reputed profession and so the statistics obtained from the data on Maharashtra are logically deducable. The state is a farming state and farming is not seen as a proper profession, and so more people who commit suicide are from this profession. 
Thus this state having the number one position is also logically deductable from the above reasoning.

This calls for the governement intervention to support farming and for us to start respecting farming as a nobel profession, and to also help out farmers to reduce the number of suicides.

This is then followed by house wives, even in the overall data we see that it's the house wives who commit the most number of suicides. This can only be prevented when the people start acknowledging the house wife and start respecting her for the work she does to keep the house livable and prosporous.


```python
# Segregating the data where Type_code is Means_adopted

df_Maharashtra_means = df_Maharashtra[df_Maharashtra['Type_code'] == 'Means_adopted']
df_Maharashtra_means.groupby('Type')['Total'].sum()
```




    Type
    By Consuming Insecticides                  55773
    By Consuming Other Poison                  13248
    By Drowning                                18923
    By Fire-Arms                                 157
    By Fire/Self Immolation                    22552
    By Hanging                                 61651
    By Jumping from (Building)                  1164
    By Jumping from (Other sites)                362
    By Jumping off Moving Vehicles/Trains        550
    By Machine                                    21
    By Over Alcoholism                          2331
    By Overdose of sleeping pills                325
    By Self Infliction of injury                 414
    By coming under running vehicles/trains     2565
    By touching electric wires                   213
    Name: Total, dtype: int64




```python
# Grouping the data into its each of its above constituent types

df_Maharashtra_bymeans = df_Maharashtra_means.groupby('Type')['Total'].sum()
df_Maharashtra_means_Type = pd.DataFrame(df_Maharashtra_bymeans).reset_index().sort_values('Total')
print(f'Total people with means of Suicide mentioned in Maharashtra {df_Maharashtra_means.shape[0]}')
```

    Total people with means of Suicide mentioned in Maharashtra 1306
    


```python
# Drawing a bar graph of the data that we have grouped above

sns.barplot(x = df_Maharashtra_means_Type['Total'], y=df_Maharashtra_means_Type['Type'])
```




    <AxesSubplot:xlabel='Total', ylabel='Type'>




    
![output_68_1](https://user-images.githubusercontent.com/50414959/112197781-74c12300-8c32-11eb-85df-d2e7002c9436.png)

    


We can see that hanging and consuming insecticides are the most common means of committing suicide in Maharashtra. This can be arrived at by the fact that most of the suicides are committed by people in the farming/agriculture and by house wives, since insecticides and rope are two of the most common things available for house wives and farmers. There have been many instances in news too about how a farmer killed himself by drinking insecticide.


```python
# Segregating the data where Type_code is Education_Status

df_Maharashtra_Education = df_Maharashtra[df_Maharashtra['Type_code'] == 'Education_Status']
df_Maharashtra_Education.groupby('Type')['Total'].sum()
```




    Type
    Diploma                                      1377
    Graduate                                     2146
    Hr. Secondary/Intermediate/Pre-Universit    11715
    Matriculate/Secondary                       41304
    Middle                                      47927
    No Education                                27088
    Post Graduate and Above                       358
    Primary                                     48474
    Name: Total, dtype: int64




```python
# Grouping the data into its each of its above constituent types

df_Maharashtra_byEducation = df_Maharashtra_Education.groupby('Type')['Total'].sum()
df_Maharashtra_Education_Type = pd.DataFrame(df_Maharashtra_byEducation).reset_index().sort_values('Total')
print(f'Total people whose Education Qualification is mentioned in Maharashtra is {df_Maharashtra_Education.shape[0]}')
```

    Total people whose Education Qualification is mentioned in Maharashtra is 192
    


```python
# Drawing a bar graph of the data that we have grouped above

sns.barplot(x = df_Maharashtra_Education_Type['Total'], y=df_Maharashtra_Education_Type['Type'])
```




    <AxesSubplot:xlabel='Total', ylabel='Type'>




    
![output_72_1](https://user-images.githubusercontent.com/50414959/112197804-7ab70400-8c32-11eb-8b6a-46c1cd9fe125.png)

    


We can see that most number of people who committed suicide are people who have not completed their secondary education, we can see a significant drop in the number of suicides by people who have completed their secondary education and have gone beyond it. Maharashtra being a farming state primarily, tends to have more rural areas than urban areas and higher education is usually not given much importance in rural areas, with students helping their parents in farming rather than studying. They are also usually married off young and end up farming. 

So, it is rightly said that education can help in the upliftment of the nation. The people need to be educated, since most educated people (ones who study beyond their secondary education) do not tend to commit suicide easily. 

So, governement and the families in these areas need to educate their children for the better.


```python
# Segregating the data where Type_code is Causes

df_Maharashtra_Causes = df_Maharashtra[df_Maharashtra['Type_code'] == 'Causes']
df_Maharashtra_Causes.groupby('Type')['Total'].sum()
```




    Type
    Bankruptcy or Sudden change in Economic Status     7099
    Cancellation/Non-Settlement of Marriage            1714
    Cancer                                             1237
    Death of Dear Person                               1585
    Divorce                                             376
    Dowry Dispute                                      3391
    Drug Abuse/Addiction                              12671
    Failure in Examination                             3284
    Fall in Social Reputation                          1370
    Family Problems                                   65341
    Ideological Causes/Hero Worshipping                 108
    Illegitimate Pregnancy                              121
    Illness (Aids/STD)                                 1268
    Insanity/Mental Illness                           14859
    Love Affairs                                       2872
    Not having Children (Barrenness/Impotency           120
    Not having Children (Barrenness/Impotency)         1251
    Other Prolonged Illness                           33808
    Paralysis                                          1069
    Physical Abuse (Rape/Incest Etc.)                   394
    Poverty                                            4083
    Professional/Career Problem                        3019
    Property Dispute                                    818
    Suspected/Illicit Relation                          813
    Unemployment                                       4493
    Name: Total, dtype: int64




```python
# Grouping the data into its each of its above constituent types

df_Maharashtra_byCauses = df_Maharashtra_Causes.groupby('Type')['Total'].sum()
df_Maharashtra_Causes_Type = pd.DataFrame(df_Maharashtra_byCauses).reset_index().sort_values('Total')
print(f'Total people in Maharashtra whose cause of Suicide is mentioned is {df_Maharashtra_Causes.shape[0]}')
```

    Total people in Maharashtra whose cause of Suicide is mentioned is 1995
    


```python
# Drawing a bar graph of the data that we have grouped above

sns.barplot(x = df_Maharashtra_Causes_Type['Total'], y=df_Maharashtra_Causes_Type['Type'])
```




    <AxesSubplot:xlabel='Total', ylabel='Type'>




    
![output_76_1](https://user-images.githubusercontent.com/50414959/112197828-81457b80-8c32-11eb-99f4-a5fbadcaeb8f.png)

    


We can see that the major cause of suicide is attributed to Family problems. Lack of proper education, getting married at a young age and lack of amenities tends to cause rifts in families, usually these problems can be solved but then suicide is chosen as the easy way out, and its the women(the house wives) who usually commit suicide as we saw above. 

So, it can be said educating a female child and saying no to marriage at an young age can significantly reduce the number of suicides. This is where the indepth analysis helps us, see how we are able to start forming logical conclusions and are able to target the places that matter and give the highest results the most instead of blindly trying some measures and hoping that they would help.

#### Now lets do an analysis on the state of Tamil Nadu which is the state with the 4th highest number of suicides recorded.


```python
# Segregating the data of Tamil Nadu.

df_TamilNadu = df[df['State']=='Tamil Nadu']
df_TamilNadu
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>Year</th>
      <th>Type_code</th>
      <th>Type</th>
      <th>Gender</th>
      <th>Age_group</th>
      <th>Total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>202690</th>
      <td>Tamil Nadu</td>
      <td>2001</td>
      <td>Causes</td>
      <td>Death of Dear Person</td>
      <td>Female</td>
      <td>0-14</td>
      <td>2</td>
    </tr>
    <tr>
      <th>202696</th>
      <td>Tamil Nadu</td>
      <td>2001</td>
      <td>Causes</td>
      <td>Poverty</td>
      <td>Female</td>
      <td>0-14</td>
      <td>4</td>
    </tr>
    <tr>
      <th>202701</th>
      <td>Tamil Nadu</td>
      <td>2001</td>
      <td>Causes</td>
      <td>Failure in Examination</td>
      <td>Female</td>
      <td>0-14</td>
      <td>19</td>
    </tr>
    <tr>
      <th>202702</th>
      <td>Tamil Nadu</td>
      <td>2001</td>
      <td>Causes</td>
      <td>Other Prolonged Illness</td>
      <td>Female</td>
      <td>0-14</td>
      <td>5</td>
    </tr>
    <tr>
      <th>202703</th>
      <td>Tamil Nadu</td>
      <td>2001</td>
      <td>Causes</td>
      <td>Fall in Social Reputation</td>
      <td>Female</td>
      <td>0-14</td>
      <td>2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>209471</th>
      <td>Tamil Nadu</td>
      <td>2012</td>
      <td>Social_Status</td>
      <td>Married</td>
      <td>Male</td>
      <td>0-100+</td>
      <td>8122</td>
    </tr>
    <tr>
      <th>209472</th>
      <td>Tamil Nadu</td>
      <td>2012</td>
      <td>Social_Status</td>
      <td>Widowed/Widower</td>
      <td>Male</td>
      <td>0-100+</td>
      <td>311</td>
    </tr>
    <tr>
      <th>209473</th>
      <td>Tamil Nadu</td>
      <td>2012</td>
      <td>Social_Status</td>
      <td>Divorcee</td>
      <td>Male</td>
      <td>0-100+</td>
      <td>39</td>
    </tr>
    <tr>
      <th>209474</th>
      <td>Tamil Nadu</td>
      <td>2012</td>
      <td>Social_Status</td>
      <td>Seperated</td>
      <td>Male</td>
      <td>0-100+</td>
      <td>662</td>
    </tr>
    <tr>
      <th>209475</th>
      <td>Tamil Nadu</td>
      <td>2012</td>
      <td>Social_Status</td>
      <td>Never Married</td>
      <td>Male</td>
      <td>0-100+</td>
      <td>1614</td>
    </tr>
  </tbody>
</table>
<p>4477 rows × 7 columns</p>
</div>




```python
# Segregating the data where Type_code is Social_Status

df_TamilNadu_Social = df_TamilNadu[df_TamilNadu['Type_code'] == 'Social_Status']
df_TamilNadu_Social.groupby('Type')['Total'].sum()
```




    Type
    Divorcee             1777
    Married            115005
    Never Married       29005
    Seperated            9512
    Widowed/Widower      8514
    Name: Total, dtype: int64




```python
# Grouping the data into its each of its above constituent types

df_TamilNadu_bySocial = df_TamilNadu_Social.groupby('Type')['Total'].sum()
df_TamilNadu_Social_Type = pd.DataFrame(df_TamilNadu_bySocial).reset_index().sort_values('Total')
print(f"Total number of people in Tamil Nadu whose Social Status is mentioned {df_TamilNadu_Social.shape[0]}")
```

    Total number of people in Tamil Nadu whose Social Status is mentioned 120
    


```python
# Drawing a bar graph of the data that we have grouped above

sns.barplot(y=df_TamilNadu_Social_Type['Type'],x=df_TamilNadu_Social_Type['Total'])
```




    <AxesSubplot:xlabel='Total', ylabel='Type'>




    
![output_82_1](https://user-images.githubusercontent.com/50414959/112197857-89052000-8c32-11eb-8d07-80f13e65a98c.png)

    


We can see that more number of people who commit suicide are married and the reason for this has already been discussed in the analysis performed few blocks above in the analysis performed for the state of Maharashtra.


```python
# Segregating the data where Type_code is Professional_Profile

df_TamilNadu_Professional = df_TamilNadu[df_TamilNadu['Type_code'] == 'Professional_Profile']
df_TamilNadu_Professional.groupby('Type')['Total'].sum()
```




    Type
    Farming/Agriculture Activity         10491
    House Wife                           28165
    Professional Activity                 3407
    Public Sector Undertaking             3864
    Retired Person                        1239
    Self-employed (Business activity)    11650
    Service (Government)                  2825
    Service (Private)                    17159
    Student                               5818
    Unemployed                           20770
    Name: Total, dtype: int64




```python
# Grouping the data into its each of its above constituent types

df_TamilNadu_byProfessional = df_TamilNadu_Professional.groupby('Type')['Total'].sum()
df_TamilNadu_Professional_Type = pd.DataFrame(df_TamilNadu_byProfessional).reset_index().sort_values('Total')
print(f"Total number of people in Tamil Nadu whose Profession is mentioned is {df_TamilNadu_Professional.shape[0]}")
```

    Total number of people in Tamil Nadu whose Profession is mentioned is 860
    


```python
# Drawing a bar graph of the data that we have grouped above

sns.barplot(x=df_TamilNadu_Professional_Type['Total'], y= df_TamilNadu_Professional_Type['Type'])
```




    <AxesSubplot:xlabel='Total', ylabel='Type'>




    
![output_86_1](https://user-images.githubusercontent.com/50414959/112197877-8e626a80-8c32-11eb-8724-255ae0b484b8.png)

    


We can see that more number of people who commit suicide are House wives and the reason for this has already been discussed in the analysis performed few blocks above in the analysis performed for the state of Maharashtra.


```python
# Segregating the data where Type_code is Means_adopted

df_TamilNadu_means = df_TamilNadu[df_TamilNadu['Type_code'] == 'Means_adopted']
df_TamilNadu_means.groupby('Type')['Total'].sum()
```




    Type
    By Consuming Insecticides                  23335
    By Consuming Other Poison                  35531
    By Drowning                                10638
    By Fire-Arms                                 144
    By Fire/Self Immolation                    23089
    By Hanging                                 39880
    By Jumping from (Building)                   966
    By Jumping from (Other sites)               1273
    By Jumping off Moving Vehicles/Trains        789
    By Machine                                   287
    By Over Alcoholism                          1854
    By Overdose of sleeping pills               1198
    By Self Infliction of injury                 391
    By coming under running vehicles/trains     2759
    By touching electric wires                  1341
    Name: Total, dtype: int64




```python
# Grouping the data into its each of its above constituent types

df_TamilNadu_bymeans = df_TamilNadu_means.groupby('Type')['Total'].sum()
df_TamilNadu_means_Type = pd.DataFrame(df_TamilNadu_bymeans).reset_index().sort_values('Total')
print(f"Total number of people in Tamil Nadu whose means of Suicide is mentioned is {df_TamilNadu_means.shape[0]}")
```

    Total number of people in Tamil Nadu whose means of Suicide is mentioned is 1374
    


```python
# Drawing a bar graph of the data that we have grouped above

sns.barplot(x=df_TamilNadu_means_Type['Total'], y= df_TamilNadu_means_Type['Type'])
```




    <AxesSubplot:xlabel='Total', ylabel='Type'>




    
![output_90_1](https://user-images.githubusercontent.com/50414959/112197904-94584b80-8c32-11eb-9a34-6cedfcba6b98.png)

    


We can see that Hanging is the most common means of committing suicide followed by consuming poison and insecticides. We can see in Tamil Nadu that committing suicide by Fire/self immolation is the 4th highest means people use to commit suicide and based on this, one of the conclusions that we can arrive at is that given the most number of suicides are committed by housewives, who are obviously married(thereby contibuting to marriage being in the top), it makes sense that choosing fire/self immolation is in the 4th place, since for housewives in Tamil Nadu fire and kerosene are easily accessible along with chemicals and there are also a lot of news articles that say that victim has burned themselves to death. This is only seen in Tamil Nadu while for the rest of the states Fire/self immolation tends to be the least used method.


```python
# Segregating the data where Type_code is Education_Status

df_TamilNadu_Education = df_TamilNadu[df_TamilNadu['Type_code'] == 'Education_Status']
df_TamilNadu_Education.groupby('Type')['Total'].sum()
```




    Type
    Diploma                                      3348
    Graduate                                     4633
    Hr. Secondary/Intermediate/Pre-Universit    16244
    Matriculate/Secondary                       22762
    Middle                                      41180
    No Education                                39499
    Post Graduate and Above                      1393
    Primary                                     34754
    Name: Total, dtype: int64




```python
# Grouping the data into its each of its above constituent types

df_TamilNadu_byEducation = df_TamilNadu_Education.groupby('Type')['Total'].sum()
df_TamilNadu_Education_Type = pd.DataFrame(df_TamilNadu_byEducation).reset_index().sort_values('Total')
print(f"Total number of people in Tamil Nadu whose Education Qualification is mentioned is {df_TamilNadu_Education.shape[0]}")
```

    Total number of people in Tamil Nadu whose Education Qualification is mentioned is 192
    


```python
# Drawing a bar graph of the data that we have grouped above

sns.barplot(x=df_TamilNadu_Education_Type['Total'], y= df_TamilNadu_Education_Type['Type'])
```




    <AxesSubplot:xlabel='Total', ylabel='Type'>




    
![output_94_1](https://user-images.githubusercontent.com/50414959/112197923-991cff80-8c32-11eb-9471-832bffc402b3.png)

    


The analysis we can make here is almost the same as that of Maharashtra, which can be referred from the blocks above.


```python
# Segregating the data where Type_code is Causes

df_TamilNadu_causes = df_TamilNadu[df_TamilNadu['Type_code'] == 'Causes']
df_TamilNadu_causes.groupby('Type')['Total'].sum()
```




    Type
    Bankruptcy or Sudden change in Economic Status     1753
    Cancellation/Non-Settlement of Marriage            1031
    Cancer                                              368
    Death of Dear Person                                994
    Divorce                                             593
    Dowry Dispute                                      1234
    Drug Abuse/Addiction                               1952
    Failure in Examination                             3011
    Fall in Social Reputation                          1789
    Family Problems                                   49663
    Ideological Causes/Hero Worshipping                 575
    Illegitimate Pregnancy                              253
    Illness (Aids/STD)                                  206
    Insanity/Mental Illness                            4974
    Love Affairs                                       5684
    Not having Children (Barrenness/Impotency            78
    Not having Children (Barrenness/Impotency)          916
    Other Prolonged Illness                           31304
    Paralysis                                          1034
    Physical Abuse (Rape/Incest Etc.)                   248
    Poverty                                            3188
    Professional/Career Problem                        1374
    Property Dispute                                   2552
    Suspected/Illicit Relation                         1523
    Unemployment                                       4171
    Name: Total, dtype: int64




```python
# Grouping the data into its each of its above constituent types

df_TamilNadu_bycauses = df_TamilNadu_causes.groupby('Type')['Total'].sum()
df_TamilNadu_causes_Type = pd.DataFrame(df_TamilNadu_bycauses).reset_index().sort_values('Total')
print(f"Total number of people in Tamil Nadu whose cause of Suicide is mentioned is {df_TamilNadu_causes.shape[0]}")
```

    Total number of people in Tamil Nadu whose cause of Suicide is mentioned is 1931
    


```python
# Drawing a bar graph of the data that we have grouped above

sns.barplot(x=df_TamilNadu_causes_Type['Total'], y= df_TamilNadu_causes_Type['Type'])
```




    <AxesSubplot:xlabel='Total', ylabel='Type'>




    
![output_98_1](https://user-images.githubusercontent.com/50414959/112197942-9f12e080-8c32-11eb-86d7-2ca5ef37c7b6.png)

    


Family Problems seems to be the most common problem everywhere since almost all states have it as their primary or secondary cause. But in case of Tamil Nadu we can see that suicides due to prolonged illness tends to take the 2nd place and Love Affairs tends to take the 3rd place. We can conclude by either saying that people of Tamil Nadu are more prone to prolonged illness or the medical facilties might not be well enough and to conclude upon a point we would require more in depth data, which may or may not be available. We can see this as a problem with only a few states other than Tamil Nadu. 

To summarize we first begun by studying the data and then preprocessing it to clean it of all the possible errors, remove the unwanted data, merge or remove the redundant data which might end up giving wrong results and therfore lead to erroneous analysis. 

We then went ahead with first drawing a conculsion on which gender has the most cases and explored a bit of reasoning as to why we see males commit more suicides, then we analysed as to which age group commits the most suicides, from there we combined both the gender analysis and age group analysis, thereby making a few conculsions over there, then we made a statewise comparision where we saw Maharashtra topping the number of cases list.

We then saw each type individually and then went ahead for an indepth analysis where we analysed based on the suicide causes, means used, the profession of the people, their education status and their social status where we were able to draw some good conclusion that can be used to effectively reduce the number of suicides. This was we are able to target the places which give us really good results instead of blindly targetting without proper plan and retargetting by changing the plan based on the observed results, which can be a tiresome and cumbersome process.

We then, as a further example, extended our analysis to two states, i.e Maharashtra (1st in the number of cases) followed by Tamil Nadu (4th in the number of cases). We were able to draw some intresting insights from these analysis too.

Thus, we have performed a indepth analysis of the suicide dataset and have used the python packages to help us represent the data visually using bar graphs and pie charts, visual form of representation is always appriciated since it helps us understand and perform analysis easily. 

If you are interested you can always find more ways to interpret the data and gain more insights. Feel free to explore beyond what I have done.

Hope you understood the procedure and the functions, if you were not able to understand some of the functions you can always copy the function and search it in google for a detailed information. 

Incase you spot a mistake, am sorry, and feel free to correct it.

Thank you for reading this.

Keep Learning and have fun.
