#!/usr/bin/env python
# coding: utf-8

# Analysis of Suicides in India

# importing the required libraries 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Read the Suicide data '.csv' and check if it is has been read properly

df = pd.read_csv('Suicides in India 2001-2012.csv')
df.head()

# See the common statistics of the data

df.describe()

# See the basic description of the variables, their type and value count

df.info()

# print the all the states available in the dataset

print(df['State'].unique())

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

# Here we remove all the values whose exact causes are not known, this is to ensure that the data whose cause is not known does not
# end up affecting our analysis.

df = df.drop(df[(df.State == 'Total (Uts)') | (df.State == 'Total (All India)') | (df.State =='Total (States)')].index)
df = df.drop(df[(df.Type == 'By Other means') | (df.Type =='Other Causes (Please Specity)') | (df.Type =='Other Causes (Please Specify)') | (df.Type =='Causes Not known') | (df.Type =='By Other means (please specify)')].index)
df = df.drop(df[df['Total'] == 0].index)

#Plot a bargraph based on the gender to see which gender people commit the most suicides

sns.set(rc={'figure.figsize':(10,8)})
sns.barplot(x=df['Gender'], y=df['Total'])

# Draw the same bargraph as a pie chart to see the percentage stats

df_gender = df.groupby('Gender')['Total'].sum()
df_gender_type = pd.DataFrame(df_gender).reset_index().sort_values('Total')
labels = df_gender_type['Gender']
exp = (0,0.1)
plt.pie(df_gender_type['Total'], labels=labels, autopct='%1.1f%%', explode = exp, shadow=True)
plt.show()

# Here we are spliting the data into different age groups

gp_age = df.groupby('Age_group')['Total'].sum()
df_age_type = pd.DataFrame(gp_age).reset_index().sort_values('Age_group')
df_age_type = df_age_type.drop([0])
df_age_type

# Draw a bargraph categorising the people into different age groups

sns.barplot(x = df_age_type['Age_group'], y = df_age_type['Total'])

# Draw the same bargraph as a pie chart to see the percentage stats

labels = df_age_type['Age_group']
exp = (0,0.1,0.1,0,0)
plt.pie(df_age_type['Total'], labels=labels, autopct='%1.1f%%', explode = exp, shadow=True)
plt.show()

# Here we are categorising the people into different age groups and proceed to futher 
# split them on the basis of their gender

df_age = df.drop(df[(df['Age_group'] == '0-100+')].index)
df_age = df_age.groupby(['Age_group', 'Gender'])['Total'].sum()
df_age_gender = pd.DataFrame(df_age).reset_index().sort_values('Age_group')
df_age_gender

# Draw a bargraph for the above data that we had split

sns.barplot(x = "Age_group", y = "Total", hue = "Gender", data=df_age_gender)

# Here we are grouping the data statewise so as to see which state has the most cases.

group = df.groupby('State')['Total'].sum()
tot_suicides = pd.DataFrame(group).reset_index().sort_values('Total',ascending=False)

# Draw a bargraph categorising the data into statewise form, which is obtained from above

fig, ax = plt.subplots(figsize=(18,6))
plot = sns.barplot(x = 'State', y = 'Total', data = tot_suicides)
plot.set_xticklabels(plot.get_xticklabels(), rotation = 45)
plt.show()

# Here we are grouping the data yearwise so as to see which year has the most cases.

group_yr = df.groupby('Year')['Total'].sum()
yr = pd.DataFrame(group_yr).reset_index().sort_values('Year',ascending=False)
print(group_yr)

# Draw a bargraph categorising the data into yearwise form, which is obtained from above

plot = sns.barplot(x = 'Year', y = 'Total', data = yr, palette = 'flare')

# Here we are grouping the data based on the Type code which consists of all the factors 
# that we wish to analyse, as mentioned above.

group_count = df['Type_code'].value_counts()
group_count

# Draw a bargraph of the data that we have grouped above

sns.set(rc={'figure.figsize':(12,9)})
sns.countplot(df['Type_code'])

# Segregating the data where Type_code is Social_Status 

df_bycode = df[df['Type_code'] == 'Social_Status']
df_bycode['Type'].unique()

# Grouping the data into its each of its constituent types

df_social = df_bycode.groupby('Type')['Total'].sum()
df_social_type = pd.DataFrame(df_social).reset_index().sort_values('Total')
print(f'People who have mentioned their Social Status {df_bycode.shape[0]}')

# Drawing a pie chart of the data that we have grouped above

labels = df_social_type['Type']
exp = (0,0,0,0,0.1)
plt.pie(df_social_type['Total'], labels=labels, autopct='%1.1f%%', explode = exp, shadow=True)
plt.show()

#Drawing the same pie chart as a bargraph to see the absolute values

sns.barplot(x = df_social_type['Total'], y=df_social_type['Type'])

# Segregating the data where Type_code is Professional_Profile

df_bycode = df[df['Type_code'] == 'Professional_Profile']
df_bycode['Type'].unique()

# Grouping the data into its each of its above constituent types

df_byProfession = df_bycode.groupby('Type')['Total'].sum()
df_Profession_Type = pd.DataFrame(df_byProfession).reset_index().sort_values('Total')
print(f'People who mentioned their Professions {df_bycode.shape[0]}')

# Drawing a pie chart of the data that we have grouped above

labels = df_Profession_Type['Type']
exp = (0,0,0,0,0,0,0,0.1,0.1,0.1)
plt.pie(df_Profession_Type['Total'], labels=labels, autopct='%1.1f%%', explode = exp, shadow=True)
plt.show()

#Drawing the same pie chart as a bargraph to see the absolute values

sns.barplot(x = df_Profession_Type['Total'], y=df_Profession_Type['Type'])

# Segregating the data where Type_code is Means_adopted

df_bycode = df[df['Type_code'] == 'Means_adopted']
df_bycode['Type'].unique()

# Grouping the data into its each of its above constituent types

df_bymeans = df_bycode.groupby('Type')['Total'].sum()
df_means = pd.DataFrame(df_bymeans).reset_index().sort_values('Total')
print(f'Total number of People whose means of suicide is mentioned is {df_bycode.shape[0]}')

# Drawing a pie chart of the data that we have grouped above

labels = df_means['Type']
exp = [0]*(len(labels.unique())-3) + [0.1,0.1,0.1]
plt.pie(df_means['Total'], labels=labels, autopct='%1.1f%%', explode = exp, shadow=True)
plt.show()

#Drawing the same pie chart as a bargraph to see the absolute values

sns.barplot(x = df_means['Total'], y=df_means['Type'])

# Segregating the data where Type_code is Education_Status

df_bycode = df[df['Type_code'] == 'Education_Status']
df_bycode['Type'].unique()

# Grouping the data into its each of its above constituent types

df_byeducation = df_bycode.groupby('Type')['Total'].sum()
df_education = pd.DataFrame(df_byeducation).reset_index().sort_values('Total')
print(f'Total number of People whose Educational Status is mentioned is {df_bycode.shape[0]}')

# Drawing a pie chart of the data that we have grouped above

labels = df_education['Type']
plt.pie(df_education['Total'], labels=labels, autopct='%1.1f%%', explode = [0,0,0,0,0,0.1,0.1,0.1])
plt.show()

#Drawing the same pie chart as a bargraph to see the absolute values

sns.barplot(x = df_education['Total'], y=df_education['Type'])

# Segregating the data where Type_code is Causes

df_bycode = df[df['Type_code'] == 'Causes']
df_bycode['Type'].unique()

# Grouping the data into its each of its above constituent types

df_bycauses = df_bycode.groupby('Type')['Total'].sum()
df_causes = pd.DataFrame(df_bycauses).reset_index().sort_values('Total')
print(f'Total causes that have led to Suicide {df_bycode.shape[0]}')

# Drawing a pie chart of the data that we have grouped above

labels = df_causes['Type']
exp = [0]*(len(labels.unique())-3) + [0.1,0.1,0.1]
plt.pie(df_causes['Total'], labels=labels, autopct='%1.1f%%', explode = exp)
plt.show()

#Drawing the same pie chart as a bargraph to see the absolute values

sns.barplot(x = df_causes['Total'], y=df_causes['Type'])

# Analysing the data of the state of Maharashtra

# Segregating the data of Maharashtra.

df_Maharashtra = df[df['State']=='Maharashtra']
df_Maharashtra

# Segregating the data where Type_code is Social_Status

df_Maharashtra_Social = df_Maharashtra[df_Maharashtra['Type_code'] == 'Social_Status']
df_Maharashtra_Social.groupby('Type')['Total'].sum()

# Grouping the data into its each of its above constituent types

df_Maharashtra_bySocial = df_Maharashtra_Social.groupby('Type')['Total'].sum()
df_Maharashtra_Social_Type = pd.DataFrame(df_Maharashtra_bySocial).reset_index().sort_values('Total')
print(f'Total people with Social Status mentioned in Maharashtra {df_Maharashtra_Social.shape[0]}')

# Drawing a bar graph of the data that we have grouped above

sns.barplot(x = df_Maharashtra_Social_Type['Total'], y=df_Maharashtra_Social_Type['Type'])

# Segregating the data where Type_code is Professional_Profile

df_Maharashtra_Professional = df_Maharashtra[df_Maharashtra['Type_code'] == 'Professional_Profile']
df_Maharashtra_Professional.groupby('Type')['Total'].sum()

# Grouping the data into its each of its above constituent types

df_Maharashtra_byProfessional = df_Maharashtra_Professional.groupby('Type')['Total'].sum()
df_Maharashtra_Professional_Type = pd.DataFrame(df_Maharashtra_byProfessional).reset_index().sort_values('Total')
print(f'Total people with Profession mentioned in Maharashtra {df_Maharashtra_Professional.shape[0]}')

# Drawing a bar graph of the data that we have grouped above

sns.barplot(x = df_Maharashtra_Professional_Type['Total'], y=df_Maharashtra_Professional_Type['Type'])

# Segregating the data where Type_code is Means_adopted

df_Maharashtra_means = df_Maharashtra[df_Maharashtra['Type_code'] == 'Means_adopted']
df_Maharashtra_means.groupby('Type')['Total'].sum()

# Grouping the data into its each of its above constituent types

df_Maharashtra_bymeans = df_Maharashtra_means.groupby('Type')['Total'].sum()
df_Maharashtra_means_Type = pd.DataFrame(df_Maharashtra_bymeans).reset_index().sort_values('Total')
print(f'Total people with means of Suicide mentioned in Maharashtra {df_Maharashtra_means.shape[0]}')

# Drawing a bar graph of the data that we have grouped above

sns.barplot(x = df_Maharashtra_means_Type['Total'], y=df_Maharashtra_means_Type['Type'])

# Segregating the data where Type_code is Education_Status

df_Maharashtra_Education = df_Maharashtra[df_Maharashtra['Type_code'] == 'Education_Status']
df_Maharashtra_Education.groupby('Type')['Total'].sum()

# Grouping the data into its each of its above constituent types

df_Maharashtra_byEducation = df_Maharashtra_Education.groupby('Type')['Total'].sum()
df_Maharashtra_Education_Type = pd.DataFrame(df_Maharashtra_byEducation).reset_index().sort_values('Total')
print(f'Total people whose Education Qualification is mentioned in Maharashtra is {df_Maharashtra_Education.shape[0]}')

# Drawing a bar graph of the data that we have grouped above

sns.barplot(x = df_Maharashtra_Education_Type['Total'], y=df_Maharashtra_Education_Type['Type'])

# Segregating the data where Type_code is Causes

df_Maharashtra_Causes = df_Maharashtra[df_Maharashtra['Type_code'] == 'Causes']
df_Maharashtra_Causes.groupby('Type')['Total'].sum()

# Grouping the data into its each of its above constituent types

df_Maharashtra_byCauses = df_Maharashtra_Causes.groupby('Type')['Total'].sum()
df_Maharashtra_Causes_Type = pd.DataFrame(df_Maharashtra_byCauses).reset_index().sort_values('Total')
print(f'Total people in Maharashtra whose cause of Suicide is mentioned is {df_Maharashtra_Causes.shape[0]}')

# Drawing a bar graph of the data that we have grouped above

sns.barplot(x = df_Maharashtra_Causes_Type['Total'], y=df_Maharashtra_Causes_Type['Type'])

# Analysing the data of the state of Tamil Nadu

# Segregating the data of Tamil Nadu.

df_TamilNadu = df[df['State']=='Tamil Nadu']
df_TamilNadu

# Segregating the data where Type_code is Social_Status

df_TamilNadu_Social = df_TamilNadu[df_TamilNadu['Type_code'] == 'Social_Status']
df_TamilNadu_Social.groupby('Type')['Total'].sum()

# Grouping the data into its each of its above constituent types

df_TamilNadu_bySocial = df_TamilNadu_Social.groupby('Type')['Total'].sum()
df_TamilNadu_Social_Type = pd.DataFrame(df_TamilNadu_bySocial).reset_index().sort_values('Total')
print(f"Total number of people in Tamil Nadu whose Social Status is mentioned {df_TamilNadu_Social.shape[0]}")

# Drawing a bar graph of the data that we have grouped above

sns.barplot(y=df_TamilNadu_Social_Type['Type'],x=df_TamilNadu_Social_Type['Total'])

# Segregating the data where Type_code is Professional_Profile

df_TamilNadu_Professional = df_TamilNadu[df_TamilNadu['Type_code'] == 'Professional_Profile']
df_TamilNadu_Professional.groupby('Type')['Total'].sum()

# Grouping the data into its each of its above constituent types

df_TamilNadu_byProfessional = df_TamilNadu_Professional.groupby('Type')['Total'].sum()
df_TamilNadu_Professional_Type = pd.DataFrame(df_TamilNadu_byProfessional).reset_index().sort_values('Total')
print(f"Total number of people in Tamil Nadu whose Profession is mentioned is {df_TamilNadu_Professional.shape[0]}")

# Drawing a bar graph of the data that we have grouped above

sns.barplot(x=df_TamilNadu_Professional_Type['Total'], y= df_TamilNadu_Professional_Type['Type'])

# Segregating the data where Type_code is Means_adopted

df_TamilNadu_means = df_TamilNadu[df_TamilNadu['Type_code'] == 'Means_adopted']
df_TamilNadu_means.groupby('Type')['Total'].sum()

# Grouping the data into its each of its above constituent types

df_TamilNadu_bymeans = df_TamilNadu_means.groupby('Type')['Total'].sum()
df_TamilNadu_means_Type = pd.DataFrame(df_TamilNadu_bymeans).reset_index().sort_values('Total')
print(f"Total number of people in Tamil Nadu whose means of Suicide is mentioned is {df_TamilNadu_means.shape[0]}")

# Drawing a bar graph of the data that we have grouped above

sns.barplot(x=df_TamilNadu_means_Type['Total'], y= df_TamilNadu_means_Type['Type'])

# Segregating the data where Type_code is Education_Status

df_TamilNadu_Education = df_TamilNadu[df_TamilNadu['Type_code'] == 'Education_Status']
df_TamilNadu_Education.groupby('Type')['Total'].sum()

# Grouping the data into its each of its above constituent types

df_TamilNadu_byEducation = df_TamilNadu_Education.groupby('Type')['Total'].sum()
df_TamilNadu_Education_Type = pd.DataFrame(df_TamilNadu_byEducation).reset_index().sort_values('Total')
print(f"Total number of people in Tamil Nadu whose Education Qualification is mentioned is {df_TamilNadu_Education.shape[0]}")

# Drawing a bar graph of the data that we have grouped above

sns.barplot(x=df_TamilNadu_Education_Type['Total'], y= df_TamilNadu_Education_Type['Type'])

# Segregating the data where Type_code is Causes

df_TamilNadu_causes = df_TamilNadu[df_TamilNadu['Type_code'] == 'Causes']
df_TamilNadu_causes.groupby('Type')['Total'].sum()

# Grouping the data into its each of its above constituent types

df_TamilNadu_bycauses = df_TamilNadu_causes.groupby('Type')['Total'].sum()
df_TamilNadu_causes_Type = pd.DataFrame(df_TamilNadu_bycauses).reset_index().sort_values('Total')
print(f"Total number of people in Tamil Nadu whose cause of Suicide is mentioned is {df_TamilNadu_causes.shape[0]}")

# Drawing a bar graph of the data that we have grouped above

sns.barplot(x=df_TamilNadu_causes_Type['Total'], y= df_TamilNadu_causes_Type['Type'])
