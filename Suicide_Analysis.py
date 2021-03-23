#!/usr/bin/env python
# coding: utf-8

# ### Analysis of Suicides in India

# Here we are analysing the Suicide dataset given above, the dataset consists of the data, from all over India, of the number of people who have committed suicide during the period between 2001 and 2012, the State that they are from, the year, the cause, their Education qualification, their Social Status, their Profession, their age and the means that they adopted to commmit suicide.
# 
# The dataset is fairly huge having around 237,520 rows/values and 7 columns, good for analysis. 
# 
# Here we will be trying to answer some basic questions and also try to draw a few other conclusions that is hidden in the data.
# 
# The modules that we would be using for this purpose are :
# 1. pandas - for reading the dataset and for it's inbuilt functions that help us manupulate the data in the dataset.
# 2. numpy - for performing all the mathematical computations on the data.
# 3. seaborn - for the beautiful bargraphs and plots that it has, which can be used to visualize the data, helpful for analysis
# 4. matplotlib - for drawing the piechart's, also used to aid our analysis.
# 
# Incase you do not have any of the above packages you can install them using the command -pip install <library_name> in Anaconda prompt or the command prompt.
# 
# Incase you want to try this on your own, download the dataset and the relevant format of the code whether '.ipnyb' or '.py', put both the dataset and the code in the same directory and then run it locally in your machine.
# 
# Try to do it on your own for better understanding and to learn the concepts.
# 
# Let's begin, Happy Learning.

# In[1]:


# importing the required libraries 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


# Read the Suicide data '.csv' and check if it is has been read properly

df = pd.read_csv('Suicides in India 2001-2012.csv')
df.head()


# In[3]:


# See the common statistics of the data

df.describe()


# In[4]:


# See the basic description of the variables, their type and value count

df.info()


# In[5]:


# print the all the states available in the dataset

print(df['State'].unique())


# In[6]:


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


# In[7]:


# Here we remove all the values whose exact causes are not known, this is to ensure that the data whose cause is not known does not
# end up affecting our analysis.

df = df.drop(df[(df.State == 'Total (Uts)') | (df.State == 'Total (All India)') | (df.State =='Total (States)')].index)
df = df.drop(df[(df.Type == 'By Other means') | (df.Type =='Other Causes (Please Specity)') | (df.Type =='Other Causes (Please Specify)') | (df.Type =='Causes Not known') | (df.Type =='By Other means (please specify)')].index)
df = df.drop(df[df['Total'] == 0].index)


# In[8]:


#Plot a bargraph based on the gender to see which gender people commit the most suicides

sns.set(rc={'figure.figsize':(10,8)})
sns.barplot(x=df['Gender'], y=df['Total'])


# In[9]:


# Draw the same bargraph as a pie chart to see the percentage stats

df_gender = df.groupby('Gender')['Total'].sum()
df_gender_type = pd.DataFrame(df_gender).reset_index().sort_values('Total')
labels = df_gender_type['Gender']
exp = (0,0.1)
plt.pie(df_gender_type['Total'], labels=labels, autopct='%1.1f%%', explode = exp, shadow=True)
plt.show()


# From the above bargraph and pie chart we can see that more number of Males commit suicide than females, reasons why this happens is not yet known from the analysis performed until now.
# 
# But we can now establish the fact that more number of males commit suicide than females, causes based on the number of people who have committed suicide from the period of 2001 to 2012 62.8% are men and rest 37.2% women.

# In[10]:


# Here we are spliting the data into different age groups

gp_age = df.groupby('Age_group')['Total'].sum()
df_age_type = pd.DataFrame(gp_age).reset_index().sort_values('Age_group')
df_age_type = df_age_type.drop([0])
df_age_type


# In[11]:


# Draw a bargraph categorising the people into different age groups

sns.barplot(x = df_age_type['Age_group'], y = df_age_type['Total'])


# In[12]:


# Draw the same bargraph as a pie chart to see the percentage stats

labels = df_age_type['Age_group']
exp = (0,0.1,0.1,0,0)
plt.pie(df_age_type['Total'], labels=labels, autopct='%1.1f%%', explode = exp, shadow=True)
plt.show()


# From the above charts we can see that more number of Suicides are committed by the people belonging to the age groups 15-29 yrs (36.6%) closely followed by those in the age group 30-44 yrs (34.1%).
# 
# This gives rise to the conculsion that Suicides are more common among teens, teenagers and the working population then among others, thereby helping us to focus, the limited resources for Suicide prevention campaigns, for people in those age groups.

# In[13]:


# Here we are categorising the people into different age groups and proceed to futher 
# split them on the basis of their gender

df_age = df.drop(df[(df['Age_group'] == '0-100+')].index)
df_age = df_age.groupby(['Age_group', 'Gender'])['Total'].sum()
df_age_gender = pd.DataFrame(df_age).reset_index().sort_values('Age_group')
df_age_gender


# In[14]:


# Draw a bargraph for the above data that we had split

sns.barplot(x = "Age_group", y = "Total", hue = "Gender", data=df_age_gender)


# From the above bargraph we can see that, in the age group 30-44 (Working class) more number of males commit suicide than females, and in the age group of 15-29 (teenage) the number is almost the same with men excceding the women by not so huge a margin. 
# The reasons for this may be many but we cannot draw a concrete conclusion on the reason given the fact that we either do not have enough data or we would have to analyse further.

# In[15]:


# Here we are grouping the data statewise so as to see which state has the most cases.

group = df.groupby('State')['Total'].sum()
tot_suicides = pd.DataFrame(group).reset_index().sort_values('Total',ascending=False)


# In[16]:


# Draw a bargraph categorising the data into statewise form, which is obtained from above

fig, ax = plt.subplots(figsize=(18,6))
plot = sns.barplot(x = 'State', y = 'Total', data = tot_suicides)
plot.set_xticklabels(plot.get_xticklabels(), rotation = 45)
plt.show()


# From the above bargraph we can see that Maharastra has registered the most number of Suicides, followed by West Bengal, Andhra Pradesh and Tamil Nadu. West Bengal, Andhra Pradesh and Tamil Nadu have almost the same number of cases while Maharastra exceeds them by a wide margin. 
# 
# Towards the end you can see that the North-eastern states, and the union territories have the least number of Suicides recorded.

# In[18]:


# Here we are grouping the data yearwise so as to see which year has the most cases.

group_yr = df.groupby('Year')['Total'].sum()
yr = pd.DataFrame(group_yr).reset_index().sort_values('Year',ascending=False)
print(group_yr)


# In[19]:


# Draw a bargraph categorising the data into yearwise form, which is obtained from above

plot = sns.barplot(x = 'Year', y = 'Total', data = yr, palette = 'flare')


# From the above graph we can conclude that the number of Suicide cases has been raising year by year with 2010 and 2011 recording the highest number of cases in 12 years. We can see that the number of cases shows a clear decline in 2012 from its steady rise in the previous years. 

# Let's take the analysis further by analying all the individual factors like Social status, the means used, the causes, their Educational qualification and Professional Profile to gather more insights on why people commit suicide. This way we can gather some important conclusions which can help us in reducing the number of cases by helping us target the major reasons thereby giving us significant results than could be obtained if we were to attempt to reduce the cases without the any sort of analysis.

# In[20]:


# Here we are grouping the data based on the Type code which consists of all the factors 
# that we wish to analyse, as mentioned above.

group_count = df['Type_code'].value_counts()
group_count


# In[21]:


# Draw a bargraph of the data that we have grouped above

sns.set(rc={'figure.figsize':(12,9)})
sns.countplot(df['Type_code'])


# From the above graph we can see how much of data is available for each of the catergory mentioned.

# Now let's analyse each of the catergory in more detail and see which type in each category contribues to more number of Suicides.

# Let's begin our analysis with the Social Status of the people

# In[22]:


# Segregating the data where Type_code is Social_Status 

df_bycode = df[df['Type_code'] == 'Social_Status']
df_bycode['Type'].unique()


# In[23]:


# Grouping the data into its each of its constituent types

df_social = df_bycode.groupby('Type')['Total'].sum()
df_social_type = pd.DataFrame(df_social).reset_index().sort_values('Total')
print(f'People who have mentioned their Social Status {df_bycode.shape[0]}')


# In[24]:


# Drawing a pie chart of the data that we have grouped above

labels = df_social_type['Type']
exp = (0,0,0,0,0.1)
plt.pie(df_social_type['Total'], labels=labels, autopct='%1.1f%%', explode = exp, shadow=True)
plt.show()


# In[25]:


#Drawing the same pie chart as a bargraph to see the absolute values

sns.barplot(x = df_social_type['Total'], y=df_social_type['Type'])


# From the above pie chart and bargraph we can conclude that more number of Suicides occurs among married people followed by unmarried people. The least number of cases can be seen for divorced people. We can draw a conclusion here that the chance a person might likely commit suicide increases if he/she is married, i.e married couple have a higher chance of committing Suicide which means that many married people are more likely unhappy with their lives.

# In[26]:


# Segregating the data where Type_code is Professional_Profile

df_bycode = df[df['Type_code'] == 'Professional_Profile']
df_bycode['Type'].unique()


# In[27]:


# Grouping the data into its each of its above constituent types

df_byProfession = df_bycode.groupby('Type')['Total'].sum()
df_Profession_Type = pd.DataFrame(df_byProfession).reset_index().sort_values('Total')
print(f'People who mentioned their Professions {df_bycode.shape[0]}')


# In[28]:


# Drawing a pie chart of the data that we have grouped above

labels = df_Profession_Type['Type']
exp = (0,0,0,0,0,0,0,0.1,0.1,0.1)
plt.pie(df_Profession_Type['Total'], labels=labels, autopct='%1.1f%%', explode = exp, shadow=True)
plt.show()


# In[29]:


#Drawing the same pie chart as a bargraph to see the absolute values

sns.barplot(x = df_Profession_Type['Total'], y=df_Profession_Type['Type'])


# From the above pie chart and bargraph we can conclude that more number of Suicides occur amongst house wives followed by those involved in Agricultural activities, Private service, people who are unemployed and the rest. Lowest Suicides are found among people who are in Government service and those retired. A conclusion that we might be able to draw her is that house wives are the most stressed people, and it makes sense since most of the house wives are not even acknowledged for their work, while some are even insulted and criticised. Only if house wives are given respect will the number of house wives committing suicide reduce. 
# 
# Farmers case is more delicate, farming is considered as a low key profession in India with farmers not having education, not being supported and in most cases are cheated. They work hard to feed the nation while we turn our backs upon them. Farmers committing suicide will not change unless governement and the people intervine to help the farmers and instead of looking down on the profession we must look up to the profession

# In[30]:


# Segregating the data where Type_code is Means_adopted

df_bycode = df[df['Type_code'] == 'Means_adopted']
df_bycode['Type'].unique()


# In[31]:


# Grouping the data into its each of its above constituent types

df_bymeans = df_bycode.groupby('Type')['Total'].sum()
df_means = pd.DataFrame(df_bymeans).reset_index().sort_values('Total')
print(f'Total number of People whose means of suicide is mentioned is {df_bycode.shape[0]}')


# In[32]:


# Drawing a pie chart of the data that we have grouped above

labels = df_means['Type']
exp = [0]*(len(labels.unique())-3) + [0.1,0.1,0.1]
plt.pie(df_means['Total'], labels=labels, autopct='%1.1f%%', explode = exp, shadow=True)
plt.show()


# In[33]:


#Drawing the same pie chart as a bargraph to see the absolute values

sns.barplot(x = df_means['Total'], y=df_means['Type'])


# We can see that most of the Suicides happen with the victim choosing hanging as the means of death, followed by consuming insecticides or other poison, it is then followed by fire/immolation, drowning with the least number of victims using Machines. 
# 
# Now we can see the hidden facts that the data has to offer.
# 
# Since most number of victims are house wives followed by those in Agricultural activities, the means are also outh to make sense. Hanging, consuming poison or burning is the easy way out for house wives and farmers since they are easily accessible to them given their profession. 

# In[34]:


# Segregating the data where Type_code is Education_Status

df_bycode = df[df['Type_code'] == 'Education_Status']
df_bycode['Type'].unique()


# In[35]:


# Grouping the data into its each of its above constituent types

df_byeducation = df_bycode.groupby('Type')['Total'].sum()
df_education = pd.DataFrame(df_byeducation).reset_index().sort_values('Total')
print(f'Total number of People whose Educational Status is mentioned is {df_bycode.shape[0]}')


# In[36]:


# Drawing a pie chart of the data that we have grouped above

labels = df_education['Type']
plt.pie(df_education['Total'], labels=labels, autopct='%1.1f%%', explode = [0,0,0,0,0,0.1,0.1,0.1])
plt.show()


# In[37]:


#Drawing the same pie chart as a bargraph to see the absolute values

sns.barplot(x = df_education['Total'], y=df_education['Type'])


# Most number of Suicides are seen among the people who have not completed their primary education, followed by those who have only completed till Secondary and the least cases are seen amongst postgraduates. This establishes a very important fact that education is vital in the society today.
# 
# Education has the potential to change peoples lives, helping them lead good lives.
# 
# Thus to reduce the number of Suicides the governement must focus on educating the people until their Secondary level.

# In[38]:


# Segregating the data where Type_code is Causes

df_bycode = df[df['Type_code'] == 'Causes']
df_bycode['Type'].unique()


# In[39]:


# Grouping the data into its each of its above constituent types

df_bycauses = df_bycode.groupby('Type')['Total'].sum()
df_causes = pd.DataFrame(df_bycauses).reset_index().sort_values('Total')
print(f'Total causes that have led to Suicide {df_bycode.shape[0]}')


# In[40]:


# Drawing a pie chart of the data that we have grouped above

labels = df_causes['Type']
exp = [0]*(len(labels.unique())-3) + [0.1,0.1,0.1]
plt.pie(df_causes['Total'], labels=labels, autopct='%1.1f%%', explode = exp)
plt.show()


# In[41]:


#Drawing the same pie chart as a bargraph to see the absolute values

sns.barplot(x = df_causes['Total'], y=df_causes['Type'])


# Family problems is the singlest most important cause that forces people to commit suicide. Family is the support and strength for any individual, if they is not there then what do the people have to live and work for. 
# 
# We were able to arrive at a conclusion above that Housewives are the people who commit the most number of suicides and here we are able to conclude that most causes are Family Problems. So, being unhappy at house is the singlest huge factor that contributes to more number of cases among people. 
# 
# This can be solved by giving equal power to women in the house and by treating them with respect. Their contribution has to be acknowleged and appreciated for a health and happy family. This way we can also reduce the number of cases significantly.

# #### Now lets analyse further by considering the state of Maharashtra which has the most number of suicides recorded.

# In[42]:


# Segregating the data of Maharashtra.

df_Maharashtra = df[df['State']=='Maharashtra']
df_Maharashtra


# In[43]:


# Segregating the data where Type_code is Social_Status

df_Maharashtra_Social = df_Maharashtra[df_Maharashtra['Type_code'] == 'Social_Status']
df_Maharashtra_Social.groupby('Type')['Total'].sum()


# In[44]:


# Grouping the data into its each of its above constituent types

df_Maharashtra_bySocial = df_Maharashtra_Social.groupby('Type')['Total'].sum()
df_Maharashtra_Social_Type = pd.DataFrame(df_Maharashtra_bySocial).reset_index().sort_values('Total')
print(f'Total people with Social Status mentioned in Maharashtra {df_Maharashtra_Social.shape[0]}')


# In[45]:


# Drawing a bar graph of the data that we have grouped above

sns.barplot(x = df_Maharashtra_Social_Type['Total'], y=df_Maharashtra_Social_Type['Type'])


# We can see that Married people are the ones who commit more suicide, reasons may be family problems and the fact that house wives commit the most suicides that adds up. But from what we see we can observe that the number of married people who commit suicide far outweigh the other factors thus establishing the fact that most of the married couples are generally unhappy for each other.

# In[46]:


# Segregating the data where Type_code is Professional_Profile

df_Maharashtra_Professional = df_Maharashtra[df_Maharashtra['Type_code'] == 'Professional_Profile']
df_Maharashtra_Professional.groupby('Type')['Total'].sum()


# In[47]:


# Grouping the data into its each of its above constituent types

df_Maharashtra_byProfessional = df_Maharashtra_Professional.groupby('Type')['Total'].sum()
df_Maharashtra_Professional_Type = pd.DataFrame(df_Maharashtra_byProfessional).reset_index().sort_values('Total')
print(f'Total people with Profession mentioned in Maharashtra {df_Maharashtra_Professional.shape[0]}')


# In[48]:


# Drawing a bar graph of the data that we have grouped above

sns.barplot(x = df_Maharashtra_Professional_Type['Total'], y=df_Maharashtra_Professional_Type['Type'])


# We can see that more people involved in Farming/Agriculture Activity end up committing suicide in the state of Maharashtra. Maharashtra is primarily a farming state with many people in the state involved directly or indirectly on agriculture. We have also concluded before that farmers are the most affected because farming is not seen as a reputed profession and so the statistics obtained from the data on Maharashtra are logically deducable. The state is a farming state and farming is not seen as a proper profession, and so more people who commit suicide are from this profession. 
# Thus this state having the number one position is also logically deductable from the above reasoning.
# 
# This calls for the governement intervention to support farming and for us to start respecting farming as a nobel profession, and to also help out farmers to reduce the number of suicides.
# 
# This is then followed by house wives, even in the overall data we see that it's the house wives who commit the most number of suicides. This can only be prevented when the people start acknowledging the house wife and start respecting her for the work she does to keep the house livable and prosporous.

# In[49]:


# Segregating the data where Type_code is Means_adopted

df_Maharashtra_means = df_Maharashtra[df_Maharashtra['Type_code'] == 'Means_adopted']
df_Maharashtra_means.groupby('Type')['Total'].sum()


# In[50]:


# Grouping the data into its each of its above constituent types

df_Maharashtra_bymeans = df_Maharashtra_means.groupby('Type')['Total'].sum()
df_Maharashtra_means_Type = pd.DataFrame(df_Maharashtra_bymeans).reset_index().sort_values('Total')
print(f'Total people with means of Suicide mentioned in Maharashtra {df_Maharashtra_means.shape[0]}')


# In[51]:


# Drawing a bar graph of the data that we have grouped above

sns.barplot(x = df_Maharashtra_means_Type['Total'], y=df_Maharashtra_means_Type['Type'])


# We can see that hanging and consuming insecticides are the most common means of committing suicide in Maharashtra. This can be arrived at by the fact that most of the suicides are committed by people in the farming/agriculture and by house wives, since insecticides and rope are two of the most common things available for house wives and farmers. There have been many instances in news too about how a farmer killed himself by drinking insecticide.

# In[52]:


# Segregating the data where Type_code is Education_Status

df_Maharashtra_Education = df_Maharashtra[df_Maharashtra['Type_code'] == 'Education_Status']
df_Maharashtra_Education.groupby('Type')['Total'].sum()


# In[53]:


# Grouping the data into its each of its above constituent types

df_Maharashtra_byEducation = df_Maharashtra_Education.groupby('Type')['Total'].sum()
df_Maharashtra_Education_Type = pd.DataFrame(df_Maharashtra_byEducation).reset_index().sort_values('Total')
print(f'Total people whose Education Qualification is mentioned in Maharashtra is {df_Maharashtra_Education.shape[0]}')


# In[54]:


# Drawing a bar graph of the data that we have grouped above

sns.barplot(x = df_Maharashtra_Education_Type['Total'], y=df_Maharashtra_Education_Type['Type'])


# We can see that most number of people who committed suicide are people who have not completed their secondary education, we can see a significant drop in the number of suicides by people who have completed their secondary education and have gone beyond it. Maharashtra being a farming state primarily, tends to have more rural areas than urban areas and higher education is usually not given much importance in rural areas, with students helping their parents in farming rather than studying. They are also usually married off young and end up farming. 
# 
# So, it is rightly said that education can help in the upliftment of the nation. The people need to be educated, since most educated people (ones who study beyond their secondary education) do not tend to commit suicide easily. 
# 
# So, governement and the families in these areas need to educate their children for the better.

# In[55]:


# Segregating the data where Type_code is Causes

df_Maharashtra_Causes = df_Maharashtra[df_Maharashtra['Type_code'] == 'Causes']
df_Maharashtra_Causes.groupby('Type')['Total'].sum()


# In[56]:


# Grouping the data into its each of its above constituent types

df_Maharashtra_byCauses = df_Maharashtra_Causes.groupby('Type')['Total'].sum()
df_Maharashtra_Causes_Type = pd.DataFrame(df_Maharashtra_byCauses).reset_index().sort_values('Total')
print(f'Total people in Maharashtra whose cause of Suicide is mentioned is {df_Maharashtra_Causes.shape[0]}')


# In[57]:


# Drawing a bar graph of the data that we have grouped above

sns.barplot(x = df_Maharashtra_Causes_Type['Total'], y=df_Maharashtra_Causes_Type['Type'])


# We can see that the major cause of suicide is attributed to Family problems. Lack of proper education, getting married at a young age and lack of amenities tends to cause rifts in families, usually these problems can be solved but then suicide is chosen as the easy way out, and its the women(the house wives) who usually commit suicide as we saw above. 
# 
# So, it can be said educating a female child and saying no to marriage at an young age can significantly reduce the number of suicides. This is where the indepth analysis helps us, see how we are able to start forming logical conclusions and are able to target the places that matter and give the highest results the most instead of blindly trying some measures and hoping that they would help.

# #### Now lets do an analysis on the state of Tamil Nadu which is the state with the 4th highest number of suicides recorded.

# In[74]:


# Segregating the data of Tamil Nadu.

df_TamilNadu = df[df['State']=='Tamil Nadu']
df_TamilNadu


# In[59]:


# Segregating the data where Type_code is Social_Status

df_TamilNadu_Social = df_TamilNadu[df_TamilNadu['Type_code'] == 'Social_Status']
df_TamilNadu_Social.groupby('Type')['Total'].sum()


# In[60]:


# Grouping the data into its each of its above constituent types

df_TamilNadu_bySocial = df_TamilNadu_Social.groupby('Type')['Total'].sum()
df_TamilNadu_Social_Type = pd.DataFrame(df_TamilNadu_bySocial).reset_index().sort_values('Total')
print(f"Total number of people in Tamil Nadu whose Social Status is mentioned {df_TamilNadu_Social.shape[0]}")


# In[61]:


# Drawing a bar graph of the data that we have grouped above

sns.barplot(y=df_TamilNadu_Social_Type['Type'],x=df_TamilNadu_Social_Type['Total'])


# We can see that more number of people who commit suicide are married and the reason for this has already been discussed in the analysis performed few blocks above in the analysis performed for the state of Maharashtra.

# In[62]:


# Segregating the data where Type_code is Professional_Profile

df_TamilNadu_Professional = df_TamilNadu[df_TamilNadu['Type_code'] == 'Professional_Profile']
df_TamilNadu_Professional.groupby('Type')['Total'].sum()


# In[63]:


# Grouping the data into its each of its above constituent types

df_TamilNadu_byProfessional = df_TamilNadu_Professional.groupby('Type')['Total'].sum()
df_TamilNadu_Professional_Type = pd.DataFrame(df_TamilNadu_byProfessional).reset_index().sort_values('Total')
print(f"Total number of people in Tamil Nadu whose Profession is mentioned is {df_TamilNadu_Professional.shape[0]}")


# In[64]:


# Drawing a bar graph of the data that we have grouped above

sns.barplot(x=df_TamilNadu_Professional_Type['Total'], y= df_TamilNadu_Professional_Type['Type'])


# We can see that more number of people who commit suicide are House wives and the reason for this has already been discussed in the analysis performed few blocks above in the analysis performed for the state of Maharashtra.

# In[65]:


# Segregating the data where Type_code is Means_adopted

df_TamilNadu_means = df_TamilNadu[df_TamilNadu['Type_code'] == 'Means_adopted']
df_TamilNadu_means.groupby('Type')['Total'].sum()


# In[66]:


# Grouping the data into its each of its above constituent types

df_TamilNadu_bymeans = df_TamilNadu_means.groupby('Type')['Total'].sum()
df_TamilNadu_means_Type = pd.DataFrame(df_TamilNadu_bymeans).reset_index().sort_values('Total')
print(f"Total number of people in Tamil Nadu whose means of Suicide is mentioned is {df_TamilNadu_means.shape[0]}")


# In[67]:


# Drawing a bar graph of the data that we have grouped above

sns.barplot(x=df_TamilNadu_means_Type['Total'], y= df_TamilNadu_means_Type['Type'])


# We can see that Hanging is the most common means of committing suicide followed by consuming poison and insecticides. We can see in Tamil Nadu that committing suicide by Fire/self immolation is the 4th highest means people use to commit suicide and based on this, one of the conclusions that we can arrive at is that given the most number of suicides are committed by housewives, who are obviously married(thereby contibuting to marriage being in the top), it makes sense that choosing fire/self immolation is in the 4th place, since for housewives in Tamil Nadu fire and kerosene are easily accessible along with chemicals and there are also a lot of news articles that say that victim has burned themselves to death. This is only seen in Tamil Nadu while for the rest of the states Fire/self immolation tends to be the least used method.

# In[68]:


# Segregating the data where Type_code is Education_Status

df_TamilNadu_Education = df_TamilNadu[df_TamilNadu['Type_code'] == 'Education_Status']
df_TamilNadu_Education.groupby('Type')['Total'].sum()


# In[69]:


# Grouping the data into its each of its above constituent types

df_TamilNadu_byEducation = df_TamilNadu_Education.groupby('Type')['Total'].sum()
df_TamilNadu_Education_Type = pd.DataFrame(df_TamilNadu_byEducation).reset_index().sort_values('Total')
print(f"Total number of people in Tamil Nadu whose Education Qualification is mentioned is {df_TamilNadu_Education.shape[0]}")


# In[70]:


# Drawing a bar graph of the data that we have grouped above

sns.barplot(x=df_TamilNadu_Education_Type['Total'], y= df_TamilNadu_Education_Type['Type'])


# The analysis we can make here is almost the same as that of Maharashtra, which can be referred from the blocks above.

# In[71]:


# Segregating the data where Type_code is Causes

df_TamilNadu_causes = df_TamilNadu[df_TamilNadu['Type_code'] == 'Causes']
df_TamilNadu_causes.groupby('Type')['Total'].sum()


# In[72]:


# Grouping the data into its each of its above constituent types

df_TamilNadu_bycauses = df_TamilNadu_causes.groupby('Type')['Total'].sum()
df_TamilNadu_causes_Type = pd.DataFrame(df_TamilNadu_bycauses).reset_index().sort_values('Total')
print(f"Total number of people in Tamil Nadu whose cause of Suicide is mentioned is {df_TamilNadu_causes.shape[0]}")


# In[73]:


# Drawing a bar graph of the data that we have grouped above

sns.barplot(x=df_TamilNadu_causes_Type['Total'], y= df_TamilNadu_causes_Type['Type'])


# Family Problems seems to be the most common problem everywhere since almost all states have it as their primary or secondary cause. But in case of Tamil Nadu we can see that suicides due to prolonged illness tends to take the 2nd place and Love Affairs tends to take the 3rd place. We can conclude by either saying that people of Tamil Nadu are more prone to prolonged illness or the medical facilties might not be well enough and to conclude upon a point we would require more in depth data, which may or may not be available. We can see this as a problem with only a few states other than Tamil Nadu. 

# To summarize we first begun by studying the data and then preprocessing it to clean it of all the possible errors, remove the unwanted data, merge or remove the redundant data which might end up giving wrong results and therfore lead to erroneous analysis. 
# 
# We then went ahead with first drawing a conculsion on which gender has the most cases and explored a bit of reasoning as to why we see males commit more suicides, then we analysed as to which age group commits the most suicides, from there we combined both the gender analysis and age group analysis, thereby making a few conculsions over there, then we made a statewise comparision where we saw Maharashtra topping the number of cases list.
# 
# We then saw each type individually and then went ahead for an indepth analysis where we analysed based on the suicide causes, means used, the profession of the people, their education status and their social status where we were able to draw some good conclusion that can be used to effectively reduce the number of suicides. This was we are able to target the places which give us really good results instead of blindly targetting without proper plan and retargetting by changing the plan based on the observed results, which can be a tiresome and cumbersome process.
# 
# We then, as a further example, extended our analysis to two states, i.e Maharashtra (1st in the number of cases) followed by Tamil Nadu (4th in the number of cases). We were able to draw some intresting insights from these analysis too.

# Thus, we have performed a indepth analysis of the suicide dataset and have used the python packages to help us represent the data visually using bar graphs and pie charts, visual form of representation is always appriciated since it helps us understand and perform analysis easily. 
# 
# If you are interested you can always find more ways to interpret the data and gain more insights. Feel free to explore beyond what I have done.
# 
# Hope you understood the procedure and the functions, if you were not able to understand some of the functions you can always copy the function and search it in google for a detailed information. 
# 
# Incase you spot a mistake, am sorry, and feel free to correct it.
# 
# Thank you for reading this.
# 
# Keep Learning and have fun.
