library(tidyverse)



df.empty1 = data.frame(var1=0.0,var2=0.0,var3=0.0,var4="")
df.empty2 = data.frame(var1=0.0,var2=0.0,var3=0.0,var4="")
df.empty1$var1=11.5
df.empty1$var2=as.integer(3)
df.empty1$var4="sb1"
df.empty2$var1=12.9
df.empty2$var2=4
df.empty2$var4="sb2"
df.empty=rbind(df.empty1,df.empty2)
print(df.empty)


data=relig_income

new_data=data%>%
  pivot_longer(cols=-religion,
               names_to="income",
               values_to="frequency")%>%
  mutate(income=factor(income,levels=c("<&10k",
                                       "$10-20k",
                                       "$20-30k",
                                       "$30-40k",
                                       "$40-50k",
                                       "$50-75k",
                                       "$75-100k",
                                       "$100-150k",
                                       ">150k",
                                       "Don't know/refused")))


# graphic=new_data %>%
#   ggplot(aes(x=religion,y=frequency,fill=income))+
#   geom_bar(stat="identity",position="fill")+
#   scale_fill_viridis_d()+
#   labs(x="Religion",y="Proportion",fill="Income")+
#   coord_flip()+
#   theme_minimal()
# print(graphic)
data(mtcars)
graphic2=ggplot(mtcars,aes(wt,mpg))+
  geom_point(size=6,shape=1)

mtcars2=mtcars
mtcars2$gear=as.factor(mtcars2$gear)

graphic3=ggplot(mtcars2,aes(wt,mpg,fill=gear))+
  geom_point(size=6,shape=21)

print(graphic3)
