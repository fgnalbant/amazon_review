import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.max_columns",None)
pd.set_option("display.expand_frame_repr",False)
pd.set_option("display.float_format",lambda x: "%.5f" %x)
from sklearn.preprocessing import MinMaxScaler

df_=pd.read_csv("D:/Gizem Nalbant/datasets/amazon_review.csv",low_memory=False) # DtypeWarning kapatmak için
df=df_.copy()
df.head()
df.shape
df.info()

#Adım 1: Ürünün ortalama puanını hesaplanması
df["overall"].mean()

#Adım 2: Tarihe göre ağırlıklı puan ortalamasını hesaplanması
df["reviewTime"] = pd.to_datetime(df["reviewTime"])
current_date=df["reviewTime"].max()
df["review_days"] = (current_date - df["reviewTime"]).dt.days

df["review_days"].quantile([.25, .50, .75]) # 280 , 430 , 600


def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return df.loc[df["review_days"] <= 280, "overall"].mean() * w1 / 100 + \
            df.loc[(df["review_days"] > 280) & (df["review_days"] <= 430), "overall"].mean() * w2 / 100 + \
            df.loc[(df["review_days"] > 430) & (df["review_days"] <= 600), "overall"].mean() * w3 / 100 + \
            df.loc[df["review_days"] > 600, "overall"].mean() * w4 / 100

time_based_weighted_average(df)

#Adım 3: Ağırlıklandırılmış puanlamada her bir zaman diliminin ortalamasını karşılaştırıp yorumlayınız

df.loc[df["review_days"] <= 280, "overall"].mean() #4.69
df.loc[(df["review_days"] > 280) & (df["review_days"] <= 430), "overall"].mean() #4.63
df.loc[(df["review_days"] > 430) & (df["review_days"] <= 600), "overall"].mean() #4.64
df.loc[df["review_days"] > 600, "overall"].mean() #4.44

##zamanla puan ortalaması artmış.Bu da üründe zamanla geliştirmeler yapıldığını ortaya koymaktadır.


#Görev 2: Ürün için ürün detay sayfasında görüntülenecek 20 review’i belirleyiniz
#Adım 1: helpful_no değişkenini üretiniz

df["helpful_no"] = df["total_vote"] - df["helpful_yes"]
df.head(50)


#Adım 2: score_pos_neg_diff, score_average_rating ve wilson_lower_bound skorlarını hesaplayıp veriye ekleyiniz.

def score_pos_neg_diff(helpful_yes, helpful_no):
    return helpful_yes - helpful_no

def score_average_rating(helpful_yes, helpful_no):
    if helpful_yes + helpful_no == 0:
        return 0
    return helpful_yes / (helpful_yes + helpful_no)


def wilson_lower_bound(helpful_yes, helpful_no, confidence=0.95):
    n = helpful_yes + helpful_no
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * helpful_yes / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


votes = pd.DataFrame({"helpful_yes": df["helpful_yes"], "helpful_no": df["helpful_no"]})

df["score_pos_neg_diff"] = votes.apply(lambda x: score_pos_neg_diff(x["helpful_yes"],
                                                                             x["helpful_no"]), axis=1)

df["score_average_rating"] = votes.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)

df["wilson_lower_bound"] = votes.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)


#Adım3: 20 yorumu belirleyiniz.Sonuçları yorumlayınız.

df.sort_values("wilson_lower_bound", ascending=False).head(20)



