from pyspark.sql.types import *
from operator import add
from pyspark.sql import functions as F
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
import string
from pyspark.ml.feature import StopWordsRemover
from pyspark.sql.window import Window
from gensim.parsing.preprocessing import STOPWORDS

conf = SparkConf().setMaster("local").setAppName("My App")
sc = SparkContext(conf = conf)
spark = SparkSession(sc)

stop_words = list(STOPWORDS)
stop_words.extend(list(''+' '+string.punctuation + string.digits))

with open("nytimes_news_articles.txt", "r") as f:  # 打开文件
    data = f.read()  # 读取文件
data = data.split('\n\n')
key = []
for i in range(len(data)):
    if 'URL: http://www.nytimes.com/' in data[i]:
        tem = '/'.join(data[i][39:].split('/')[:-1])
        continue
    temp = data[i].lower()
    key.append((tem, temp.split(' ')))

pairRDD = sc.parallelize(key).toDF(['class','message'])
remover = StopWordsRemover(stopWords=stop_words,inputCol='message', outputCol='message_clean')
pairRDD = remover.transform(pairRDD).select('class', 'message_clean')
count_RDD=pairRDD.withColumn("word",F.explode("message_clean")).groupBy("class","word").count()
windows_spec = Window.partitionBy("class").orderBy(F.col("count").desc())
most_words = count_RDD.withColumn("rank", F.rank().over(windows_spec)).filter("rank <=10")
most_words.show()

most_words.toPandas().to_csv('task3_ans.csv')
