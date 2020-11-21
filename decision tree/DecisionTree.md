```pyspark
from pyspark.sql.functions import *
from pyspark.sql.types import *
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…



```pyspark
(data, test) = spark.read.csv("s3://decisiontree-bigdata/train.csv", header=True, inferSchema=True).drop("Dates").drop("Descript") \
                    .drop("Resolution").drop("X").drop("Y").randomSplit([0.002, 0.9998])
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…



```pyspark
data.count()
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    1817


```pyspark
data.schema.names
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    ['Category', 'DayOfWeek', 'PdDistrict', 'Address']


```pyspark
data.select("Category").distinct().rdd.map(lambda r : r[0]).collect()
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    ['LIQUOR LAWS', 'BRIBERY', 'LOITERING', 'VEHICLE THEFT', 'LARCENY/THEFT', 'SECONDARY CODES', 'WARRANTS', 'ASSAULT', 'EXTORTION', 'VANDALISM', 'TRESPASS', 'STOLEN PROPERTY', 'BURGLARY', 'FRAUD', 'MISSING PERSON', 'DRUG/NARCOTIC', 'NON-CRIMINAL', 'OTHER OFFENSES', 'DRUNKENNESS', 'SUSPICIOUS OCC', 'ROBBERY', 'EMBEZZLEMENT', 'RECOVERED VEHICLE', 'KIDNAPPING', 'SEX OFFENSES FORCIBLE', 'DRIVING UNDER THE INFLUENCE', 'ARSON', 'FORGERY/COUNTERFEITING', 'WEAPON LAWS', 'PROSTITUTION', 'DISORDERLY CONDUCT', 'RUNAWAY', 'BAD CHECKS']


```pyspark
def calTargetEntropy(data, target):
  targetCol = data.select(target) 
  total = targetCol.count() 
  targetColMap = targetCol.groupBy(target).count()
  targetEntropyCol = targetColMap.withColumn("probability", col("count") / lit(total)).withColumn("entropy", - col("probability") * log(2.0, col("probability"))).agg({"entropy": "sum"})
  return targetEntropyCol.collect()[0][0]
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…



```pyspark
def getBestFeature(data, target):
  columns = data.schema.names
  bestGain = 0.0
  bestFeature = ""
  total = data.count()
  targetEntropy = calTargetEntropy(data, "Category")
  for name in columns:
    if name == target: continue
    df = data.groupBy(name, target).count()
    totaldf = data.groupBy(name).count().toDF(name, "totalcount")
    joindf = df.join(totaldf, on = [name])
    resultdf = joindf.withColumn("proc", col("count") / col("totalcount")).withColumn("entropy", - col("proc") * log(2.0, col("proc"))).groupBy(name).agg({"entropy":"sum"}).toDF(name, "entropy")
    procdf = totaldf.withColumn("proc", col("totalcount") / lit(total)).select(name, "proc")
    entropy = resultdf.join(procdf, on=[name]).withColumn("mul", col("proc") * col("entropy")).agg({"mul": "sum"}).collect()[0][0]  
    if targetEntropy - entropy > bestGain:
      bestGain = targetEntropy - entropy
      bestFeature = name
  return bestFeature
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…



```pyspark
def filterData(data, colName, value):
  filtered = data.filter(col(colName) == value).drop(colName)
  return filtered
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…



```pyspark
def getMajorClass(data, target):
    temp = data.select(target).groupBy(target).count()
    return temp.orderBy(col("count").desc()).collect()[0][0]
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…



```pyspark
major = getMajorClass(data, "Category")
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…



```pyspark
major
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    'LARCENY/THEFT'


```pyspark
dic = {}
dic['value']=1
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…



```pyspark
dic
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    {'value': 1}


```pyspark
def createDecisionTree(data, target, features, times):
  targetClasses = data.select(target).distinct().rdd.map(lambda r : r[0]).collect()
  if len(targetClasses) == 0:
    raise Exception("Class Error")

  # stop when only one category left
  if len(targetClasses) == 1:
    return targetClasses[0]

  if len(data.columns) <= 1:
    raise Exception("Columns Error")
  
  # return major classes when only one feature left
  if len(features) == 1:
    return getMajorClass(data, target)
  
  bestFeature = getBestFeature(data, target)
  # sc.parallelize([bestFeature]).coalesce(1).saveAsTextFile("s3://decisiontree-bigdata/" + str(times))
  tree = {bestFeature: {}}
  subFeatures = features.remove(bestFeature)
  values = data.select(bestFeature).distinct().rdd.map(lambda r : r[0]).collect()
  for value in values:
    times += 1
    tree[bestFeature][value] = createDecisionTree(filterData(data, bestFeature, value), target, subFeatures, times)
    
  return tree
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…



```pyspark
tree = createDecisionTree(data, "Category", data.schema.names, 0)
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…



```pyspark
tree
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    'LARCENY/THEFT'


```pyspark

```
