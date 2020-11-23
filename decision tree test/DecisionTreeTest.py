# Databricks notebook source
data = spark.read.option("delimiter", ";").csv("/FileStore/tables/testData.csv", header=True, inferSchema=True)

# COMMAND ----------

total = data.count()

# COMMAND ----------

total

# COMMAND ----------

data.show()

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.types import *

# COMMAND ----------

def calTargetEntropy(data, target):
  targetCol = data.select(target) 
  total = targetCol.count() 
  targetColMap = targetCol.groupBy(target).count()
  targetEntropyCol = targetColMap.withColumn("probability", col("count") / lit(total)).withColumn("entropy", - col("probability") * log(2.0, col("probability"))).agg({"entropy": "sum"})
  return targetEntropyCol.collect()[0][0]

# COMMAND ----------

def filterData(data, colName, value):
  filtered = data.filter(col(colName) == value).drop(colName)
  return filtered

# COMMAND ----------

filtered = filterData(data, "outlook", "Rain")

# COMMAND ----------

filtered.show()

# COMMAND ----------

columns = data.schema.names

# COMMAND ----------

columns

# COMMAND ----------

values = data.select('humidity').distinct().rdd.map(lambda r : r[0]).collect()

# COMMAND ----------

values

# COMMAND ----------

result

# COMMAND ----------

def getBestFeature(data, target):
  columns = data.schema.names
  bestGain = 0.0
  bestFeature = ""
  total = data.count()
  targetEntropy = calTargetEntropy(data, target)
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

# COMMAND ----------

bestFeature = getBestFeature(data, "y")

# COMMAND ----------

bestFeature

# COMMAND ----------

def getMajorClass(data, target):
  return data.select(target).groupBy(target).count().sort(col("count").desc()).collect()[0][0]

# COMMAND ----------

major = getMajorClass(data, 'y')

# COMMAND ----------

major

# COMMAND ----------

data.select("y").distinct().rdd.map(lambda r : r[0]).collect()

# COMMAND ----------

def createDecisionTree(data, target):
  targetClasses = data.select(target).distinct().rdd.map(lambda r : r[0]).collect()
  # stop when only one category left
  if len(targetClasses) <= 1:
    return targetClasses[0]
  
  # return major classes when only one feature left
  if len(data.columns) <= 2:
    return getMajorClass(data, target)
  
  bestFeature = getBestFeature(data, target)
  
  tree = {bestFeature: {}}

  columns = data.schema.names
  values = data.select(bestFeature).distinct().rdd.map(lambda r : r[0]).collect()
  for value in values:
    tree[bestFeature][value] = createDecisionTree(filterData(data, bestFeature, value), target)
    
  return tree

# COMMAND ----------

tree = createDecisionTree(data, "y")

# COMMAND ----------

tree

# COMMAND ----------

keys = tree.keys()

# COMMAND ----------

for key in keys:
  print(key)

# COMMAND ----------

data.schema.names

# COMMAND ----------

def classify(dataDict, tree):
  for key in tree.keys():
    secondDict = tree[key]
    value = dataDict[key]
    nextDict = secondDict[value]
    if isinstance(nextDict, dict):
      return classify(dataDict, nextDict)
    else:
      return nextDict
      

# COMMAND ----------

import json

target = 'y'
attributes = data.schema.names
def fit(row):
  result = classify(row.asDict(), tree)
  check = result == row.y
  return [result, row.y, check]

# COMMAND ----------

test = data.rdd.map(fit)

# COMMAND ----------

test.collect()

# COMMAND ----------


