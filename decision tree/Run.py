data = spark.read.csv("s3://decisiontree-bigdata/Titanic_train.csv", header=True, inferSchema=True)

data.show()

data.printSchema

filtered = data.select("Survived", "Sex", "SibSp", "Parch", "Embarked").na.drop()

filtered.show()

from pyspark.sql.functions import *
from pyspark.sql.types import *

def calTargetEntropy(data, target):
  targetCol = data.select(target) 
  total = targetCol.count() 
  targetColMap = targetCol.groupBy(target).count()
  targetEntropyCol = targetColMap.withColumn("probability", col("count") / lit(total)).withColumn("entropy", - col("probability") * log(2.0, col("probability"))).agg({"entropy": "sum"})
  return targetEntropyCol.collect()[0][0]

def filterData(data, colName, value):
  filtered = data.filter(col(colName) == value).drop(colName)
  return filtered

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
    if targetEntropy - entropy >= bestGain: # use larger or equal to avoid returning emtpy string
      bestGain = targetEntropy - entropy
      bestFeature = name
  return bestFeature

def getMajorClass(data, target):
  return data.select(target).groupBy(target).count().sort(col("count").desc()).collect()[0][0]

def createDecisionTree(data, target): 
  schema = data.schema.names
 # if target not in schema or data.count() < 1:
 #   return "null"
  
  targetClasses = data.select(target).distinct().rdd.map(lambda r : r[0]).collect()
  # stop when only one category left
  if len(targetClasses) <= 1:
    return targetClasses[0]
  
  # return major classes when only one feature left
  if len(data.schema.names) <= 1:
    return getMajorClass(data, target)
  
  bestFeature = getBestFeature(data, target)
  
  tree = {bestFeature: {}}

  columns = data.schema.names
  values = data.select(bestFeature).distinct().rdd.map(lambda r : r[0]).collect()
  for value in values:
    if not value:
        nullValue = value
    tree[bestFeature][value] = createDecisionTree(filterData(data, bestFeature, value), target)
    
  return tree



tree = createDecisionTree(filtered, "Survived")

tree

def classify(dataDict, tree):
  for key in tree.keys():
    secondDict = tree[key]
    value = dataDict[key]
    
    if value == None:
        return "null"
    nextDict = secondDict[value]

  if isinstance(nextDict, dict):
    return classify(dataDict, nextDict)
  else:
    return nextDict

def fit(row):
  global emptyTree, emptySecDict, emptyDataDict, emptyKey
  result = classify(row.asDict(), tree)
  check = result == row.Survived
  return [result, row.Survived, check]

test = data.rdd.map(fit)

test.collect()

emptyValue

emptyDataDict

totalTest = test.count()

final = test.map(lambda row: row[2]).map(lambda cell: (cell, 1)).reduceByKey(lambda x,y: x + y)

final.collect()

ans = final.mapValues(lambda x: x / totalTest)

ans.collect()






