data = spark.read.csv("s3://decisiontree-bigdata/Titanic_train.csv", header=True, inferSchema=True)

clearData = data.na.drop()

filtered = data.select("PassengerId", "Survived", "Sex", "SibSp", "Parch", "Embarked").na.drop()

filtered.show()

filtered.schema.names

from pyspark.sql.functions import *
from pyspark.sql.types import *
import copy

def calTargetEntropy(data, target):
  targetCol = data.select(target) 
  total = targetCol.count() 
  targetColMap = targetCol.groupBy(target).count()
  targetEntropyCol = targetColMap.withColumn("probability", col("count") / lit(total)).withColumn("entropy", - col("probability") * log(2.0, col("probability"))).agg({"entropy": "sum"})
  return targetEntropyCol.collect()[0][0]

def filterData(data, colName, value):
  filtered = data.filter(col(colName) == value).drop(colName)
  return filtered

def getBestFeature(data, target, serial):
  columns = data.schema.names
  bestGain = -1
  bestFeature = ""
  total = data.count()
  targetEntropy = calTargetEntropy(data, target)
  for name in columns:
    if name == target or name == serial: continue
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

def getMajorClass(data, target):
  return data.select(target).groupBy(target).count().sort(col("count").desc()).collect()[0][0]

def createDecisionTree(data, target, serial): 
  schema = copy.deepcopy(data.schema.names)
  schema.remove(target)
  schema.remove(serial)
  
  targetClasses = data.select(target).distinct().rdd.map(lambda r : r[0]).collect()
  # stop when only one category left
  if len(targetClasses) <= 1:
    return targetClasses[0]
  
  # return major classes when only one feature left
  if len(schema) <= 0:
    return getMajorClass(data, target)
  
  bestFeature = getBestFeature(data, target, serial)
  
  tree = {bestFeature: {}}

  values = data.select(bestFeature).distinct().rdd.map(lambda r : r[0]).collect()
  for value in values:
    if not value:
        nullValue = value
    tree[bestFeature][value] = createDecisionTree(filterData(data, bestFeature, value), target, serial)
    
  return tree

tree = createDecisionTree(filtered, "Survived", "PassengerId")

tree

def classify(dataDict, tree):
  for key in tree.keys():
    secondDict = tree[key]
    value = dataDict[key]
    
    if value == None:
        return "null"
    
    if value not in secondDict.keys():
        for secKey in secondDict.keys():
            nextDict = secondDict[secKey]
    else:  
        nextDict = secondDict[value]

  if isinstance(nextDict, dict):
    return classify(dataDict, nextDict)
  else:
    return nextDict

def fit(row):
  result = classify(row.asDict(), tree)
  return [row.PassengerId, result]

testData = spark.read.csv("s3://decisiontree-bigdata/Titanic_test.csv", header=True, inferSchema=True)

totalCount = testData.count()

totalCount

test = testData.rdd.map(fit)

test.collect()

def toCSVLine(data):
  return ','.join(str(d) for d in data)

header = sc.parallelize([['PassengerId', 'Survived']])

finalResult = header.union(test)

finalResult.collect()

finalResult.map(toCSVLine).saveAsTextFile("s3://decisiontree-bigdata/result.csv")


