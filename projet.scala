import org.apache.spark.sql.functions._
import scala.math.sqrt
import org.apache.spark.mllib.linalg._
import org.apache.spark.sql.Row
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.RandomForest 
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator


var df = spark.read.format("csv").option("header", "true").load("creditcard.csv")

var dataFrameValues = Seq.empty[(String,Float,Float)].toDF("column_name","max","min")

for (i<-0 to df.columns.size-1) {

    var maxi = df.agg(max(df.columns(i))).first.get(0).toString.toFloat
    var mini = df.agg(min(df.columns(i))).first.get(0).toString.toFloat
    var valuesToAppend = Seq( (df.columns(i),maxi,mini)).toDF()
    dataFrameValues = dataFrameValues.union(valuesToAppend)

}

var meanValues = Seq.empty[(String,Float)].toDF("column_name","mean")
for (i<-0 to df.columns.size-1) {
    var mean = df.agg(sum(df.columns(i))).first.get(0).toString.toFloat/df.columns.size
    var valuesToAppend = Seq((df.columns(i),mean)).toDF()
    meanValues =meanValues.union(valuesToAppend)
}

var standardDeviationValues = Seq.empty[(String,Double)].toDF("column_name","standard_deviation")
var meanArray = meanValues.select("mean").rdd.map(r => r(0).toString.toFloat).collect()
for (i<-0 to df.columns.size-1) {
    var values = df.select(df.columns(i)).rdd.map(r => r(0).toString.toFloat).collect()
    var variance =0F
    for (j<-0L to df.count()-1) {
        variance = variance + (values(j.toInt) - meanArray(i))*(values(j.toInt)-meanArray(i))
    }
    variance = variance / (df.count()-1)
    var standard_deviation = sqrt(variance)
    var valuesToAppend = Seq((df.columns(i),standard_deviation)).toDF()
    standardDeviationValues = standardDeviationValues.union(valuesToAppend)
}

var alteredDataFrame = spark.createDataFrame(sc.emptyRDD[Row],df.schema)

//create time alteredColumn dataframe and then append other columns to that dataframe
var TimeValue = df.select(df.columns(0)).rdd.map(r => r(0).toString.toDouble).collect()
var TimeMeanValue = meanValues.take(1)(0)(1).toString.toDouble
for (j<-0 to (TimeValue.size-1)/10 ) {
    if (TimeValue(j)==null) {
        TimeValue(j) = TimeMeanValue
    }
}
var AlteredDataFrame = Seq(TimeValue).flatten.toDF(df.columns(0))
AlteredDataFrame = AlteredDataFrame.withColumn("id1",monotonically_increasing_id)
for (i<-1 to df.columns.size-1) {
    var columnsValue = df.select(df.columns(i)).rdd.map(r => r(0).toString.toDouble).collect()
    var columnMeanValue = meanValues.take(i+1)(i)(1).toString.toDouble
    for (j<-0 to (columnsValue.size-1)/10 ) {
        if (columnsValue(j)==null) {
            columnsValue(j) = columnMeanValue
        }
    }
    var values = Seq(columnsValue).flatten.toDF(df.columns(i))
    values = values.withColumn("id2",monotonically_increasing_id)
    AlteredDataFrame = AlteredDataFrame.join(values,col("id1")===col("id2"),"inner").drop("id2")
}

AlteredDataFrame.drop("id1")

df = AlteredDataFrame

val startTimeDense= System.nanoTime
val vectorDenseData = df.rdd.map(
    s=> Vectors.dense(s.toString().substring(1,s.toString().length-1).split(",").map(_.toDouble)))
val labeledPointDense = vectorDenseData.map{
    v=> var features: Vector = Vectors.dense(v(0),v(1),v(2),v(3),
    v(4),v(5),v(6),v(7),v(8),v(9),v(10),v(11),v(12),v(13),v(14),
    v(15),v(16),v(17),v(18),v(19),v(20),v(21),v(22),v(23),v(24),v(25),v(26),v(27),v(28),v(29));LabeledPoint(v(30),features)}
val endTimeDense = System.nanoTime
val durationDense = endTimeDense - startTimeDense


val indices:Array[Int] = (0 to 29 map(n => n)).toArray
val startTimeSparse= System.nanoTime
val labeledPointSparse = df.rdd.map{ 
    s =>
    var l = s.toString().substring(1,s.toString().length-1).split(",").map(_.toDouble);
    var  featuressparse:Vector =
    Vectors.sparse(30,indices,Array(l(0),l(1),l(2),l(3),l(4),l(5),l(6),
    l(7),l(8),l(9),l(10),l(11),l(12),l(13),l(14),
    l(15),l(16),l(17),l(18),l(19),l(20),l(21),l(22),l(23),l(24),
    l(25),l(26),l(27),l(28),l(29)));
    LabeledPoint(l(30),featuressparse)}
val endTimeSparse = System.nanoTime
val durationSparse = endTimeSparse - startTimeSparse

if (durationDense > durationDense) {
    println("Dense prend plus de temps")
} else {
    println("Sparse prend plus de temps")
}

val Array(trainingDataDense, testDataDense) = labeledPointDense.randomSplit(Array(0.7,0.3))
val Array(trainingDataSparse, testDataSparse) = labeledPointSparse.randomSplit(Array(0.7,0.3))


val numClasses = 2
val categoricalFeaturesInfo = Map[Int, Int]()
val impurity = "gini"
val maxDepth = 5
val maxBins = 32

val DTmodelDense = DecisionTree.trainClassifier(trainingDataDense, numClasses,categoricalFeaturesInfo, impurity, maxDepth, maxBins)
val DTmodelSparse = DecisionTree.trainClassifier(trainingDataSparse, numClasses,categoricalFeaturesInfo, impurity, maxDepth, maxBins)

val RFmodelDense = RandomForest.trainClassifier(trainingDataDense,numClasses,categoricalFeaturesInfo,10,"auto","gini",4,100,42) //the last values are recommended based on documentation
val RFmodelSparse =RandomForest.trainClassifier(trainingDataSparse,numClasses,categoricalFeaturesInfo,10,"auto","gini",4,100,42)

val labelAndPredsDenseDT = trainingDataDense.map{ point => val prediction = DTmodelDense.predict(point.features); (point.label,prediction) }
val ErrDenseDT = labelAndPredsDenseDT.filter(r => r._1 != r._2).count().toDouble / trainingDataDense.count()
val AccuracyDenseDT = 1-ErrDenseDT

val labelAndPredsSparseDT = trainingDataSparse.map{ point => val prediction = DTmodelSparse.predict(point.features); (point.label,prediction) }
val ErrSparseDT = labelAndPredsSparseDT.filter(r => r._1 != r._2).count().toDouble / trainingDataSparse.count()
val AccuracySparseDT = 1-ErrSparseDT

val labelAndPredsDenseRF = trainingDataDense.map{ point => val prediction = RFmodelDense.predict(point.features); (point.label,prediction) }
val ErrDenseRF = labelAndPredsDenseRF.filter(r => r._1 != r._2).count().toDouble / trainingDataDense.count()
val AccuracyDenseRF = 1-ErrDenseRF

val labelAndPredsSparseRF = trainingDataSparse.map{ point => val prediction = RFmodelSparse.predict(point.features); (point.label,prediction) }
val ErrSparseRF = labelAndPredsSparseRF.filter(r => r._1 != r._2).count().toDouble / trainingDataSparse.count()
val AccuracySparseRF = 1-ErrSparseRF


val denseFeaturesVectors = labeledPointDense.map(lp => lp.features)
val scaler = new StandardScaler(withMean = true, withStd = true).fit(denseFeaturesVectors)
val scaledDataDense = labeledPointDense.map(l => LabeledPoint(l.label,scaler.transform(l.features)))
val Array(trainingDataDense2, testDataDense2) = scaledDataDense.randomSplit(Array(0.7, 0.3))
val DTmodelDense2 = DecisionTree.trainClassifier(trainingDataDense2, numClasses,categoricalFeaturesInfo, impurity, maxDepth, maxBins)
val labelAndPreds2 = testDataDense2.map{point => val prediction = DTmodelDense2.predict(point.features) ;(point.label, prediction) }
val ErrDenseDT2 = labelAndPreds2.filter(r => r._1 != r._2).count().toDouble / testDataDense2.count()
println(" Prediction error = " + ErrDenseDT2)
println(" Accuracy = " +(1- ErrDenseDT2))

val sparseFeaturesVectors = labeledPointSparse.map(lp => lp.features)
val scaler = new StandardScaler(withMean = true, withStd = true).fit(sparseFeaturesVectors)
val scaledDataSparse = labeledPointSparse.map(l => LabeledPoint(l.label,scaler.transform(l.features)))
val Array(trainingDataSparse2, testDataSparse2) = scaledDataDense.randomSplit(Array(0.7, 0.3))
val DTmodelSparse2 = DecisionTree.trainClassifier(trainingDataSparse2, numClasses,categoricalFeaturesInfo, impurity, maxDepth, maxBins)
val labelAndPreds2 = testDataSparse2.map{point => val prediction = DTmodelSparse2.predict(point.features) ;(point.label, prediction) }
val ErrSparseDT2 = labelAndPreds2.filter(r => r._1 != r._2).count().toDouble / testDataSparse2.count()
println(" Prediction error = " + ErrSparseDT2)
println(" Accuracy = " +(1- ErrSparseDT2))