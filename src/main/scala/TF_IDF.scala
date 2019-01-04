
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel, SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.feature.{HashingTF, IDF, IDFModel}
import org.apache.spark.mllib.linalg
import org.apache.spark.rdd.RDD
import sparkcore.SparkCoreModule

object TF_IDF extends SparkCoreModule {

  val hashingTF = new HashingTF()

  def getHistoricalData: Unit = {

    import SPARK.sqlContext.implicits._
    import org.apache.spark.sql.functions._

    val pos = SPARK_CONTEXT.textFile("./datasets/review_polarity/txt_sentoken/pos/cv00*").toDF("Sentence")

    val neg = SPARK_CONTEXT.textFile("./datasets/review_polarity/txt_sentoken/neg/cv00*").toDF("Sentence")

    var data = pos.union(neg)

    val nlp = new NLPUDFS()

    SPARK.sqlContext.udf.register("score",nlp.score)

    SPARK.sqlContext.udf.register("sentiment",nlp.sentiment)

    data = data.withColumn("Sentiment",callUDF("sentiment",data.col("Sentence")))

    data = data.withColumn("Review",callUDF("score",data.col("Sentiment")))

    data = data.where(data.col("Review").isin(List(0,1):_*))

    data = data.select("Review","Sentence").where(col("Review").gt(1))

    data.show(false)

  }

  def tf_idf : (RDD[LabeledPoint],IDFModel/*,RDD[(Double, linalg.Vector)]*/) ={

    val pos: RDD[(Double,String)] = SPARK_CONTEXT.textFile("./datasets/review_polarity/txt_sentoken/pos/cv00*").map(l=>{(1,l)})

    val neg: RDD[(Double,String)] = SPARK_CONTEXT.textFile("./datasets/review_polarity/txt_sentoken/neg/cv00*").map(l=>{(0,l)})

    val reviews = pos.++(neg)

    val ratings=reviews.map{x=> x._1 }

    /*
       HashingTF and IDF are helpers in MLlib that helps us vectorize our
       synopsis instead of doing it manually
     */
    val frequency_vector=reviews.map{x=>
      val stemmed=Stemmer.tokenize(x._2)
      hashingTF.transform(stemmed)
    }
    frequency_vector.cache()

    /*
       http://en.wikipedia.org/wiki/Tf%E2%80%93idf
       https://spark.apache.org/docs/1.3.0/mllib-feature-extraction.html
      */
    val idf: IDFModel = new IDF().fit(frequency_vector)

    val tfidf=idf.transform(frequency_vector)

    /*produces (rating,vector) tuples*/

    val zipped: RDD[(Double, linalg.Vector)] =ratings.zip(tfidf)

    /*Now we transform them into LabeledPoints*/

    //val Array(training, test) = zipped.randomSplit(Array(0.7, 0.3))

    val labeledPoints: RDD[LabeledPoint] = zipped.map{case (label,vector)=> LabeledPoint(label,vector)}

    (labeledPoints,idf)

  }

  /*def getTestVectors :  RDD[linalg.Vector] ={

    val read = new ReadHTML

    val testDataFile: RDD[String] = read.getReviews.map(_._2)

    /*We only have synopsis now. The rating is what we want to achieve.*/
    val testVectors: RDD[linalg.Vector] =testDataFile.map{ x=>
      val stemmed=Stemmer.tokenize(x)
      hashingTF.transform(stemmed)
    }

    testVectors.cache()

    testVectors

  }*/


  def getPredictionsBysvm(labeledPoints:RDD[LabeledPoint],idf:IDFModel,test : RDD[linalg.Vector]): Unit ={


    val numIterations = 100

    val model: SVMModel = SVMWithSGD.train(labeledPoints, numIterations)

    model.clearThreshold()

    val scoreAndLabels = labeledPoints.map { point =>
      val score = model.predict(point.features)
      (score, point.label)
    }
    // Get evaluation metrics.
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)

    val auROC = metrics.areaUnderROC()

    val tfidf_test: RDD[linalg.Vector] = idf.transform(test)

    getPredictionsByParallel(Left(model),tfidf_test)

  }


  def getPredictionsByNB(labeledPoints:RDD[LabeledPoint],idf:IDFModel,test : RDD[linalg.Vector]): Unit ={
     time {
       val model: NaiveBayesModel = NaiveBayes.train(labeledPoints)
       /*--- Model is trained now we get it to classify our test file with only synopsis ---*/
       val tfidf_test: RDD[linalg.Vector] = idf.transform(test)
       getPredictionsByParallel(Right(model), tfidf_test)
     }
  }

  def getPredictionsByParallel(model : Either[SVMModel,NaiveBayesModel],tfidf_test: RDD[linalg.Vector]): RDD[Double] ={
    time {
      model match {
        case Right(x) => x.predict(tfidf_test)
        case Left(x) => x.predict(tfidf_test)
      }
    }
  }

  def getPredictionsBySeq(model : Either[SVMModel,NaiveBayesModel],tfidf_test: RDD[linalg.Vector]): Unit ={
    val result = time{
      val test = tfidf_test.collect()
      model match {
        case Right(x)=>test.map(v=>x.predict(v))
        case Left(x)=>test.map(v=>x.predict(v))
      }
    }
  }

  def time[R](block: => R): R = {
    val t0 = System.nanoTime()
    val result = block
    val t1 = System.nanoTime()
    println("Elapsed time: " + ((t1 - t0) / 1000000000.0) + " sec")
    result
  }


  def main(args: Array[String]): Unit = {

    val read = new ReadHTML

    val rdd = read.getReviews

    val (labeledPoints,idf) = tf_idf

    rdd.collect().grouped(200).toList.foreach(v=>{
      getPredictionsByNB(labeledPoints,idf,SPARK_CONTEXT.parallelize(v))
    })

  }


}


//https://www.tonytruong.net/movie-rating-prediction-with-apache-spark-and-hortonworks/