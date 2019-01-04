
import TF_IDF.hashingTF
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.linalg
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, RowFactory}
import org.apache.spark.sql.types.{DataTypes, StructField}
import org.jsoup.Jsoup
import sparkcore.SparkCoreModule

import scala.collection.JavaConverters._
import scala.collection.mutable.ListBuffer

class ReadHTML extends SparkCoreModule {

  def getReviews: RDD[linalg.Vector] ={

    val html: RDD[(String,String)] = SPARK_CONTEXT.wholeTextFiles("/home/leadsemantics-08/work/product/development/srikanthbackup/mywork/LC/datasets/movie/000*")
    val name: StructField = DataTypes.createStructField("Name", DataTypes.StringType, true)
    val sentence: StructField = DataTypes.createStructField("para", DataTypes.createArrayType(DataTypes.StringType), true)
    var list = new ListBuffer[StructField]()
    list += name
    list += sentence
    val schemata = DataTypes.createStructType(list.toArray)
    val dataRDD: RDD[Row] =html.map(v=>{
      val doc = Jsoup.parse(v._2)
      doc.body().getElementsByTag("p").eachText().asScala
      RowFactory.create(doc.title(),doc.body().getElementsByTag("p").eachText().asScala)
    })

    var data = SPARK.createDataFrame(dataRDD,schemata)

    import org.apache.spark.sql.functions._
    data.select("")

    data = data.withColumn("Sentence",explode(data.col("para")))

    //data.drop(data.col("para"))

    /*val nlp = new NLPUDFS()

    SPARK.sqlContext.udf.register("score",nlp.score)

    SPARK.sqlContext.udf.register("sentiment",nlp.sentiment)

    data = data.withColumn("Sentiment",callUDF("sentiment",data.col("Sentence")))

    data = data.withColumn("Review",callUDF("score",data.col("Sentiment")))

    data = data.where(data.col("Review").isin(List(0,1):_*))

    data = data.select("Review","Sentence")*/

    data.show(false)

    data.rdd.map{ x=>
      val stemmed=Stemmer.tokenize(x.getString(2))
      hashingTF.transform(stemmed)
    }

   // data.rdd.map(row=> (row.getDouble(0),row.getString(1)))

  }

}
