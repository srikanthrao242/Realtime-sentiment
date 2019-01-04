package sparkcore

import java.io.File

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.sql.{SQLContext, SparkSession}

/**
  * Created by srikanth on 5/20/18.
  */
trait SparkCoreModule {
  private val warehouseLocation: File = new File("spark-warehouse")
  val successful = warehouseLocation.mkdir()
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)
  val conf = new SparkConf()
  conf.setMaster("local[4]")
  conf.setAppName("LC")
  conf.set("spark.sql.warehouse.dir", warehouseLocation.getAbsolutePath)
  conf.set("spark.sql.catalogImplementation", "hive")
  conf.set("hive.metastore.warehouse.dir", warehouseLocation.getAbsolutePath)

  final implicit lazy val SPARK = SparkSession
    .builder().config(conf)
    .getOrCreate()

  final implicit lazy val SPARK_CONTEXT = SPARK.sparkContext
}
