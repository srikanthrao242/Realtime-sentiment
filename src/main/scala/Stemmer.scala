import org.apache.lucene.analysis.en.EnglishAnalyzer
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute
import scala.collection.mutable.ArrayBuffer

object Stemmer {

  // Adopted from
  // https://chimpler.wordpress.com/2014/06/11/classifiying-documents-using-naive-bayes-on-apache-spark-mllib/

  def tokenize(content: String): Seq[String] = {
    val analyzer = new EnglishAnalyzer()

    val tokenStream = analyzer.tokenStream("contents", content)
    //CharTermAttribute is what we're extracting
    val term = tokenStream.addAttribute(classOf[CharTermAttribute])

    tokenStream.reset() // must be called by the consumer before consumption to clean the stream

    var result = ArrayBuffer.empty[String]

    while (tokenStream.incrementToken()) {
      val termValue = term.toString
      if (!(termValue matches ".*[\\d\\.].*")) {
        result += term.toString
      }
    }
    tokenStream.end()
    tokenStream.close()
    result
  }
}