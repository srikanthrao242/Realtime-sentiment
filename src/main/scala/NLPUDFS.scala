import java.util.Properties

import edu.stanford.nlp.ling.CoreAnnotations
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations
import edu.stanford.nlp.pipeline.StanfordCoreNLP
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations
import edu.stanford.nlp.simple.{Document, Sentence}
import org.apache.spark.sql.functions.udf
import scala.collection.JavaConverters._

class NLPUDFS extends Serializable{

  private var pipeline: StanfordCoreNLP = _

  def getOrCreateSentimentPipeline() ={
    if (pipeline == null) {
      val props = new Properties
      props.put("annotators", "tokenize, ssplit, parse, sentiment")
      //Alternatively run with parse model -> increase speed by a fator of about 16 -> see Readme.md
      //			props.put("annotators", "tokenize, ssplit, pos, parse, sentiment");
      //			props.put("parse.model", "edu/stanford/nlp/models/srparser/englishSR.beam.ser.gz");
      pipeline = new StanfordCoreNLP(props)
    }
    pipeline
  }
  val upperUDF = udf { s: String => s.toUpperCase }

  /**
    * Splits a document into sentences.
    */

  val ssplit =udf{
    document : String=>{
      new Document(document).sentences.asScala.map(s => s.text()).toArray
    }
  }

  /**
    * Tokenizes a sentence into words.
    */

  val tokenize = udf{
    sentence:String=>{
      new Sentence(sentence).words.asScala.toArray
    }
  }

  /**
    * Generates the part of speech tags of the sentence.
    */

  val pos = udf{
    sentence:String=>{
      new Sentence(sentence).posTags.asScala.toArray
    }
  }
  /**
    * Generates the word lemmas of the sentence.
    */

  val lemma = udf{
    sentence:String=>{
      new Sentence(sentence).lemmas.asScala.toArray
    }
  }
  /**
    * Generates the named entity tags of the sentence.
    */
  val ner = udf{
    sentence:String=>{
      new Sentence(sentence).nerTags.asScala.toArray
    }
  }

  /**
    * Measures the sentiment of an input sentence on a scale of 0 (strong negative) to 4 (strong positive)
    * If the input contains multiple sentences, only the first one is used.
    */

  val sentiment = udf{
    sentence: String=>{
      val pipeline = getOrCreateSentimentPipeline()
      val annotation = pipeline.process(sentence)
      val tree = annotation.get(classOf[CoreAnnotations.SentencesAnnotation]).get(0).get(classOf[SentimentCoreAnnotations.SentimentAnnotatedTree])
      RNNCoreAnnotations.getPredictedClass(tree)
    }
  }

  /**
    * Map tree labels for average sentiment calculated for a paragraph (multiple sentences -> )
    * If avg sentiment is positive and less than 2.0 than sentiment labelk is neg,
    * if is equal to 2.0 sentiment lable is neu (neutral) if id greater than 2 and less or equal to 4,
    * than label is pos.
    * Sentiment label unk means that sentences tree contains no sentence
    */

  val score = udf{
    d:Double=>{
      var sentiment :Double = -1
      if (d < 1.5 && d >= 0.0) sentiment = 0
      else if (d >= 3 && d <= 4.0) sentiment = 1
      else sentiment = -1
      sentiment
    }
  }

}
