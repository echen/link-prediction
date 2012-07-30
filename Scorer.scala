import scala.collection.mutable.Map
import scala.io._

/**
 * Given a file of predictions on a test set, score these predictions
 * according to the mean average precision.
 *
 * @author Edwin Chen
 */
object Scorer {
    
  def main(args: Array[String]) = {    
    // Read in our predictions.    
    val myPredictionsFile = args(0) // Example row: "23,15 25 9 50" means that we predict user 23 is following users 15, 25, 9, and 50.    
    val myPredictions: List[(Int, Array[Int])] = 
      Source.fromFile(myPredictionsFile).getLines().map { line: String =>
        val Array(src, predictedDests) = line.split(",")
        (src.toInt, predictedDests.split(" ").map { _.toInt }.distinct)
      }.toList
      
    // Read in the set of followings that we want to predict.
    val followings = Map[Int, Set[Int]]()
    Source.fromFile("my_data/my_test_with_responses.csv").getLines().foreach { line: String =>
      val Array(src, actualDests) = line.split(",")
      followings(src.toInt) = actualDests.split(" ").map { _.toInt }.toSet
    }
    
    // Calculate the average precisions for each user.
    val averagePrecisions =
      myPredictions.map { case (src, predictedDests) =>
        calculateAveragePrecision(predictedDests, getFollowings(src.toInt, followings))
      }
      
    // Calculate the mean of these average precisions.
    val map = if (averagePrecisions.size == 0) {
                0.0
              } else {
                averagePrecisions.sum / averagePrecisions.size
              }

    println(map)
  }

  /**
   * Calculate the average precision of a sequence of predicted followings, given the
   * true set.
   * See http://www.kaggle.com/c/FacebookRecruiting/details/Evaluation for more details
   * on the average precision metric.
   *
   * Examples:
   *
   *   calculateAveragePrecision( [A, B, C], [A, C, X] )
   *     => (1/1 + 2/3) / 3 ~ 0.56
   *
   *   calculateAveragePrecision( [A, B], [A, B, C] )
   *     => (1/1 + 2/2) /3 ~ 0.67
   */
  def calculateAveragePrecision(predictions: Seq[Int], actuals: Set[Int]) : Float = {
    var numCorrect = 0
    val precisions = 
      predictions.zipWithIndex.map { case (prediction, index) =>
        if (actuals.contains(prediction)) {
          numCorrect += 1
          numCorrect.toFloat / (index + 1)
        } else {
          0
        }
      }
    
    if (actuals.size == 0) 0 else (precisions.sum / actuals.size)
  }
  
  def getFollowings(source : Int, followings : Map[Int, Set[Int]]) : Set[Int] = {
    followings.getOrElse(source, Set[Int]())
  }

}