import scala.collection.mutable.Map
import scala.collection.mutable.Set
import scala.io.Source

/**
 * Given a graph and a set of nodes, calculate a propagation score
 * (similar in form to a personalized PageRank) for each node.
 *
 * The propagation score starting at a node N is defined as follows:
 * - Give node N an initial score S.
 * - Propagate the score equally to each of N's neighbors (followers and
 *   followings).
 * - Each first-level neighbor then keeps half of its score and then
 *   propagates the (original, non-halved) score to its neighbors.
 * - These second-level neighbors then repeat the process, except that
 *   neighbors traveled to via a backwards/follower link don't keep
 *   half of their score.
 * 
 * Note: the original form of this algorithm comes from a post on
 * the Kaggle forum by Den Souleo.
 *
 * @author Edwin Chen
 */
object PropagationScore {
  
  // A CSV (with header) file of src, dest. (where src is following dest)
  val TrainingSetFilename = "data/train.csv"

  def main(args: Array[String]) = {
    val followers = Map[Int, Set[Int]]()    
    val followings = Map[Int, Set[Int]]()
    
    // Read in the training graph.
    Source.fromFile(TrainingSetFilename).getLines().drop(1).foreach { line: String =>
      val Array(src, dest) = line.split(",").map { _.toInt }
      
      if (!followers.contains(dest)) {
        followers(dest) = Set[Int]()
      }
      
      if (!followings.contains(src)) {
        followings(src) = Set[Int]()
      }
      
      followers(dest) += src
      followings(src) += dest      
    }
        
    // Read in the test nodes that we want to calculate propagation scores on.
    val testSetNodes = Source.fromFile("data/test.csv").getLines().drop(1).map { _.toInt }
     
    // Calculate propagation scores on the test nodes.
    val ps = PropagationScore(followers, followings)        
    testSetNodes.foreach { node =>
      val recommendedNodesAndScores = ps.propagate(node)
      recommendedNodesAndScores.foreach { x => println(node + "\t" + x._1 + "\t" + x._2) }
    }
  }
}

case class PropagationScore(val followers: Map[Int, Set[Int]], val followings: Map[Int, Set[Int]]) {

  // The number of propagation iterations to perform.
  // This is also the max distance the starting node propagates out to.
  val NumIterations = 3
  
  // The maximum number of nodes (sorted by propagation score) to keep.
  val MaxNodesToKeep = 25
  
  /**
   * Calculate propagation scores around the current user.
   *
   * In the first propagation round, we
   *
   * - Give the starting node N an initial score S.
   * - Propagate the score equally to each of N's neighbors (followers 
   *   and followings).
   * - Each first-level neighbor then duplicates and keeps half of its score
   *   and then propagates the original again to its neighbors.
   *
   * In further rounds, neighbors then repeat the process, except that neighbors traveled to 
   * via a backwards/follower link don't keep half of their score.
   *
   * @return a sorted list of (node, propagation score) pairs.
   */
  def propagate(user: Int): List[(Int, Double)] = {
    val scores = Map[Int, Double]()    
    
    // We propagate the score equally to all neighbors.
    val scoreToPropagate = 1.0 / (getFollowings(user).size + getFollowers(user).size)

    (getFollowings(user).toList ++ getFollowers(user).toList).foreach { x =>
      // Propagate the score...
      continuePropagation(scores, x, scoreToPropagate, 1)
      // ...and make sure it keeps half of it for itself.
      scores(x) = scores.getOrElse(x, 0: Double) + scoreToPropagate / 2      
    }
    
    scores.toList.sortBy { -_._2 }
                 .filter { nodeAndScore =>
                   val node = nodeAndScore._1
                   !getFollowings(user).contains(node) && node != user
                  }
                  .take(MaxNodesToKeep)
  }
  
  /**
   * In further rounds, neighbors repeat the process above, except that neighbors traveled to 
   * via a backwards/follower link don't keep half of their score.
   */
  def continuePropagation(scores: Map[Int, Double], user: Int, score: Double, currIteration: Int): Unit = {
    if (currIteration < NumIterations && score > 0) {
      val scoreToPropagate = score / (getFollowings(user).size + getFollowers(user).size)
      
      getFollowings(user).foreach { x =>
        // Propagate the score...        
        continuePropagation(scores, x, scoreToPropagate, currIteration + 1)        
        // ...and make sure it keeps half of it for itself.        
        scores(x) = scores.getOrElse(x, 0: Double) + scoreToPropagate / 2
      }
      
      getFollowers(user).foreach { x =>
        // Propagate the score...
        continuePropagation(scores, x, scoreToPropagate, currIteration + 1)
        // ...but backward links (except for the starting node's immediate neighbors)
        // don't keep any score for themselves.
      }
    }
  }
  
  def getFollowers(user: Int): Set[Int] = {
    followers.get(user).getOrElse(Set[Int]())
  }
  
  def getFollowings(user: Int): Set[Int] = {
    followings.get(user).getOrElse(Set[Int]())
  }
  
}
