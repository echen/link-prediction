/**
 * Given a directed graph of follower and following edges, compute personalized PageRank scores
 * around specified starting nodes.
 *
 * A personalized PageRank is similar to standard PageRank, except that when randomly teleporting 
 * to a new node, the surfer always teleports back to the given source node being personalized
 * (rather than to a node chosen uniformly at random, as in the standard PageRank algorithm).
 *
 * In other words, the random surfer in the personalized PageRank model works as follows:
 * - He starts at the source node X that we want to calculate a personalized PageRank around.
 * - At step i: with probability p, the surfer moves to a neighboring node chosen uniformly at random; 
 *              with probability $1-p$, the surfer instead teleports back to the original source node X.
 *
 * The limiting probability that the surfer is at node N is the personalized PageRank score of node N around X.
 *
 * @author Edwin Chen
 */
case class PersonalizedPageRank(val followers: Map[Int, Set[Int]], val followings: Map[Int, Set[Int]]) {
  
  val NumIterations = 3
  val MaxNodesToKeep = 25
  
  /**
   * Calculate a personalized PageRank around the given user, and return a list of the
   * nodes with the highest personalized PageRank scores.
   *
   * @return A list of (node, probability of landing at this node after running a personalized
   *         PageRank for K iterations) pairs.
   */
  def pageRank(user: Int): List[(Int, Double)] = {
    // This map holds the probability of landing at each node, up to the current iteration.
    val probs = Map[Int, Double]()    
    probs(user) = 1 // We start at this user.
    
    val pageRankProbs = pageRankHelper(user, probs, NumIterations)
    pageRankProbs.toList
                 .sortBy { -_._2 }
  }
  
  /**
   * Simulates running a personalized PageRank for one iteration.
   *
   * Parameters:
   * start - the start node to calculate the personalized PageRank around
   * probs - a map from nodes to the probability of being at that node at the start of the current
   *         iteration
   * numIterations - the number of iterations remaining
   * alpha - with probability alpha, we follow a neighbor; with probability 1 - alpha, we teleport
   *         back to the start node
   *
   * @return A map of node -> probability of landing at that node after the specified number of iterations.
   */
  def pageRankHelper(start: Int, probs: Map[Int, Double], numIterations: Int, alpha: Double = 0.5): Map[Int, Double] = {
    if (numIterations <= 0) {
      probs
    } else {
      // This map holds the updated set of probabilities, after the current iteration.
      val probsPropagated = Map[Int, Double]()
      
      // With probability 1 - alpha, we teleport back to the start node.
      probsPropagated(start) = 1 - alpha
    
      // Propagate the previous probabilities...
      probs.foreach { case (node, prob) =>      
        val forwards = getFollowings(node)
        val backwards = getFollowers(node)
      
        // With probability alpha, we move to a follower...
        // And each node equally distributes its current probability to its neighbors.
        val probToPropagate = alpha * prob / (forwards.size + backwards.size)        
        (forwards.toList ++ backwards.toList).foreach { neighbor =>
          if (!probsPropagated.contains(neighbor)) {
            probsPropagated(neighbor) = 0
          }
          probsPropagated(neighbor) += probToPropagate
        }
      }
    
      pageRankHelper(start, probsPropagated, numIterations - 1, alpha)
    }
  }

  def getFollowers(user: Int) = {
    followers.get(user).getOrElse(Set[Int]())
  }
  
  def getFollowings(user: Int) = {
    followings.get(user).getOrElse(Set[Int]())
  }
}
