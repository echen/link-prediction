import com.twitter.scalding._
import com.twitter.scalding.mathematics._

/**
 * Calculate personalized PageRank scores.
 *
 * @author Edwin Chen
 */
class PersonalizedPageRankInScalding(args : Args) extends Job(args) {

  import Matrix._
  
  // TSV file with (user1, user2) columns, meaning that user1 follows user2.  
  val GraphFilename = args.getOrElse("graph", "graph.csv")
  
  // Output TSV file containing the personalized PageRank scores.
  val OutputFilename = args("output")
  
  // Number of personalized PageRank scores to output for each user.
  val NumScores = 50

// ***********************************************
// STEP 1. Load the adjacency graph into a matrix.
// ***********************************************

  val following = Tsv(GraphFilename, ('user1, 'user2, 'weight))

  // Binary matrix where cell (u1, u2) means that u1 follows u2.
  val followingMatrix = following.toMatrix[Int,Int,Double]('user1, 'user2, 'weight)

  // Binary matrix where cell (u1, u2) means that u1 is followed by u2.  
  val followerMatrix = followingMatrix.transpose

  // Note: we could also form this adjacency matrix differently, by placing different
  // weights on the following vs. follower edges.
  val undirectedAdjacencyMatrix = (followingMatrix + followerMatrix).rowL1Normalize

  // Create a diagonal users matrix (to be used in the "teleportation back home" step).
  val usersMatrix =
    following.unique('user1)
             .map('user1 -> ('user2, 'weight)) { user1: Int => (user1, 1) }
             .toMatrix[Int, Int, Double]('user1, 'user2, 'weight)

// ***************************************************
// STEP 2. Compute the personalized PageRank scores.
// See http://nlp.stanford.edu/projects/pagerank.shtml
// for more information on personalized PageRank.
// ***************************************************

  // Compute personalized PageRank by running for three iterations,
  // and output the top candidates.
  val pprScores = personalizedPageRank(usersMatrix, undirectedAdjacencyMatrix, usersMatrix, 0.5, 3)
  pprScores.topRowElems(numCandidates).write(Tsv(OutputFilename))

  /**
   * Performs a personalized PageRank iteration. The ith row contains the
   * personalized PageRank probabilities around node i.
   *
   * Note the interpretation: 
   *   - with probability 1 - alpha, we go back to where we started.
   *   - with probability alpha, we go to a neighbor.
   *
   * Parameters:
   *   
   *   startMatrix - a (usually diagonal) matrix, where the ith row specifies
   *                 where the ith node teleports back to.
   *   adjacencyMatrix
   *   prevMatrix - a matrix whose ith row contains the personalized PageRank
   *                probabilities around the ith node.
   *   alpha - the probability of moving to a neighbor (as opposed to teleporting
   *           back to the start).
   *   numIterations - the number of personalized PageRank iterations to run. 
   */
  def personalizedPageRank(startMatrix: Matrix[Int, Int, Double],
                           adjacencyMatrix: Matrix[Int, Int, Double],
                           prevMatrix: Matrix[Int, Int, Double],
                           alpha: Double,              
                           numIterations: Int): Matrix[Int, Int, Double] = {                
      if (numIterations <= 0) {
        prevMatrix
      } else {
        val updatedMatrix = startMatrix * (1 - alpha) + (prevMatrix * adjacencyMatrix) * alpha
        personalizedPageRank(startMatrix, adjacencyMatrix, updatedMatrix, alpha, numIterations - 1)
      }
  }
}
