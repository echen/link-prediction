import com.twitter.scalding._
import com.twitter.scalding.mathematics._

/**
 * Calculate some graph-based similarity measures using Scalding's Matrix library.
 *
 * For example, we calculate the Jaccard similarity between users when users are
 * represented as sets of their followers.
 *
 * I didn't use this code for the competition, but it's a nice illustration of what
 * Scalding can do.
 *
 * @author Edwin Chen
 */
class GraphSimilarityFeatures(args : Args) extends Job(args) {

  import Matrix._
  
  // TSV file with (user1, user2) columns, meaning that user1 follows user2.  
  val GraphFilename = args.getOrElse("graph", "graph.csv")
  
  // For each user, we keep up to this number of other similar users.
  val NumCandidatesToKeep = 25
  
// *************************************************************
// STEP 1. Load the following and follower graphs into matrices.
// *************************************************************
    
  // A binary adjacency matrix, where cell (u1, u2) means that u1 follows u2.
  val followingMatrix = 
    // TODO: Add a CSV source.
    Tsv(GraphFilename, ('user1, 'user2))
      .map(() -> 'weight) { u : Unit => 1 }
      .toMatrix[Int, Int, Double]('user1, 'user2, 'weight)

  // A binary adjacency matrix, where cell (u1, u2) means that u1 is followed by u2.
  val followerMatrix = followingMatrix.transpose
      
// **************************************************
// STEP 2. Compute various graph similarity measures.
// **************************************************
  
  /**
   * Cell (u1, u2) contains the cosine similarity between u1 and u2,
   * when u1 and u2 are represented as vectors of the users they are following.
   */
  val unnormalizedFollowingSimilarity =
    MatrixSimilarity.cosine(followingMatrix, followingMatrix)
      .topRowElems(NumCandidatesToKeep)
      .write(Tsv("unnormalized_following_cosine_similarity.tsv"))
      
  /**
   * Cell (u1, u2) contains the Jaccard similarity between u1 and u2,
   * when u1 and u2 are represented as vectors of the users they are followed by.
   */
  val unnormalizedFollowerSimilarity =
    MatrixSimilarity.jaccard(followerMatrix, followerMatrix)
      .topRowElems(NumCandidatesToKeep)    
      .write(Tsv("unnormalized_follower_jaccard_similarity.tsv"))
      
  /**
   * In general, we want to compute
   *
   *   ProbFollows(Alice, Bob) = ScoreX(Alice, User1) * ScoreY(User1, Bob) + ScoreX(Alice, User2) * ScoreY(User2, Bob) + ...
   *
   * Note that ProbFollowsMatrix = ScoreXMatrix * ScoreYMatrix.transpose.
   *
   * For example, ScoreX could be the binary following relation, ScoreYMatrix could be Jaccard similarity, and then
   * ProbFollows(Alice, Bob) would simply iterate over each of the people Alice is following, and sum up their similarity
   * with Bob.
   *
   * Other options for ScoreX:
   *   - follows (weighted, unweighted)
   *   - followedBy (weighted, unweighted)
   *
   * Other options for ScoreY:
   *   - following-based vs. follower-based similarity
   *   - cosine vs. Jaccard similarity
   *   - weighted vs. unweighted
   *   - regularized vs. unregularized
   *
   * More musings:
   *   - "Find everybody I follow. Find the people similar to them."
   *   - "Find people similar to me. Find everyone they follow."
   */  
  val similarities = followingMatrix * unnormalizedFollowerSimilarity.transpose
  
  // Remove from the similarities matrix people that are already being followed.
  val oneMinusFollowingMatrix = sims.binarizeAs[Double] - followingMatrix  
  val similaritiesWithoutAlreadyFollowing = similarities.hProd(oneMinusFollowingMatrix)
  
  // Remove from this new matrix the diagonal.
  val result = similaritiesWithoutAlreadyFollowing - similaritiesWithoutAlreadyFollowing.diagonal
  
  // For each user, print out the top N most similar users (to be used as predictions for
  // who they are following).
  result.topRowElems(25).write(Tsv("jaccard_similarity_to_followings_by_followers.tsv"))  

}
