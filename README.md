A couple weeks ago, Facebook launched a [link prediction contest on Kaggle](http://www.kaggle.com/c/FacebookRecruiting/), with the goal of recommending missing edges in a social graph. [I love investigating social networks](http://blog.echen.me/2011/09/07/information-transmission-in-a-social-network-dissecting-the-spread-of-a-quora-post/), so I dug around a little, and since I did well enough to score one of the coveted prizes, I'll share my approach here.

(For a bit of background, the contest provided a training dataset of edges, a test set of nodes, and contestants were asked to predict missing outbound edges on the test set, using [mean average precision](http://www.kaggle.com/c/FacebookRecruiting/details/Evaluation) as the evaluation metric.)

# Exploring

What does the network actually look like? I wanted to play around with the data a bit first, to get a rough feel, so I made [an app](http://link-prediction.herokuapp.com/network) to interact with the local follow network around each node.

[![1 Untrimmed Network](https://dl.dropbox.com/u/10506/blog/kaggle-fb/1_untrimmed.png)](http://link-prediction.herokuapp.com/network)

(Go ahead, click on the picture to [play with the app yourself](http://link-prediction.herokuapp.com/network). It's pretty fun.)

The node in black is a selected central node from the training set, and we perform a breadth-first walk of the graph out to a maximum distance of 3. Nodes are sized according to their distance from the center, and colored according to a chosen metric (a personalized PageRank in this case, but more on this later).

We can see that the central node is friends with three other users, two of whom (on the top-left and bottom) have fairly large, disjoint networks.

There are a bunch of dangling nodes (nodes at distance 3 with only one connection to the rest of the local network) that aren't contributing much to the picture, though, so let's remove these to reveal the core structure:

[![1 Untrimmed Network](https://dl.dropbox.com/u/10506/blog/kaggle-fb/1_network.png)](http://link-prediction.herokuapp.com/network)

Since the default view doesn't encode the distinction between following and follower relationships, we can mouse over each node to see who it follows and who it's followed by. Here, for example, is the following/follower network of one of the central node's friends:

[![1 - Friend1](https://dl.dropbox.com/u/10506/blog/kaggle-fb/1_friend1.png)](https://dl.dropbox.com/u/10506/blog/kaggle-fb/1_friend1.png)

The moused over node is highlighted in black, its friends (users who both follow the node and are followed back in turn) are colored in purple, its followees are in teal, and its followers are in orange. We can see that this node has a good mix of all three types, and also that it too is friends with one of the central node's friends ([triadic closure](http://en.wikipedia.org/wiki/Triadic_closure), unite).

Here's another following/follower network, this time of the central node's friend at the bottom:

[![1 - Friend2](https://dl.dropbox.com/u/10506/blog/kaggle-fb/1_friend2.png)](https://dl.dropbox.com/u/10506/blog/kaggle-fb/1_friend2.png)

Interestingly, while the first friend had several followers-only (in orange), the second friend has none. (Which suggests, perhaps, a new type of follow-hungry feature...)

And here's one more node, a little farther out, which has nothing but followers (a celebrity, perhaps?!):

[![1 - Celebrity](https://dl.dropbox.com/u/10506/blog/kaggle-fb/1_celebrity.png)](https://dl.dropbox.com/u/10506/blog/kaggle-fb/1_celebrity.png)

## The Quiet One

Let's take a look at another graph, one whose local network is a little smaller:

[![4 Network](https://dl.dropbox.com/u/10506/blog/kaggle-fb/4_network.png)](https://dl.dropbox.com/u/10506/blog/kaggle-fb/4_network.png)

## A Social Butterfly

And one more, one whose local network is a little larger:

[![2 Network](https://dl.dropbox.com/u/10506/blog/kaggle-fb/2_network.png)](https://dl.dropbox.com/u/10506/blog/kaggle-fb/2_network.png)

[![2 Network - Friend](https://dl.dropbox.com/u/10506/blog/kaggle-fb/2_friend.png)](https://dl.dropbox.com/u/10506/blog/kaggle-fb/2_friend.png)

Again, I encourage everyone to play around with the app [here](http://link-prediction.herokuapp.com/network), and we'll revisit the question of coloring each node later.

# Distributions

I also wanted to take a more quantitative look at the graph.

Here's the distribution of the number of followers of each node in the training set (cut off at 50 followers to better see the graph -- the maximum number of followers is at 552), as well as the number of users each node is following (again, cut off at 50 -- the maximum here is 1566)

[![Training](https://dl.dropbox.com/u/10506/blog/kaggle-fb/training_dist.png)](https://dl.dropbox.com/u/10506/blog/kaggle-fb/training_dist.png)

(Nothing terribly surprising here, but that alone is good to verify. Note that based on these counts alone, and the lack of Bieberish nodes, the unknown social network does seem to consist of more local connections than a network like Twitter. For people tempted to mutter about power laws, I'll hold you off with the bitter coldness of [baby Gauss's tears](http://cscs.umich.edu/~crshalizi/weblog/491.html).)

Similarly, here are the same two graphs, but limited to the nodes in the test set alone:

[![Test](https://dl.dropbox.com/u/10506/blog/kaggle-fb/test_dist.png)](https://dl.dropbox.com/u/10506/blog/kaggle-fb/test_dist.png)

(Notice that there are relatively more test set users with 0 followees than in the full training set, and relatively fewer test set users with 0 followers. This information could be used to better simulate a validation set for model selection, though I didn't end up doing this myself.)

# Preliminary Probes

Finally, let's move on to the models themselves.

In order to quickly get up and running on a couple prediction algorithms, I started with some unsupervised approaches. For example, after building a new validation set to test performance offline, I tried:

* Recommending users who follow you (but you don't follow in return)
* Recommending users similar to you (when representing users as sets of their followers, and using cosine similarity and Jaccard similarity as the similarity metric)
* Recommending users based on a personalized PageRank score
* Recommending users that the people you follow also follow

And so on, combining the votes of these algorithms in a fairly ad-hoc way (e.g., by taking the majority vote or by ordering by the number of followers).

This worked quite well actually, but I'd been planning to move on to a more machine learned model-based approach from the beginning, so I did that next.

*My validation set was formed by deleting random edges from the full training set. A slightly better approach, as mentioned above, might have been to more accurately simulate the distribution of the official test set, but I didn't end up trying this out myself.

# Candidate Selection

In order to run a machine learning algorithm to recommend edges (which would take two nodes, a source and a candidate destination, and generate a score measuring the likelihood that the source would follow the destination), it was necessary to prune the set of candidates to run the algorithm on.

I used two approaches for this filtering step, both based on random walks on the graph.

## Personalized PageRank

The first approach was to calculate a personalized PageRank around each source node.

Briefly, a personalized PageRank is just like standard PageRank, except that when randomly teleporting to a new node, the surfer always teleports back to the given source node being personalized (rather than to a node chosen uniformly at random, as in the standard PageRank algorithm).

That is, the random surfer in the personalized PageRank model works as follows:

* He starts at the source node X that we want to calculate a personalized PageRank around.
* At step $i$: with probability $p$, the surfer moves to a neighboring node chosen uniformly at random; with probability $1-p$, the surfer instead teleports back to the original source node X.
* The limiting probability that the surfer is at node N is then the personalized PageRank score of node N around X.

Here's some Scala code that computes (approximate) personalized PageRank scores and takes the highest-scoring nodes as the candidates to feed into the machine learning model:

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
  
      val pageRankProbs = pageRankHelper(start, probs, NumPagerankIterations)
      pageRankProbs.toList
                   .sortBy { -_._2 }
                   .filter { case (node, score) =>
                      !getFollowings(user).contains(node) && node != user
                    }
                    .take(MaxNodesToKeep)
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
          // And each node distributes its current probability equally to its neighbors.
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
  
## Propagation Score

Another (similar) approach I used, based on [a proposal by another contestant on the Kaggle forums](http://www.kaggle.com/c/FacebookRecruiting/forums/t/2082/0-711-is-the-new-0), works as follows:

* Start at a specified user node and give it some score.
* In the first iteration, this user propagates its score equally to its neighbors.
* In the second iteration, each user duplicates and keeps half of its score S. It then propagates S equally to its neighbors.
* In subsequent iterations, the process is repeated, except that neighbors reached via a backwards link don't duplicate and keep half of their score. (The idea being that we want the score to reach followees and not followers.)

Here's some Scala code to calculate these propagation scores:

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

I played around with tweaking some parameters in both approaches (e.g., weighting followers and followees differently), but the natural defaults (as used in the code above) ended up performing the best.

# Features

After pruning the set of candidate destination nodes to a more feasible level, I fed pairs of (source, destination) nodes into a machine learning model. From each pair, I extracted about 30 features in total.

As mentioned above, one feature that worked quite well on its own was whether the destination node already follows the source.

I also used a wide set of similarity-based features, for example, the Jaccard similarity between the source and destination when both are represented as sets of their followers, when both are represented as sets of their followees, or when one is represented as a set of followers while the other is represented as a set of followees.

    abstract class SimilarityMetric[T] {
      def apply(set1: Set[T], set2: Set[T]): Double;
    }

    object JaccardSimilarity extends SimilarityMetric[Int] {  
      /**
       * Returns the Jaccard similarity between two sets, 0 if both are empty.
       */
      def apply(set1: Set[Int], set2: Set[Int]): Double = {
        val union = (set1.union(set2)).size
    
        if (union == 0) {
          0
        } else {
          (set1 & set2).size.toFloat / union
        }
      }
      
    }

    object CosineSimilarity extends SimilarityMetric[Int] {        
      /**
       * Returns the cosine similarity between two sets, 0 if both are empty.
       */
      def apply(set1: Set[Int], set2: Set[Int]): Double = {
        if (set1.size == 0 && set2.size == 0) {
          0
        } else {
          (set1 & set2).size.toFloat / (math.sqrt(set1.size * set2.size))
        }
      }
      
    }
  
    // ************
    // * FEATURES *
    // ************
  
    /**
     * Returns the similarity between user1 and user2 when both are represented as
     * sets of followers.
     */
    def similarityByFollowers(user1: Int, user2: Int)(implicit similarity: SimilarityMetric[Int]): Double = {
      similarity.apply(getFollowersWithout(user1, user2), getFollowersWithout(user2, user1))
    }
  
    // etc.

Along the same lines, I also computed a similarity score between the destination node and the source node's followees, and several variations thereof.

    /**
     * Iterate over each of user1's followings, compute their similarity with user2
     * when both are represented as sets of followers, and return the sum of these 
     * similarities.
     */
    def followerBasedSimilarityToFollowing(user1: Int, user2: Int)(implicit similarity: SimilarityMetric[Int]): Double = {
      getFollowingsWithout(user1, user2)
                          .map { similarityByFollowers(_, user2)(similarity) }
                          .sum
    }

Other features included the number of followers and followees of each node, the ratio of these, the personalized PageRank and propagation scores themselves, the number of followers in common, and triangle/closure-type features (e.g., whether the source node is friends with a node X who in turn is a friend of the destination node).

If I had had more time, I would probably have tried weighted and more regularized versions of some of these features as well (e.g., downweighting nodes with large numbers of followers when computing cosine similarity scores based on followees, or shrinking the scores of nodes we have little information about), but I didn't get a chance this time around.

# Feature Understanding

But what are these features actually *doing*? As much as I enjoy black-box models, I'm an insights guy, so I'm even more interested in what my models are doing and *why*. So let's use the same app I built before to take a look.

Here's the local network of node 317 (different from the node above), where each node is colored by its personalized PageRank (higher scores are in darker red):

[![317 - Personalized PageRank](https://dl.dropbox.com/u/10506/blog/kaggle-fb/317_propagation.png)](https://dl.dropbox.com/u/10506/blog/kaggle-fb/317_propagation.png)

If we look at the following vs. follower relationships of the central node (recall that purple is friends, teal is followings, orange is followers):

[![317 - Personalized PageRank](https://dl.dropbox.com/u/10506/blog/kaggle-fb/317_following_followers.png)](https://dl.dropbox.com/u/10506/blog/kaggle-fb/317_following_followers.png)

...we can see that, as expected (because I double-weighted edges in my personalized PageRank calculation that represented both following and follower), the darkest red nodes are those that are friends of the central node, while those in a following-only or follower-only relationship have a lower score.

Now how does my propagation score compare to personalized PageRank? Here I colored each node according to the log ratio of its propagation score and personalized PageRank:

[![317 - Log Ratio](https://dl.dropbox.com/u/10506/blog/kaggle-fb/317_log_ratio.png)](https://dl.dropbox.com/u/10506/blog/kaggle-fb/317_log_ratio.png)

Comparing this coloring with the local follow/follower network:

[![317 - Local Network of Node](https://dl.dropbox.com/u/10506/blog/kaggle-fb/317_propagation_local.png)](https://dl.dropbox.com/u/10506/blog/kaggle-fb/317_propagation_local.png)

...we can see that followed nodes (in teal) receive a higher propagation weight than friend nodes (in purple), while follower nodes (in orange) receive almost no propagation score at all.

Going back to node 1, let's look at a different metric. Here each node is colored according to its Jaccard similarity with the source, when nodes are represented by the set of their followers:

[![1 - Similarity by Followers](https://dl.dropbox.com/u/10506/blog/kaggle-fb/1_sim_by_followers.png)](https://dl.dropbox.com/u/10506/blog/kaggle-fb/1_sim_by_followers.png)

We can see that, while the PageRank and propagation metrics tended to favor nodes *close* to the central node, this Jaccard similarity feature helps us explore nodes that are further out.

However, if we look the high-scoring nodes more closely, we see that they often have only a single connection to the rest of the network:

[![1 - Single Connection](https://dl.dropbox.com/u/10506/blog/kaggle-fb/1_single_connection.png)](https://dl.dropbox.com/u/10506/blog/kaggle-fb/1_single_connection.png)

In other words, their high score is due to the fact that they don't have many connections to begin with, so their Jaccard similarity is unexpectedly high. This suggests that some regularization or shrinking is in order.

So here's a regularized version of Jaccard similarity, where we downweight nodes with few connections:

[![1 - Regularized](https://dl.dropbox.com/u/10506/blog/kaggle-fb/1-regularized.png)](https://dl.dropbox.com/u/10506/blog/kaggle-fb/1-regularized.png)

We can see that the outlier nodes are much more muted this time around.

For a starker difference, compare the following two graphs of the Jaccard similarity metric around node 317 (the first graph is an unregularized version, the second is regularized):

[![317 - Unregularized](https://dl.dropbox.com/u/10506/blog/kaggle-fb/317_unregularized.png)](https://dl.dropbox.com/u/10506/blog/kaggle-fb/317_unregularized.png)

[![317 - Regularized](https://dl.dropbox.com/u/10506/blog/kaggle-fb/317_regularized.png)](https://dl.dropbox.com/u/10506/blog/kaggle-fb/317_regularized.png)

Notice, in particular, how the popular node in the top left and the popular nodes at the bottom have a much higher score when we regularize.

And again, there are other networks and features I haven't mentioned here, so play around and discover them on the [app](http://link-prediction.herokuapp.com/) itself.

# Models

For the machine learning algorithms on top of my features, I experimented with two types of models: logistic regression (using both L1 and L2 regularization) and random forests. (If I had more time, I would probably have done some more parameter tuning and maybe tried gradient boosted trees as well.)

So what is a random forest? I wrote an [old (layman's) post](http://www.quora.com/Random-Forests/How-do-random-forests-work-in-laymans-terms/answer/Edwin-Chen-1) on it [here](http://www.quora.com/Random-Forests/How-do-random-forests-work-in-laymans-terms/answer/Edwin-Chen-1), but since we all know nobody ever clicks on these links, let's copy it over:

    Suppose you're very indecisive, so whenever you want to watch a movie, you ask your friend Willow if she thinks you'll like it. In order to answer, Willow first needs to figure out what movies you like, so you give her a bunch of movies and tell her whether you liked each one or not (i.e., you give her a labeled training set). Then, when you ask her if she thinks you'll like movie X or not, she plays a 20 questions-like game with IMDB, asking questions like "Is X a romantic movie?", "Does Johnny Depp star in X?", and so on. She asks more informative questions first (i.e., she maximizes the information gain of each question), and gives you a yes/no answer at the end.

    Thus, Willow is a decision tree for your movie preferences.

    But Willow is only human, so she doesn't always generalize your preferences very well (i.e., she overfits). In order to get more accurate recommendations, you'd like to ask a bunch of your friends, and watch movie X if most of them say they think you'll like it. That is, instead of asking only Willow, you want to ask Woody, Apple, and Cartman as well, and they vote on whether you'll like a movie (i.e., you build an ensemble classifier, aka a forest in this case).

    Now you don't want each of your friends to do the same thing and give you the same answer, so you first give each of them slightly different data. After all, you're not absolutely sure of your preferences yourself -- you told Willow you loved Titanic, but maybe you were just happy that day because it was your birthday, so maybe some of your friends shouldn't use the fact that you liked Titanic in making their recommendations. Or maybe you told her you loved Cinderella, but actually you *really really* loved it, so some of your friends should give Cinderella more weight. So instead of giving your friends the same data you gave Willow, you give them slightly perturbed versions. You don't change your love/hate decisions, you just say you love/hate some movies a little more or less (you give each of your friends a bootstrapped version of your original training data). For example, whereas you told Willow that you liked Black Swan and Harry Potter and disliked Avatar, you tell Woody that you liked Black Swan so much you watched it twice, you disliked Avatar, and don't mention Harry Potter at all.

    By using this ensemble, you hope that while each of your friends gives somewhat idiosyncratic recommendations (Willow thinks you like vampire movies more than you do, Woody thinks you like Pixar movies, and Cartman thinks you just hate everything), the errors get canceled out in the majority. Thus, your friends now form a bagged (bootstrap aggregated) forest of your movie preferences.

    There's still one problem with your data, however. While you loved both Titanic and Inception, it wasn't because you like movies that star Leonardio DiCaprio. Maybe you liked both movies for other reasons. Thus, you don't want your friends to all base their recommendations on whether Leo is in a movie or not. So when each friend asks IMDB a question, only a random subset of the possible questions is allowed (i.e., when you're building a decision tree, at each node you use some randomness in selecting the attribute to split on, say by randomly selecting an attribute or by selecting an attribute from a random subset). This means your friends aren't allowed to ask whether Leonardo DiCaprio is in the movie whenever they want. So whereas previously you injected randomness at the data level, by perturbing your movie preferences slightly, now you're injecting randomness at the model level, by making your friends ask different questions at different times.

    And so your friends now form a random forest.

Moving on, I essentially trained [scikit-learn](http://scikit-learn.org/stable/)'s classifiers on an equal split of true and false edges (sampled from the output of my pruning step, in order to match the distribution I'd get when applying my algorithm to the official test set), and compared performance on the validation set I made, with a small amount of parameter tuning:

    ########################################
    # STEP 1: Read in the training examples.
    ########################################
    truths = [] # A truth is 1 (for a known true edge) or 0 (for a false edge).
    training_examples = [] # Each training example is an array of features.
    for line in open(TRAINING_SET_WITH_FEATURES_FILENAME):
      values = [float(x) for x in line.split(",")]
      truth = values[0]
      training_example_features = values[1:]
  
      truths.append(truth)
      training_examples.append(training_example_features)

    #############################
    # STEP 2: Train a classifier.
    #############################
    rf = RandomForestClassifier(n_estimators = 500, compute_importances = True, oob_score = True)
    rf = rf.fit(training_examples, truths)

So let's look at the variable importance scores as determined by one of my random forest models, which (unsurprisingly) consistently outperformed logistic regression.

[![Random Forest Importance Scores](https://dl.dropbox.com/u/10506/blog/kaggle-fb/rf-importance-scores.png)](https://dl.dropbox.com/u/10506/blog/kaggle-fb/rf-importance-scores.png)

The random forest classifier here is one of my earlier models (using a slightly smaller subset of my full suite of features), where the targeting step consisted of taking the top 25 nodes with the highest propagation scores.

We can see that the most important variables are:

* Personalized PageRank scores. (I put in both normalized and unnormalized versions, where the normalized versions consisted of taking all the candidates for a particular source node, and scaling them so that the maximum personalized PageRank score was 1.)
* Whether the destination node already follows the source.
* How similar the source node is to the people the destination node is following, when each node is represented as a set of followers. (Note that this is more or less measuring how likely the destination is to follow the source, which we already saw is a good predictor of whether the source is likely to follow the destination.) Plus several variations on this theme (e.g., how similar the destination node is to the source node's followers, when each node is represented as a set of followees).

# Model Comparison

Now how do all of these models compare to each other? Is the random forest model universally better than the logistic regression model, or are there some sets of users for which the logistic regression model actually performs better?

To make these kinds of comparisons, I made [a small module](http://link-prediction.herokuapp.com/comparison) that allows you to select two models and visualize their sliced performance.

[![PageRank vs. Is Followed By](https://dl.dropbox.com/u/10506/blog/kaggle-fb/pagerank_vs_is_followed_by_v2.png)](http://link-prediction.herokuapp.com/comparison)

Here, I bucketed all test nodes into buckets based on (the logarithm of) their number of followers, and calculated the mean average precision of an algorithm that recommends nodes to follow using a personalized PageRank, and of an algorithm that recommends nodes that are following the source user but are not yet followed back in return.

We see that except for the case of 0 followers (where obviously the "is followed by" algorithm can do nothing), the personalized PageRank algorithm gets increasingly better in comparison: at first, the two algorithms have roughly equal performance, but as the source node gets more followers, the personalized PageRank algorithm performs increasingly better.

For completeness, here are the individually bucketed mean average precisions for each feature:

[![Individual Bucketed](https://dl.dropbox.com/u/10506/blog/kaggle-fb/raw_performance.png)](https://dl.dropbox.com/u/10506/blog/kaggle-fb/raw_performance.png)

Let's look at one more. Here I compared my "sim to followings by followers" feature (which sums the Jaccard similarity between the candidate node and each of the source node's followings, when nodes are represented by their set of followers) against my "sim by followings" feature (which computes the Jaccard similarity between the candidate and the source, when both are represented by their followings), and we can see that except for the initial case of 0 followees, the performance of the first feature gets monotonically better as the source node follows more and more users.

[![Slice by followees](https://dl.dropbox.com/u/10506/blog/kaggle-fb/slice_by_followees.png)](https://dl.dropbox.com/u/10506/blog/kaggle-fb/slice_by_followees.png)

Admittedly, building a slicer like this is probably overkill for a Kaggle competition where the set of variables to dice by is fairly limited. But imagine instead that this is the real world, where new algorithms are tried out every week and vast amounts of data are collected about every user, and we can slice the performance of a new model by a user's geography, or the user's interests, or by how frequently they log in.

# Mathematicians do it with Matrices

Let's switch directions slightly and think about how we could make our computations more matrix-based. (I didn't do this in the competition -- this is more a preview of another post I'm writing.)

## Personalized PageRank in Scalding

Personalized PageRank, for example, is an obvious fit for a matrix rewrite. Here's how it would look in [Scalding](http://blog.echen.me/2012/02/09/movie-recommendations-and-more-via-mapreduce-and-scalding/)'s new Matrix library:

(For those who don't know, Scalding is a Hadoop framework that Twitter released at the beginning of the year; see [this post on building a big data recommendation engine in Scalding](http://blog.echen.me/2012/02/09/movie-recommendations-and-more-via-mapreduce-and-scalding/) for an introduction.)

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
    
Not only is this matrix formulation a more natural way of expressing the algorithm, but since Scalding (by way of Cascading) supports both local and distributed modes, this code runs just as easily on a Hadoop cluster of thousands of machines (assuming our social network is orders of magnitude larger than the one in the contest) as on a sample of data for a laptop. Big data, big matrix style, BOOM.

## Cosine Similarity as L2-Normalized Multiplication

Here's another example. Calculating cosine similarity between all users is a natural fit for a matrix formulation since, after all, the cosine similarity between two vectors is simply their L2-normalized dot product:

    // A matrix where the cell (i, j) is 1 iff user i is followed by user j.
    val followerMatrix = ...
    
    // A matrix where cell (i, j) holds the cosine similarity between user i and user j,
    // when both are represented as sets of their followers.
    val followerBasedSimilarityMatrix = 
      followerMatrix.rowL2Normalize * followerMatrix.rowL2Normalize.transpose

## A Similarity Extension

But let's go one step further.

To change examples for ease of exposition: suppose you've bought a bunch of books on Amazon, and Amazon wants to recommend a new book you'll like. Since Amazon knows similarities between all pairs of books, one natural way to generate this recommendation is to:

1. Take every book B.
2. Calculate the similarity between B and each book you bought.
3. Sum up all these similarities to get your recommendation score for B.

In other words, the recommendation score for book B on user U is:

DidUserBuy(U, Book 1) * SimilarityBetween(Book B, Book 1) + DidUserBuy(U, Book 2) * SimilarityBetween(Book B, Book2) + ... + DidUserBuy(U, Book n) * SimilarityBetween(Book B, Book n)

Note that this is again a dot product! So it, too, can be rewritten as a matrix multiplication:

    // A matrix where cell (i, j) holds the similarity between books i and j.
    val bookSimilarityMatrix = ...
    
    // A matrix where cell (i, j) is 1 if user i has bought book j, and 0 otherwise.
    val userPurchaseMatrix = ...
    
    // A matrix where cell (i, j) holds the recommendation score of book j to user i.
    val recommendationMatrix = userPurchaseMatrix * bookSimilarityMatrix

Of course, there's a natural analogy between this score and the feature I described a while back above, where I compute a similarity score between a destination node and a source node's followees (when all nodes are represented as sets of followers):

    /**
     * Iterate over each of user1's followings, compute their similarity with user2
     * when both are represented as sets of followers, and return the sum of these 
     * similarities.
     */
    def followerBasedSimilarityToFollowings(user1: Int, user2: Int)(implicit similarity: SimilarityMetric[Int]): Double = {
      getFollowingsWithout(user1, user2)
                          .map { similarityByFollowers(_, user2)(similarity) }
                          .sum
    }

    /**
     * The matrix version of the above function.
     *
     * Why are these the same? Note that the above function simply computes:
     *   DoesUserFollow(User A, User 1) * Similarity(User 1, User B) + DoesUserFollow(User A, User 2) * Similarity(User 2, User B) + ... + DoesUserFollow(User A, User n) * Similarity(User n, User B)
     */
    val followingMatrix = ...
    val followerBasedSimilarityMatrix = 
      followerMatrix.rowL2Normalize * followerMatrix.rowL2Normalize.transpose
  
    val followerBasedSimilarityToFollowingsMatrix = followingMatrix * followerBasedSimilarityMatrix

For people comfortable expressing their computations in a vector manner, being able to write all your computations as matrix manipulations often makes experimenting with different algorithms much more fluid. (For example, imagine you want to switch from L1 normalization to L2 normalization, or that you want to express your objects as binary sets rather than weighted vectors; both of these are simple one-line changes when you have vectors and matrices as first-class objects, but become much more tedious -- *especially in a MapReduce land where this matrix library was designed to be applied!* -- when you don't.)

# End

By now, I think I've spent more time writing this post than on the contest itself, so let's wrap up.

## Tools
I often get questions about what kinds of tools I like to use, so for this competition, my kit consisted of:

* Scala, for code that needed to be fast (e.g., extracting features) or that I was going to run repeatedly (e.g., scoring my validation set).
* Python, for my machine learning models, because [scikit-learn](http://scikit-learn.org/stable/) is awesome.
* Ruby, for quick one-off scripts.
* R, for some data analysis and simple plotting.
* Coffeescript and d3, for the interactive visualizations.

## More

Finally, [here](https://github.com/echen/link-prediction)'s a link to a Github repository containing some of the code for this post, and here are a couple other posts I've written that people interested in this entry might also enjoy:

* [Information Transmission in a Social Network](http://blog.echen.me/2011/09/07/information-transmission-in-a-social-network-dissecting-the-spread-of-a-quora-post/), a case study in how information travels through a social graph.
* [A summary of the algorithms behind the Netflix Prize](http://blog.echen.me/2011/10/24/winning-the-netflix-prize-a-summary/), another crowdsourced recommendation contest for predicting movie ratings.
* [Movie recommendations in Scalding](http://blog.echen.me/2012/02/09/movie-recommendations-and-more-via-mapreduce-and-scalding/), Twitter's Scala-based Hadoop framework built on top of Cascading.

And that's it.