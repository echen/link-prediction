require 'set'

##############################################################
# Find nodes that we need to find more candidates for.
#
# Recall that our default set of candidates is found by
# running our pagerank/propagation algorithm for 3 iterations,
# essentially discovering all nodes within distance 3.
# This script outputs the nodes in the test set for which
# distance 3 isn't sufficient, so that we can find more
# candidates in some other way (e.g., by expanding the
# distance for these nodes).
##############################################################

counts = Hash.new(0)
# The file we pull candidates from.
File.open("data/propagation_scores_normalized.csv").each_with_index do |line, i|
  src, dest, score, score_normalized = line.strip.split("\t")
  src = src.to_i
  dest = dest.to_i
  
  counts[src] += 1
end

# Find which nodes we need more candidates for.
File.open("data/test.csv").each_with_index do |line, i|
  next if i == 0
  
  src = line.to_i
  count = counts[src]
  puts [src, count].join(",") if count < 10
end