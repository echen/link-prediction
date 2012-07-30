###########################################################
# Gather some statistics on the in-degree and out-degree of
# each node in the official training and test sets.
###########################################################

TRAINING_SET_FILENAME = "data/train.csv"
TEST_SET_FILENAME = "data/test.csv"

num_followers = Hash.new(0)
num_followings = Hash.new(0)
File.open(TRAINING_SET_FILENAME).each_with_index do |line, i|
  next if i == 0
  
  src, dest = line.strip.split(",").map(&:to_i)
  
  num_followings[src] += 1
  num_followers[dest] += 1
end

File.open(TEST_SET_FILENAME).each_with_index do |line, i|
  next if i == 0
  
  node = line.to_i
  puts [node, num_followers[node], num_followings[node]].join(",")
end