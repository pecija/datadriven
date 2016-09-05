#1. Installing packages needed.
  #Pkg.add("StatsBase") #Basic statistics and sampling.
  #Pkg.add("DataFrames") #This makes working with tabular data much easier.
  #Pkg.add("DecisionTree") #For decision trees!
  #Pkg.add("Gadfly") #Nice package for plots.
  #Pkg.add("Cairo") #To export plots to files.
  using StatsBase
  using DataFrames
  using DecisionTree
  using Gadfly
  using Cairo

#1. Reading data.
  cd("E:/Julia/Customer Targeting/") #Setting working directory.
  #BankFull = readtable("E:/Julia/Customer Targeting/bank-additional-full.csv", separator=';')
  BankFull = readtable("bank-additional-full.csv", separator=';')
  showcols(BankFull) #This is good to check there are no missing values. Julia is pretty unforgiving for that kind of thing.
  describe(BankFull)
  #The variable duration is not to be included in the analysis.
  delete!(BankFull, :duration)

#2. Creating training and testing datasets.
function InitialiseDatasets(dt::DataFrame)
    Total = size(dt)[1]
    TestProp = 0.3 #Proportion of the dataset reserved for testing.
    TrainSize = round((1-TestProp)*Total,0)
    TestSize = Total - TrainSize
    Total - TrainSize - TestSize #Just checking!

    dt = hcat(dt, [1:Total]) #Adding an Index column.
    rename!(dt, :x1, :Index) #Note that it appears as X1 above but using that results in error. x1 is what works!
    for i=1:Total
      dt[i,:Index]=i
    end

    dt = hcat(dt, [1:Total]) #Adding a column with random numbers that will be used to allocate to the training or testing dataset.
    rename!(dt, :x1, :RndBuzz) #Note that it appears as X1 above but using that results in error. x1 is what works!
    for i=1:Total
      dt[i,:RndBuzz]= rand(1:1000000)
    end

    dt = sort(dt, cols = :RndBuzz)

    train = dt[1:TrainSize,:]
    train = train[1:TrainSize, 1:20] #Dropping the variables Index and RndBuzz.
    test = dt[TrainSize+1:Total, :]
    test = test[1:TestSize, 1:20] #Dropping the variables Index and RndBuzz.
    #proportionmap(dt[:y])
    #proportionmap(train[:y])
    #proportionmap(test[:y])
    return train, test
end

function UnderSample (dtrain::DataFrame, prop::Real)
    undertrain = dtrain[dtrain[:y] .== "yes", :] #This dataset contains only buyers.
    nminitems = size(undertrain)[1]
    chosenones = round(nminitems*(1-prop)/prop,0) #We need only this number of majority class items.

    #Generating a list of majority class items, randomly sorting it and then dropping items until we have only those we need for the desired split.
        temp = dtrain[dtrain[:y] .== "no", :] #This dataset contains only non-buyers.
        nmaxitems = size(temp)[1]
        temp = hcat(temp, [1:nmaxitems]) #Adding an Index column.
        rename!(temp, :x1, :Index) #Note that it appears as X1 above but using that results in error. x1 is what works!
        for i=1:nmaxitems
          temp[i,:Index]=i
        end

        temp = hcat(temp, [1:nmaxitems]) #Adding a column with random numbers that will be used to randomly choose the majority class items that will be removed.
        rename!(temp, :x1, :RndBuzz) #Note that it appears as X1 above but using that results in error. x1 is what works!
        for i=1:nmaxitems
          temp[i,:RndBuzz]= rand(1:1000000)
        end

        temp = sort(temp, cols = :RndBuzz)
        temp = temp[1:chosenones, 1:20] #Dropping columns no longer needed.
        undertrain = append!(undertrain, temp) #Merging minority class with majority class.
        return undertrain
end

train, test = InitialiseDatasets(BankFull);

#3. Let's try to address the class imbalance with undersampling.

  #3.1 Let's define something we will need later on. We want it outside of the loop so that we do not needlessly repeat it.
    minitemprop = linspace(0.3, 0.9, 7) #This creates an array of different proportions for the split between buyers vs. non-buyers.
    NFolds = 5
    purities = linspace(0.6, 1, 5) #This creates an array from 0.5 to 1 with 5 evenly spaced elements.
    tp = zeros(length(minitemprop),length(purities))
    fp = zeros(length(minitemprop),length(purities))
    Success = zeros(length(minitemprop),length(purities))
    TreeDepth = 5

#------------------Test

train, test = InitialiseDatasets(BankFull);
undertrain=UnderSample (train, 0.5)
features = convert(Array, undertrain[1:19])
#features = undertrain[1:19]
#labels = undertrain[20]
labels = convert(Array, undertrain[20])
p = 0.7
nfoldCV_tree_Pedro(labels, features, p, NFolds)
generic_tree = build_tree(labels, features)
generic_tree = prune_tree(generic_tree, p)
data_test = convert(Array, test[1:19])
results_tree = apply_tree (generic_tree, data_test)
ntp = 0
nfp = 0
TestSize = size(data_test)[1]
for i=1:TestSize
  #if test[i,:y]=="yes" && test[i,:first_results]=="yes"
  if test[i,:y]=="yes" && results_tree[i]=="yes"
    ntp=ntp+1
  end
  if test[i,:y]=="no" && results_tree[i]=="yes"
    nfp=nfp+1
  end
end
SuccessRate = ntp/(ntp+nfp)

#------------------Test

    @time for p = 1:length(minitemprop)
      #Building an undersampled decision tree. Remember, Julia does not like dataframes yet so we must convert data into matrices.
      undertrain=UnderSample (train, minitemprop [p])
      features = convert(Array, undertrain[1:19])
      labels = convert(Array, undertrain[20])
      usbigtree = build_tree(labels, features)

      #Using cross validation to find the best purity level for our purposes.
      for i in 1:length(purities)
        Success [p,i] = nfoldCV_tree_Pedro(labels, features, purities[i], NFolds)
      end
    end
    println("Ta da!")

#4. Plotting the madness to see if there is method in it.
for p in 1:length(minitemprop)
  title = string("Success with ", round(minitemprop[p]*100,0) , "% minority class in training set.")
  plt=(plot(x=purities, y=Success[p,:], Geom.point, Geom.line, Guide.xlabel("Purity"), Guide.ylabel("Success"), Guide.title(title)))
  PlotName = string(round(minitemprop[p]*100,0), ".png")
  draw(PNG(PlotName, 4inch, 3inch), plt) #This saves the plot to a file.
end

#5. Testing the best trees we got before. This will show how stupid most of them actually are.
  combos = zeros(5,2)
  SuccessRate = zeros(1,5)
  Attempts = zeros(1,5)

#The best trees (one per undersample level) were:
#100% purity for 30% undersample.
  combos[1,1], combos[1,2] = (1, 0.3)
#90% purity for 40% undersample.
  combos[2,1], combos[2,2] = (0.9, 0.4)
#100% purity for 50% undersample.
  combos[3,1], combos[3,2] = (1, 0.5)
#100% purity for 60% 'undersample'.
  combos[4,1], combos[4,2] = (1, 0.6)
#70% purity for 70% 'undersample'.
  combos[5,1], combos[5,2] = (0.7, 0.7)

n=5

for j = 1:n
  #Undersampling
    train, test = InitialiseDatasets(BankFull);
    data_test = convert(Array, test[1:19]) #Remember, the 20th column is the actual result that we want to validate! That is not an input for the tree.
    TestSize = size(data_test)[1]
    prop = combos[j,2]
    pure = combos[j,1]
    undertrain = UnderSample (train, prop)
  #Building the tree.
    features = convert(Array, undertrain[1:19])
    labels = convert(Array, undertrain[20])
    generic_tree = build_tree(labels, features)
  #Tree pruning.
    generic_tree = prune_tree(generic_tree, pure)
  #Generating results.
    results_tree = apply_tree (generic_tree, data_test)
    ntp = 0
    nfp = 0
    for i=1:TestSize
      #if test[i,:y]=="yes" && test[i,:first_results]=="yes"
      if test[i,:y]=="yes" && results_tree[i]=="yes"
        ntp=ntp+1
      end
      if test[i,:y]=="no" && results_tree[i]=="yes"
        nfp=nfp+1
      end
    end
    SuccessRate[j] = ntp/TestSize
    Attempts[j] = (ntp+nfp)/TestSize
end

title = "Comparing model results for each fitted tree"
plt=(plot(y=SuccessRate, Geom.point, Guide.xlabel("Tree Number"), Guide.ylabel("Success Rate"), Guide.title(title)))
draw(PNG("Success_Rate.png", 4inch, 3inch), plt) #This saves the plot to a file.
plt=(plot(y=Attempts, Geom.point, Guide.xlabel("Tree Number"), Guide.ylabel("Attempts"), Guide.title(title)))
draw(PNG("Attempts.png", 4inch, 3inch), plt) #This saves the plot to a file.

#6 Going wild here and now trying random forest.
  #Undersampling
    train, test = InitialiseDatasets(BankFull);
    data_test = convert(Array, test[1:19]) #Remember, the 20th column is the actual result that we want to validate! That is not an input for the tree.
    TestSize = size(data_test)[1]
    prop = 0.5
    undertrain = UnderSample (train, prop)

  #Building a forest
    features = convert(Array, undertrain[1:19])
    labels = convert(Array, undertrain[20])
    n_feat = 2 #Number of random features.
    n_trees = 10 #Number of trees.
    prop_sample = 0.5 #Proportion of samples per tree.
    tree_depth = 6 #Maximum tree depth.

    Sherwood = build_forest(labels, features, n_feat, n_trees, prop_sample, tree_depth)

  #Applying the forest
    test = convert(Array, test[1:19])
    apply_forest(Sherwood, test)
