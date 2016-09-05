type Pedro_ConfusionMatrix
    classes::Vector
    matrix::Matrix{Int}
    accuracy::Float64
    tp::Int32 #True positives
    fp::Int32 #False positives
    tn::Int32 #True negatives
    fn::Int32 #False negatives
    kappa::Float64
end

function Pedro_confusion_matrix(actual::Vector, predicted::Vector)
    @assert length(actual) == length(predicted)
    N = length(actual)
    _actual = zeros(Int,N)
    _predicted = zeros(Int,N)
    classes = sort(unique([actual; predicted]))
    N = length(classes)
    for i in 1:N
        _actual[actual .== classes[i]] = i
        _predicted[predicted .== classes[i]] = i
    end
    CM = zeros(Int,N,N)
    for i in zip(_actual, _predicted)
        CM[i[1],i[2]] += 1
    end
    accuracy = trace(CM) / sum(CM)
    prob_chance = (sum(CM,1) * sum(CM,2))[1] / sum(CM)^2
    kappa = (accuracy - prob_chance) / (1.0 - prob_chance)
    tp = CM[2,2]
    tn = CM[1,1]
    fp = CM[2,1]
    fn = CM[1,2]
    return Pedro_ConfusionMatrix(classes, CM, accuracy, tp, tn, fp, fn, kappa)
end

function _nfoldCV_Pedro(classifier::Symbol, labels, features, args...)
    nfolds = args[end]
    if nfolds < 2
        return nothing
    end
    if classifier == :tree
        pruning_purity = args[1]
    elseif classifier == :forest
        nsubfeatures = args[1]
        ntrees = args[2]
        partialsampling = args[3]
    elseif classifier == :stumps
        niterations = args[1]
    end
    N = length(labels)
    ntest = _int(floor(N / nfolds))
    inds = randperm(N)
    accuracy = zeros(nfolds)
    tp = zeros(nfolds)
    tn = zeros(nfolds)
    fp = zeros(nfolds)
    fn = zeros(nfolds)
    for i in 1:nfolds
        test_inds = falses(N)
        test_inds[(i - 1) * ntest + 1 : i * ntest] = true
        train_inds = !test_inds
        test_features = features[inds[test_inds],:]
        test_labels = labels[inds[test_inds]]
        train_features = features[inds[train_inds],:]
        train_labels = labels[inds[train_inds]]
        if classifier == :tree
            model = build_tree(train_labels, train_features, 0)
            if pruning_purity < 1.0
                model = prune_tree(model, pruning_purity)
            end
            predictions = apply_tree(model, test_features)
        elseif classifier == :forest
            model = build_forest(train_labels, train_features, nsubfeatures, ntrees, partialsampling)
            predictions = apply_forest(model, test_features)
        elseif classifier == :stumps
            model, coeffs = build_adaboost_stumps(train_labels, train_features, niterations)
            predictions = apply_adaboost_stumps(model, coeffs, test_features)
        end
        cm = Pedro_confusion_matrix(test_labels, predictions)
        #accuracy[i] = cm.accuracy
        tp[i] = cm.tp
        fp[i] = cm.fp
        tn[i] = cm.tn
        fn[i] = cm.fn
        #println("\nFold ", i)
        #println(cm)
    end
    #println("\nMean Accuracy: ", mean(accuracy))
    #println("\nMean TP: ", mean(tp))
    #println("\nMean TN: ", mean(tn))
    #println("\nMean FN: ", mean(fn))
    #println("\nMean FP: ", mean(fp))
    #Results = [mean(tp), mean(fp), mean(tn), mean(fn)]
    Results = mean(tp) / (mean(tp)+mean(fp))
    return Results
end

nfoldCV_tree_Pedro(labels::Vector, features::Matrix, pruning_purity::Real, nfolds::Integer) = _nfoldCV_Pedro(:tree, labels, features, pruning_purity, nfolds)
