function nextID(maxID = 0)
    global globalID
    if maxID > globalID
        globalID = maxID
    end
	globalID += 1
	return globalID
end

function MI(x,y,base)

    res=0;
    N=length(x) ;
    x_un = unique(x);
    y_un = unique(y);
    x_count = [countnz(x.==xx) for xx in x_un];
    y_count = [countnz(y.==yy) for yy in y_un];
    str_ar = [string(x[i],y[i]) for i in 1:N];
    str_un = unique(str_ar);
    str_count = [countnz(str_ar.==x) for x in str_un];
    for i in str_un n_x = x_count[findin(x_un,parse(Int64,i[1]))];
        n_y = y_count[findin(y_un,parse(Int64,i[2]))];
        count = str_count[findin(str_un,[i])][1];
        #println("Current pair and count",i," ",count);
        #println("x_n and y_n",n_x," ",n_y);
        #println("computation: sum = sum +",count,"*(", (count/N),"*log(",base,",",(count/N),"/(",n_x[1],"*",n_y[1],")/",N^2 );
        res=res+( (count/N) * log(base,(count/N)/((n_x[1]*n_y[1])/(N^2)) ))
    end

    return res #Beispiel von MI
end

function linear(t_0, t_n, i, n, param)
    return t_0 - i * (t_0 - t_n)/n
end

ln(x) = log(e, x)

function logaritmic(t_0, t_n, i, n, param)
    return t_0 - i ^ ( ln(t_0 - t_n) / ln(n) )
end

function hyperbolic_cos(t_0, t_n, i, n, param)
     return (t_0 - t_n) / cosh((param)*(i/n))
end

function optimize!(spn, X, Xval, Xtest, iterations, tempString, t_0 , t_n,
    A, param_Dir, seed, outDir, rand_iter, check_iter, cosh_param,
    gen_pictures, verbose)

    # writecsv(joinpath(outDir,"configuration.csv"), get(history, :train_bd), )
    # csvfile = open(joinpath(outDir,"configuration.csv"),"w")
    dim1_X, dim2_X = size(X)
    dim1_Xval, dim2_Xval = size(Xval)
    dim1_Xtest, dim2_Xtest = size(Xtest)

    # assuming same dimensionality
    @assert dim2_X == dim2_Xval
    @assert dim2_X == dim2_Xtest

    # N should be much large than D
    @assert dim1_X > dim2_X

    # tuple = dim1_X, dim2_X, dim1_Xval, dim2_Xval, dim1_Xtest, dim2_Xtest, iterations, tempString, t_0, t_n, A, param_Dir, seed, outDir, rand_iter, check_iter, cosh_param
    # write(csvfile,"dim1_X, dim2_X, dim1_Xval, dim2_Xval, dim1_Xtest, dim2_Xtest, iterations, tempString, t_0, t_n, A, param_Dir, seed, outDir, rand_iter, check_iter, cosh_param  \n")
    # write(csvfile, join(tuple,","), "\n")
    # close(csvfile)

    temp = (if tempString == "linear"
                linear
            elseif tempString == "logaritmic"
                logaritmic
            elseif tempString == "hyperbolic_cos"
                hyperbolic_cos
            end
        )

    history = MVHistory()

    srand(seed)

    mycount = 1
    for i_rand in 1:rand_iter
        mycount = i_rand
    	rand_move!(spn, i_rand,mycount-1, seed)
        if gen_pictures
            spn2graphviz(spn,joinpath(outDir,"rand.dot"))
        end
    end

    i = 0
	mycount = 1
    mycount2 = 1

	for iteration in 1:iterations
        if (iteration-1)%100 == 0
            open(joinpath(outDir,"configuration.csv"),"w") do f
                write(f,"dim1_X, dim2_X, dim1_Xval, dim2_Xval, dim1_Xtest, dim2_Xtest, iterations, tempString, t_0, t_n, A, param_Dir, seed, outDir, rand_iter, check_iter, cosh_param,current_iter  \n")
                tuple = dim1_X, dim2_X, dim1_Xval, dim2_Xval, dim1_Xtest, dim2_Xtest, iterations, tempString, t_0, t_n, A, param_Dir, seed, outDir, rand_iter, check_iter, cosh_param, iteration
                write(f, join(tuple,","), "\n")
            end
        end

        if verbose
            println("iteration: ", iteration)
        end
        # spn2graphviz(spn, "test$(iteration).dot")
		currentScore = LBD_score(spn, X, A, param_Dir)

        newSPN = copySPN(spn)
        rand_move!(newSPN, iteration, mycount-1,seed)

		newScore = LBD_score(newSPN, X, A, param_Dir)

        ran = rand()
		t = temp(t_0, t_n, iteration, iterations, cosh_param)

        diff = currentScore - newScore
        ex = exp(- (diff) / t)
        if verbose
            println("random pars:", ran ,"exp: ", ex, "temp: ",t, "diff: ",diff," currentscore = ",currentScore," newscore = ",newScore)
        end
        #info(history["val_bd"])
        acceptStructure = (currentScore < newScore) || (ran < ex)
		acceptStructure &= (currentScore != newScore)

        if (currentScore > newScore) && acceptStructure
            info(ran," ",ex)
        end

		if acceptStructure

			push!(history, :train_bd, iteration, newScore/dim1_X)
        	push!(history, :val_bd, iteration, LBD_score(spn, Xval, A, param_Dir)/dim1_Xval)
        	push!(history, :test_bd, iteration, LBD_score(spn, Xtest, A, param_Dir)/dim1_Xtest)

			llhWeights!(spn, X)
			train_llh = mean(llh(spn,X))
			val_llh = mean(llh(spn,Xval))
			test_llh = mean(llh(spn,Xtest))

			push!(history, :train_llh, iteration, train_llh)
        	push!(history, :val_llh, iteration, val_llh)
        	push!(history, :test_llh, iteration, test_llh)

			spn = newSPN

            nodes = order(spn)
            push!(history, :num_nodes, iteration, length(nodes))
            push!(history, :nodes_size, iteration, Base.summarysize(nodes) / 1e6 )
            push!(history, :time_stamp, iteration, now() )


            if verbose
                println("Found new structure")
            end
            mycount += 1
        end

        newSPN = nothing

        if (iteration % check_iter) == 0 && (mycount > mycount2)
            mycount2 += 1
            if verbose
                println("saved")
            end

            columns = [:train_bd :val_bd :test_bd :train_llh :val_llh :test_llh :num_nodes :nodes_size :time_stamp]
            results = mapreduce(s -> get(history, s)[2], hcat, columns)
            (iterations_, _) = get(history, :train_bd)

            results = hcat(iterations_, results)
            writecsv(joinpath(outDir, "history.csv"), vcat(hcat([:iteration], columns), results) )

            # writecsv(joinpath(outDir,"history2.csv"), history2)
            if gen_pictures
                spn2graphviz(spn,joinpath(outDir,"optimize$(iteration).dot"))
            end
        end
	end

	return 0
end


function llhWeights!(spn, Xval; laplace = 1.e-5)

	(N, D) = size(Xval)

    reindex!(spn)
    nodes = order(spn)
    maxId = maximum(node.id for node in nodes)
    _llhval = ones(Float64, size(Xval, 1), maxId) * -Inf

    for node in nodes
		eval!(node, Xval, _llhval)
    end
    counts = sum(_llhval .> -Inf, 1)
    for _node in filter(n -> isa(n, SumNode), nodes)

        nk = [counts[child.id] for child in children(_node)] + laplace
		nnk = ones(length(_node)) ./ length(_node)

		ns = sum(nk)

        _node.weights[:] = (ns != 0) ? (nk / ns) : nnk
    end

end


function reindex!(spn)
    nodes = reverse(order(spn))
    global globalID = 0
    for node in nodes
        node.id = nextID()
    end
end

function getIndependentSets(X)

    #Berechnung von paarweisen MI (obere Dreiecksmatrix)
    (N,D) = size(X)
    MI_M = zeros(D,D)
    _indices = [(i,j) for i in 1:D, j in 1:D if (j>=i)]
    for (i,j) in _indices
            MI_M[i,j] = MI(X[:,i],X[:,j],e)
    end

    #Selektieren der Paare mit minimaler MI
    min_pairs = sort([MI_M[i,j] for (i,j) in _indices])
    _indepPairs = [(i,j) for a in min_pairs for (i,j) in _indices if MI_M[i,j]==a]

    #Ich wähle nur die disjunkte Paare aus
    indepPairs = []
    seen = []
    for i in _indepPairs
        (length(intersect(i,seen))==0) ? (push!(indepPairs,i); append!(seen,[i[1],i[2]])): q=0
    end

    return indepPairs

end

function drawWeights!(Xval, spn)

    reindex!(spn)
    nodes = order(spn)
    counts = Dict{Int,Int}()
    maxId = maximum([x.id for x in order(spn)])
    _llhval = Matrix{Float64}(size(Xval, 1), maxId)
    fill!(_llhval, -Inf)

    for node in nodes
          eval!(node, Xval, _llhval)
    end
    (n,m) = size(_llhval)

    counts = Dict(node.id => sum(_llhval[:, node.id] .> -Inf) for node in nodes)

    for sum in filter(n -> isa(n, SumNode), nodes)
        alphas = [param_Dirichlet(node, A, param_Dir) for child in children(sum)]
        nk = Int[counts[child.id] for child in children(sum)]

        sum.weights[:] = rand(et(nk + alphas))
    end
end


function constructSelectiveSPN(X, var_num::Int; test = 0)

    global globalID = 0
    sets = getIndependentSets(X)
    var_sel = []
    for i in sets[1:convert(Int64,var_num/2)]
        append!(var_sel,collect(i))
    end

    # collect all dimensions
    dims = var_sel

    # unique id
    id = 1

    # construct root node
    spn = SumNode(nextID(), scope = dims)

    # increase id counter
    id += 1

    if test == 0
        # add a single product node that splits up the scope into independent variables
        prod = ProductNode(nextID(), scope = dims)
        add!(spn, prod, 1.0) # weight = 1.0

        # increase id counter
        id += 1
    end
    # loop as long as with still have a dimenions to process
    while !isempty(dims)

        # get dimenions from the stack
        d1 = pop!(dims)
        d2 = pop!(dims)

        # split according to this dimension

        # 1. create a first sum node in the branch
        sum = SumNode(nextID(), scope = [d1,d2])

        # increase id counter
        id += 1


        # 2. create a first product node in the branch
        _prod = ProductNode(nextID(), scope = [d1,d2])

        id += 1

        for d in [d1,d2]

            # 3. create a sum node for one of the variables
            _sum = SumNode(nextID(), scope = to_arr(d))

            # increase id counter
            id += 1

            # 4. create one indicator node for each state and add them to the sum
            val1=0
            val2=1


            add!(_sum, IndicatorNode(nextID(), val1, d)) # uses random weights
            id += 1

            add!(_sum, IndicatorNode(nextID(), val2, d)) # uses random weights
            id += 1

            # add the sum to the product node
            add!(_prod,_sum)

        end

        if test == 0
            add!(sum,_prod)
            add!(prod, sum)
        else
            add!(spn,_prod)
        end

        sum.weights = ones(length(sum)) ./ length(sum)

    end

    # as the weights of the internal sum nodes are random, we have to normalize the spn.
    SumProductNetworks.normalize!(spn)
#    spnplot(spn, "initialSelectiveNetwork")

    return spn
end

#
# function spn_copy(node; idIncrement = 0)
#
#     source = node
#     #println(source.id)
#     nodes = order(source)
#     maxId = maximum(Int[node.id for node in nodes])
#
#     destinationNodes = Vector{SPNNode}()
#     id2index = Dict{Int, Int}()
#     Old_to_new = Dict{Int, Int}()
#
#    for node in nodes
#  #    println("Node id", copy(node.id))
#      Increment = maxId - copy(node.id) + 1
#      maxId += 1
#      if isa(node, NormalDistributionNode)
#
#        id_ = nextID()
#        Old_to_new[copy(node.id)] = id_
#        dnode = NormalDistributionNode(id_, copy(node.scope))
#        dnode.μ = copy(node.μ)
#        dnode.σ = copy(node.σ)
#        push!(destinationNodes, dnode)
#        id2index[id_] = length(destinationNodes)
#      elseif isa(node, MultivariateFeatureNode)
#             id_ = nextID()
#             Old_to_new[copy(node.id)] = id_
#  			dnode = MultivariateFeatureNode(id_, copy(node.scope))
#  			dnode.weights[:] = node.weights
#  			push!(destinationNodes, dnode)
#  			id2index[id_] = length(destinationNodes)
#      elseif isa(node, IndicatorNode)
#        id_ = nextID()
#        Old_to_new[copy(node.id)] = id_
#        dnode = IndicatorNode(id_, copy(node.value), copy(node.scope))
#        push!(destinationNodes, dnode)
#        id2index[id_] = length(destinationNodes)
#      elseif isa(node, SumNode)
#        id_ = nextID()
#        Old_to_new[copy(node.id)] = id_
#        dnode = SumNode(id_, scope = copy(node.scope))
#        cids = Int[child.id for child in children(node)]
#        # println(cids)
#        # println(id2index)
#        for (i, cid) in enumerate(cids)
#          try
#              add!(dnode, destinationNodes[id2index[Old_to_new[cid]]], copy(node.weights[i]))
#          catch
#              info("dest node id = $(cids), transform = $(Old_to_new) all ids $(id2index)")
#          end
#        end
#  			push!(destinationNodes, dnode)
#  			id2index[id_] = length(destinationNodes)
#  		elseif isa(node, ProductNode)
#             id_ = nextID()
#             Old_to_new[copy(node.id)] = id_
#  			dnode = ProductNode(id_, scope = copy(node.scope))
#  			cids = Int[child.id for child in children(node)]
#  			for (i, cid) in enumerate(cids)
#  				add!(dnode, destinationNodes[id2index[Old_to_new[cid]]])
#  			end
#
#  			push!(destinationNodes, dnode)
#  			id2index[id_] = length(destinationNodes)
#
#  		else
#  			throw(TypeError(node, "Node type not supported."))
#  		end
#
#  	end
#
#    # add!(parents(source)[1],destinationNodes[end])
#    return destinationNodes[end]
#
# end
#

# function spnn_split!(node_id, var_id, spn)
#     nodes = order(spn)
#     nnode = [x for x in nodes if x.id == var_id][1]
#     _var = nnode.scope
#     _id = node_id
#
#     for node in nodes
#         if (node.id == _id)
#
#             _nodes = order(node)
#             for _node in _nodes
#                 if isa(_node,SumNode)
#                     indicators = [x for x in children(_node) if (isa(x,IndicatorNode) && x.scope == _var && x.value == 0)]
#                     if (length(indicators)==1)
#
#                         # We assume that for every node there is only one parent
#                         # And the parent of the node is the sum node and we split
#                         # This sum node, which is again the parent of the found _node
#
#                         id1 = _node.id
#                         # println("ID ",_node.id)
#                         _weights = _node.weights
#                         _scope = _node.scope
#
#                         #Split will be only over the sum node
#                         node1 = SumNode(getID(), scope = _scope)
#                         node1.weights = _weights
#                         maxId = maximum(Int[node.id for node in nodes])
#                         id2 = maxId+1
#                         node2 = SumNode(getID(), scope = _scope)
#                         node2.weights = _weights
#
#                         share1 = [x for x in children(_node) if !(isa(x,IndicatorNode) && x.value == 0)]
#                         share2 = [x for x in children(_node) if !(isa(x,IndicatorNode) && x.value == 1)]
#
#                         # println([x.id for x in share1])
#                         # println([x.id for x in share2])
#
#                         for x in share1
#                             add!(node1, x)
#                         end
#                         for x in share2
#                             add!(node2, x)
#                         end
#                         add!(parents(_node)[1],node1)
#                         add!(parents(_node)[1],node2)
#
#
#                         remove!(parents(_node)[1],findfirst(children(parents(_node)[1]) .== _node))
#                     end
#                 end
#             end
#         end
#     end
#
# end
#


function isNodeValid!(node::Leaf)
    return true
end

function isNodeValid!(node::ProductNode)

    childScopes = [isa(child, Leaf) ? [child.scope] : child.scope for child in children(node)]

    # check for decomposability
    isDecomposable = length(reduce(union, childScopes)) == sum(length(x) for x in childScopes)

    if !isDecomposable
        warn("Product node $(node.id) is not decomposable!")
    end

    scope = reduce(union, childScopes)
    node.scope = unique(scope)

    return isDecomposable
end

function isNodeValid!(node::SumNode)

    @assert length(node) > 0
    childScopes = [isa(child, Leaf) ? [child.scope] : child.scope for child in children(node)]

    # check for consistency
    baseScope = sort(childScopes[1])
    isConsistent = all(baseScope == sort(x) for x in childScopes)

    if !isConsistent
        warn("Sum node $(node.id) is not consistent! Child scopes: ")
    end

    node.scope = unique(childScopes[1])
    return isConsistent
end

function isValid!(node::SumNode)

    # get nodes in topological order
    nodes = order(node)

    validSPN = true

    for node in nodes
        validSPN &= isNodeValid!(node)
    end

    return validSPN
end

function dismiss!(node::Leaf, var, states::Vector)
    # Product node with indicators as children??
    if var == node.scope && node.id in states
        remove!(parents(node)[1], findfirst(children(parents(node)[1]) .== node))
    end
end

function dismiss!(node::ProductNode, var, states::Vector)
    for child in children(node)
        dismiss!(child, var, states)
    end

    if length(node) == 0
        remove!(parents(node)[1], findfirst(children(parents(node)[1]) .== node))
    end
end

function dismiss!(node::SumNode, var, states::Vector; verbose = false)

    to_remove = Vector{Int}(0)
    for (i,child) in enumerate(children(node))

        value = unique([x.id for x in order(child) if isa(x,Leaf) && var == x.scope])

        if length(intersect(var, child.scope)) >= 1 && length(intersect(states,value)) == length(value)
            #        println("removed ",child.id)
            push!(to_remove,i)
        end
    end

    sort!(to_remove, rev = true)
    if verbose
        info("sum node id = ",node.id,"remove kids: ",to_remove)
    end
    for i in to_remove
        remove!(node,i)
    end

    try
        for child in children(node)
            dismiss!(child, var, states)
            #info("fine")
        end
    catch
        info("length of the node = ",length(node),"with id = ",node.id, "state id = ", states, "var = ", var)
    end

    if length(node) == 0
        if !isempty(parents(node))
            remove!(parents(node)[1], findfirst(children(parents(node)[1]) .== node))
        end
    end
end

function to_arr(num::Int)
    return [num]
end

function to_arr(ar::Vector)
    return ar
end

function double_parents(spn)
    return any(x -> (length(parents(x)) > 1), order(spn))
end

function reduce_spn!(spn,mycount;verbose = false, makePlot = false)
#TO implement: the retainment of the sum node over the IndicatorNode
    nodes = order(spn)

    # No nodes without children
    _nodes = filter(x -> !isa(x,Leaf), nodes)
    no_children_nodes = filter(x -> length(x) == 0, _nodes)
    if makePlot
        time = string(now(), ":", "-")
        spn2graphviz(spn,"before nodes without children cleaning($(time)): structure $(mycount).dot")
    end
    if verbose
        info("\nnodes without children: ",[x.id for x in no_children_nodes])
        a = filter(y->y.id == 118,_nodes)
        if length(a) > 0
            info("\nchildren of 118",[x.id for x in children(a[1])])
        end
    end
    # @assert length(spn.scope) == 16 "Missing variable 0"
    for node in _nodes
        if length(node) != 0
            to_remove = Vector{Int}(0)
            for (i,child) in enumerate(children(node))
                if  !isa(child,Leaf) && length(child) == 0
     #               print("fff")
                    push!(to_remove,i)
                end
            end
            sort!(to_remove, rev = true)
            for i in to_remove
                remove!(node,i)
            end
        end
    end
    if makePlot
        spn2graphviz(spn,"after nodes without children cleaning($(time)): structure $(mycount).dot")
    end
    no_children_nodes = filter(x -> length(x) == 0, _nodes)
    if verbose
        info("\nnodes without children after the cleaning: ",[x.id for x in no_children_nodes])
    end
    # spn2graphviz(spn2, "before.dot")
    # @assert !double_parents(spn) "1"
    # @assert length(spn.scope) == 16 "Missing variable 1"
    # spn2graphviz(spn2, "after.dot")

    # Short Wire
    one_child_nodes = filter(x -> (length(x) == 1) && !isa(children(x)[1], Leaf), _nodes)
    one_one_child_nodes = filter(x -> length(children(x)[1]) == 1, one_child_nodes)
    if verbose
        info("\nOne child nodes with the struct $(mycount): ",[x.id for x in one_child_nodes])
        info("\nOne one child nodes with the struct $(mycount): ",[x.id for x in one_one_child_nodes])
        info("\nChildren of one one child:",[[y.id for y in children(x)] for x in one_one_child_nodes])

        info("\nChildren of chidlren of one one child:",[[y.id for y in children(children(x)[1])] for x in one_one_child_nodes])

        shortwire = 1
    end
    for (i,node) in enumerate(one_one_child_nodes)
        if makePlot
            spn2graphviz(spn,"before shortwire($(time)): structure $(mycount).dot")
        end
        child_ = children(children(node)[1])[1]
        empty!(child_.parents)
        add!(node, child_)
        remove!(node, 1)
        @assert length(node) == 1
        @assert length(parents(child_)) == 1
        if verbose
            info("Shortwired $(shortwire) time(s)! Node id = $(node.id).dot")
        end
        if makePlot
            spn2graphviz(spn,"structure $(mycount), shortwire $(shortwire) ($(time  )).dot")
        end
        if verbose
            shortwire+=1
        end
    end
    # for node in one_child_nodes
    #     to_remove = Vector{Int}(0)
    #     for (i,child) in enumerate(children(node))
    #         # soll man das lassen? !isa(children(child)[1],Leaf)
    #         if !isa(child,Leaf) && length(child) == 1
    #             push!(to_remove,i)
    #         end
    #     end
    #     sort!(to_remove, rev = true)
    #     for i in to_remove
    #         add!(node,children(children(node)[i])[1])
    #         remove!(children(node)[i],1)
    #         remove!(node,i)
    #     end
    # end
    # @assert !double_parents(spn) "2"
    # @assert length(spn.scope) == 16 "Missing variable 2"
    #
    # nodes_no_leaves = [x for x in order(spn) if !isa(x, Leaf)]
    # nodes_without_children = length([x.id for x in nodes_no_leaves if length(children(x)) == 0]) != 0
    # while nodes_without_children
    #     to_remove = Vector{Int}(0)


    #Collapse Products, Sums !!Only for the Product Nodes
    #spnplot(spn,"before_collapse")
    collapse_nodes = filter(x -> !isa(x,Leaf) && length(children(x)) != 0 && length(filter(child -> typeof(child) == typeof(x) && typeof(child) == ProductNode,children(x))) > 0, nodes)

    for node in collapse_nodes
            to_remove = Vector{Int}(0)
            for (i,child) in enumerate(children(node))
                if typeof(child) == typeof(node) && typeof(child) == ProductNode
                    push!(to_remove,i)
                end
            end

            if length(to_remove) > 0
                sort!(to_remove, rev = true)

                for i in to_remove
                    to_remove1 = Vector{Int}(0)
                    for (j,child) in enumerate(children(children(node)[i]))
                        add!(node,child)
                        # push!(to_remove,j)
                        push!(to_remove1,j)
                        # remove!(children(node)[i],j)
                    end
                    sort!(to_remove1, rev = true)
                    for j in to_remove1
                        # println(to_remove1)
                        # println(length(children(node)[i]))
                        remove!(children(node)[i],j)
                    end

                    remove!(node,i)
                end
            end

    end
    # reindex!(spn)
    # spn2graphviz(spn2, "before.dot")
    # @assert length(spn.scope) == 16 "Missing variable 3"
    # @assert !double_parents(spn) "3"
    # spn2graphviz(spn2, "before.dot")


    # Delete Sum nodes with the only child, that have one parent
    SumNodes = filter(x -> isa(x,SumNode) && length(parents(x)) == 1 && length(x) == 1 && !(isa(children(x)[1],Leaf)), nodes)

    for node in SumNodes
        child_ = children(node)[1]
        parent_ = parents(node)[1]

        empty!(child_.parents)
        add!(parent_, child_)
        remove!(parent_, findfirst(children(parent_) .== node))
        @assert length(parents(child_)) == 1
    end

    #Collapse Products, Sums !!Only for the Product Nodes
    #spnplot(spn,"before_collapse")
    collapse_nodes = filter(x -> !isa(x,Leaf) && length(children(x)) != 0 && length(filter(child -> typeof(child) == typeof(x) && typeof(child) == ProductNode,children(x))) > 0, nodes)

    for node in collapse_nodes
            to_remove = Vector{Int}(0)
            for (i,child) in enumerate(children(node))
                if typeof(child) == typeof(node) && typeof(child) == ProductNode
                    push!(to_remove,i)
                end
            end

            if length(to_remove) > 0
                sort!(to_remove, rev = true)

                for i in to_remove
                    to_remove1 = Vector{Int}(0)
                    for (j,child) in enumerate(children(children(node)[i]))
                        add!(node,child)
                        # push!(to_remove,j)
                        push!(to_remove1,j)
                        # remove!(children(node)[i],j)
                    end
                    sort!(to_remove1, rev = true)
                    for j in to_remove1
                        # println(to_remove1)
                        # println(length(children(node)[i]))
                        remove!(children(node)[i],j)
                    end

                    remove!(node,i)
                end
            end

    end

    #Clean scope
    for node in nodes

        if typeof(node.scope) == Int64
            node.scope = [x.scope for x in order(node) if isa(x,IndicatorNode)][1]
        else
            node.scope = unique([x.scope for x in order(node) if isa(x,IndicatorNode)])
        end
    end

    #spnplot(spn,"after_collapse")
end


#
# function spn_split!(node::ProductNode, var, states, maxId)
#
#     states = to_arr(states)
#     _children = children(node)
#     len = length(_children)
#     #println("C1_ind node id",node.id)
#     #println("var = ",var)
#     #println("states = ", states)
#     pairs = [(i,j) for i in 1:len for j in 1:len if (i>j && (length(intersect(var,children(node)[i].scope)) >= 1 | length(intersect(var,children(node)[j].scope)) >= 1) && (length(intersect(states,[x.value for x in order(children(node)[i]) if isa(x,Leaf)])) >= 2 | length(intersect(states,[x.value for x in order(children(node)[j]) if isa(x,Leaf)])) >= 1))  ]
#     pairs_vec = vcat([i for (i,j) in pairs],[j for (i,j) in pairs])
#     pairs_vec = unique(pairs_vec)
#
#     #println("pairs_vec ", pairs_vec)
#     # C1_ind = [i for i in pairs_vec if length(intersect(states,[x.value for x in order(children(node)[i]) if isa(x,Leaf) && var in x.scope])) >= 1][1]
#     if length(pairs_vec) <= 1
#         warn("Error, node id", node.id)
#         return -1
#     end
#     C1_ind = pairs_vec[1]
#
#     # C2_ind = [x for x in pairs_vec if x != C1_ind][1]
#     C2_ind = pairs_vec[2]
#
#     C1 = _children[C1_ind]
#     C2 = _children[C2_ind]
#     #println("Node ",node.id,"C1 ", C1_ind,"C2 ", C2_ind)
#
#     all_states = [x.value for x in order(C1) if isa(x,Leaf) && var in x.scope]
#     println(all_states)
#
#     if isempty(all_states)
#         return spn_split!(node, C2, C1, var, states, maxId)
#     else
#         return spn_split!(node, C1, C2, var, states, maxId)
#     end
# end
#
function unique_union(arr)
    scope = []
    for x in arr
    	if typeof(x) == Vector{Int64}
    		scope = vcat(scope,x)
    	else
    		push!(scope,x)
    	end
    end
    scope = unique(scope)
    return scope
end

function construct_SPN(node::SumNode,maxID::Int64)
    _children = children(node)
    S = SumNode(nextID(maxID), scope = unique_union(node.scope))
    for child in _children
        add!(S, construct_SPN(child,nextID(maxID)))
    end

    return S
end

function construct_SPN(node::ProductNode,maxID::Int64)
    _children = children(node)
    P = ProductNode(nextID(maxID), scope = unique_union(node.scope))
    for child in _children
        add!(P, construct_SPN(child,nextID(maxID)))
    end
    return P
end

function construct_SPN(node::IndicatorNode,maxID::Int64)
    scope = node.scope
    val = node.value

    return IndicatorNode(nextID(maxID), val, scope)
end

function spn_split!(spn, node::ProductNode, C1, C2, var, randState::Int, mycount; verbose = false, makePlot = false)

    _children = children(node)
    len = length(_children)

    to_remove = Vector{Int}(0)

    C1_ind = findfirst(_children .== C1)
    push!(to_remove,C1_ind)
    C2_ind = findfirst(_children .== C2)
    push!(to_remove,C2_ind)

    if verbose
        info("Children id's: $([x.id for x in children(node)])")
    end
    if makePlot
        spn2graphviz(node,"$(mycount)_try_split_remove.dot")
    end
    sort!(to_remove, rev = true)
    for i in to_remove
        remove!(node,i)
    end

    if makePlot
        spn2graphviz(node,"$(mycount)_did_split_remove.dot")
    end
    # spn2graphviz(spn, "spn_without_children.dot")

    # info("max id = ",maxId," order length = ",length(order(C1)))
    C1_strich = construct_SPN(C1,nextID())
    # maxId += 1
    # info("max id = ",maxId," order length = ",length(order(C2)))
    C2_strich = construct_SPN(C2,nextID())
    # maxId += 1

    randState = rand(Ix_N(C1_strich,var))
    if verbose
        info("value = ", randState)
    end
    randstate = rand([x for x in order(C1_strich) if isa(x,Leaf) && var == x.scope && randState == x.value])

    all_states = [x for x in order(C1_strich) if isa(x,Leaf) && var == x.scope]
    # all_states = filter(x ->  isa(x,Leaf) && var == x.scope, order(C1_strich))
    if verbose
        info("States: ",all_states)
    end
    @assert length(all_states) >= 2

    I_without_state = [x for x in all_states if x.value != randstate.value]
    # I_without_state = filter(x -> x.value != randstate.value, all_states)
    #info("C1_strich kids before split", children(C1_strich))
    # spn2graphviz(C1_strich, "C1_strich.dot")
    if verbose
        info("Ids to delete : ", [x.id for x in I_without_state])
    end

    if makePlot
        spn2graphviz(C1_strich,"$(mycount)_try_split_dismiss.dot")
    end
    dismiss!(C1_strich, var, [x.id for x in I_without_state])

    if makePlot
        spn2graphviz(C1_strich,"$(mycount)_did_split_dismiss.dot")
    end
    #info("C1_strich kids after split", children(C1_strich))
    # spn2graphviz(C1_strich, "C1_strich_no_state.dot")

    # spn2graphviz(C1, "C1.dot")
    if verbose
        info("Ids to delete : ", map(x -> x.id, filter(n -> isa(n,Leaf) && n.scope == var && n.value == randstate.value, order(C1))))
    end

    if makePlot
        spn2graphviz(C1,"$(mycount)_try_split_dismiss_2.dot")
    end

    dismiss!(C1, var,  map(x -> x.id, filter(n -> isa(n,Leaf) && n.scope == var && n.value == randstate.value, order(C1))))

    if makePlot
        spn2graphviz(C1,"$(mycount)_did_split_dimsiss_2.dot")
    end
    # spn2graphviz(C1, "C1_no_states.dot")




    P_strich = ProductNode(nextID(), scope = union(C1.scope,C2.scope))
    if verbose
        info("C1 scope = $(C1.scope), C2 scope = $(C2.scope)")
    end

    C1.parents = []
    C2.parents = []

    if makePlot
        spn2graphviz(P_strich,"$(mycount)_try_split_add.dot")
    end
    add!(P_strich, C1)
    add!(P_strich, C2)

    if makePlot
        spn2graphviz(P_strich,"$(mycount)_did_split_add.dot")
    end
    P_2strich = ProductNode(nextID(), scope = union(C1_strich.scope,C2_strich.scope))
    if verbose
        info("C1_strich scope = $(C1_strich.scope), C2_strich scope = $(C2_strich.scope)")
    end

    if makePlot
        spn2graphviz(P_2strich,"$(mycount)_try_split_add2.dot")
    end
    add!(P_2strich, C1_strich)
    add!(P_2strich, C2_strich)

    if makePlot
        spn2graphviz(P_2strich,"$(mycount)_did_split_add2.dot")
    end
    S = SumNode(nextID(), scope = union(P_strich.scope,P_2strich.scope))

    if makePlot
        spn2graphviz(S,"$(mycount)_try_split_add3.dot")
    end
    add!(S, P_strich,0.5)
    add!(S, P_2strich,0.5)

    if makePlot
        spn2graphviz(S,"$(mycount)_did_split_add3.dot")
    end

    if makePlot
        spn2graphviz(node,"$(mycount)_try_split_add4.dot")
    end
    add!(node,S)
    if makePlot
        spn2graphviz(node,"$(mycount)_did_split_add4.dot")
    end
    if makePlot
        spn2graphviz(spn,"$(mycount)_did_split($(node.id),$(var))_spn.dot")
    end

    # spn2graphviz(spn, "spn_ready.dot")
    return 0
end

function Ix(spn)

    NodeIxs = Dict()

    for node in order(spn)
        if isa(node, Leaf)
            NodeIxs[node.id] = Dict()
            # info("Node id", node.scope)
            # info("Node Scope", node.scope)
            # info("uniques values ", unique([node.value]))
            NodeIxs[node.id][node.scope] = [node.value]
        else
            childIxs = [NodeIxs[child.id] for child in children(node)]
            NodeIxs[node.id] = reduce((x, y) -> merge(vcat, x, y), childIxs)
        end
    end

    return NodeIxs
end

function Ix_N(node,var)
    Ix = [x.value for x in order(node) if isa(x,Leaf) && var in x.scope]
    return unique(Ix)
end

function spn_merge!(mycount, spn, node::SumNode, variable::Int, IxNodes::Dict, maxId; verbose = false , makePlot = false)

    if makePlot
        spn2graphviz(node,"$(mycount)_try_merge_remove.dot")
    end

    C1 = rand(children(node))
    C2 = rand([x for x in children(node) if x.id != C1.id])

    S = filter(n -> collect(keys(IxNodes[n.id])) == [variable], filter(n -> isa(n, SumNode), order(C1)))
    for s in S
        for k in unique(collect(IxNodes[C2.id][variable]))
            #@assert !(k in Ix_N(child1,variable)) "Merge condition was broken at node $(node.id) and child1 $(child1.id) and child2 $(child2.id) and variabe $(variable)"
            # maxId +=1
            add!(s, construct_SPN(filter(n -> isa(n,Leaf) && n.value == k && n.scope == variable,order(C2))[1],nextID()))
            # maxId +=1
        end
    end

    remove!(node,findfirst(children(node) .== C2))

    if makePlot
        spn2graphviz(node,"$(mycount)_did_merge_remove.dot")
    end


    return 0
end


function isSel(spn, verbose = false)

    # relabel all nodes, to ensure all is fine
    reindex!(spn)
    @assert isValid!(spn)
    isSelective = true

    # check for selectivity
    max_var = maximum(spn.scope)
	nodes = order(spn)

	pos_states_tuples = collect(product(Iterators.repeated(0:1,max_var)...))
	len_ = length(pos_states_tuples)
	pos_states_mat = zeros(Float64, len_, max_var)
	for i in 1:len_
		pos_states_mat[i,:] = collect(pos_states_tuples[i])
	end

	_llhval = Matrix{Float64}(len_, length(nodes))
	fill!(_llhval, -Inf)
    if verbose
        info("Nodes length $(length(nodes))")
        info("Nodes ids $([x.id for x in nodes])")
    end
	for node in nodes
        if verbose
            info("node id $(node.id)")
        end
        eval!(node, pos_states_mat, _llhval)
	end

    sum_nodes = [x for x in nodes if isa(x,SumNode)]
    for _node in sum_nodes
        if length(_node) != 0
            ch_len = length(_node)
            children_ = children(_node)
            for i in 1:ch_len
                for j in 1:i-1
                    #supp1 = get_support(_llhval,_children[i].id)
                    supp1 = find(x -> x > -Inf, _llhval[:,children_[i].id])
                    supp2 = find(x -> x > -Inf, _llhval[:,children_[j].id])
                    #supp2 = get_support(_llhval,_children[j].id)
                    isSelective &= length(intersect(supp1,supp2)) == 0
                    @assert isSelective "Selectivity broken on node $(_node.id) and children $(children_[i].id) and $(children_[j].id)"

                end
            end
        end
    end

    return isSelective
end

function children_of_parents(node::SumNode, counts)
    if isempty(parents(node))
        return length(node) * (counts[node.id] > 0) + 1 * (counts[node.id] == 0)
    else
        return (length(node) * (counts[node.id] > 0) + 1 * (counts[node.id] == 0)) * children_of_parents(parents(node)[1],counts)
    end
    #return isempty(parents(node)) ? 1 : (length(node) * (counts[node.id] > 0 && flag == 1) + 1 * (counts[node.id] == 0 || flag == 0)) * children_of_parents(parents(node)[1],counts,1)
end

function children_of_parents(node::SPNNode, counts)
    return children_of_parents(parents(node)[1],counts)
end

function param_Dirichlet(node::SumNode, A::Float64, method::Symbol, counts)

    K = length(node)

    if method == :bde
        R = isempty(parents(node)) ? 1 : children_of_parents(parents(node)[1],counts)
        return ones(K) * (A / (K * R))
    elseif method == :k2
        return ones(K) * (A / K)
    else
        error("Unknown method $method")
    end
end

function LBD_score(spn, Xval, A, param_Dir; laplace = 1.e-5)

    (N, D) = size(Xval)

    nodes = order(spn)
    reindex!(spn)

    maxId = maximum(node.id for node in nodes)
    _llhval = ones(Float64, size(Xval, 1), maxId) * -Inf

    # evaluate network
    for node in nodes
          eval!(node, Xval, _llhval)
    end

    for node in reverse(nodes)

          parents_ = parents(node)

          if length(parents_) > 0
            llh_parent = _llhval[:,parents_[1].id]
            ids_to_deactivate = find(llh_parent .== -Inf)
            _llhval[ids_to_deactivate, node.id] = -Inf
          end
    end

    counts = sum(_llhval .> -Inf, 1)
    LBD_score = zeros(length(nodes))

    for (ni, node) in enumerate(filter(n -> isa(n, SumNode), nodes))

        children_ = children(node)

        # Compute BD score
        alphas = param_Dirichlet(node, A, param_Dir,counts)
        sum1 = sum(alphas)
        sum2 = sum(counts[x.id] for x in children_)
        sum3 = sum(lgamma(alphas[1] + counts[x.id]) - lgamma(alphas[1]) for x in children_)

        LBD_score[ni] = (lgamma(sum1) - lgamma(sum1 + sum2)) + sum3
    end

    return sum(LBD_score)
end



function _children(x,nodes)
    res = []
    for i in x
        append!(res,children(i))
    end
    return res
end

function _get_vars(nodes)
    vars = []
    for i in nodes
        for j in i.scope
            if !(j in vars)
                append!(vars,j)
            end
        end
    end
    return vars
end

function regular_selective_S_internal_(node, IxNodes::Dict )

    for var in node.scope

        l = Int(0)

        for i in 1:length(node)
            states_i = get(IxNodes[children(node)[i].id], var, [])
            for j in 1:i-1
                states_j = get(IxNodes[children(node)[j].id], var, [])

                l += (length(intersect(states_i, states_j)) == 0) &&
                    (length(states_i) > 0) && (length(states_j) > 0)
            end
        end

        # Nehmen wir aus dennen alle, für die es mind. 1 Variable gibt, für die es für jede 2
        # Kinder gilt (in unserem Fall Anzahl von Paaren ist 1, generell aber -
        # Int(factorial(length(children(x))) / (factorial(2) * factorial(length(children(x))-2)))), dass die beide mindestens ein Zustand von dieser Variable haben,
        # Durchschnitt von Indizes von dieser Variable von dieser Kinder ist aber leer.

        t = factorial(length(node)) / (factorial(2) * factorial(length(node)-2))

        if l == t
            return true
        end
    end
    return false
end

function regular_selective_S(spn, IxNodes::Dict)

    ### Wir wollen alle regular selective Knoten auswählen
    #ToDo - überprufen, dass alle andere Variable gleichen Support haben
    # Nehmen wir alle Summenknoten
    nodes = filter(n -> isa(n, SumNode), order(spn))
    nodes = filter(n -> !all(isa.(children(node), Leaf)), nodes)

    nodes = filter(x -> regular_selective_S_internal_(x, IxNodes), nodes)

    return nodes
end

function get_regular_var(spn, node, IxNodes,mycount)
    scope = keys(IxNodes[node.id])
    info("Scope",scope)

    #Danger
    @assert mapreduce( var -> all(mapreduce(child -> var in keys(IxNodes[child.id]), &, children(node))), & ,scope) spn2graphviz(spn,"$(mycount) not all vars udner the sum node.id = $(node.id).dot")
    regular_vars = map(
        var -> isempty(mapreduce(child -> IxNodes[child.id][var], intersect, children(node))), scope)
    @assert sum(regular_vars) == 1 spn2graphviz(spn,"node.id = $(node.id).dot")

    i = findfirst(regular_vars)
    return collect(scope)[i]
end

function rand_move!(spn, counter, mycount, seed ; makePlot = false, verbose = false)

    if verbose
        info("After reindex: $([x.id for x in order(spn)])")
    end

    nodes = order(spn)

    # return state
    state = -1

    # Nehmen zufällig eine der Transformationen auf dem Graph
    stop = 0
    stop_1 = false
    stop_2 = false
    method = rand([1,2])

    IxNodes = Ix(spn)
    maxId = maximum(x.id for x in order(spn))
    global globalID
    globalID = maxId

    if method == 1

        # --
        # Merge move
        # Wir wollen alle regular selective Knoten auswählen
        # --

        mask(n) = isa(n, SumNode) && length(n)>1 && length(n.scope) >= 2
        nodes_ = filter(n -> mask(n), nodes)

        nodes_vars = Dict(node.id => get_regular_var(spn, node, IxNodes,mycount) for node in nodes_)

        nodes_ = filter(n -> !all(isa.(children(n), Leaf)), nodes_)

        if length(nodes_) != 0
            spn2graphviz(spn,"test___")
            nodes_ = filter(n -> nodes_vars[n.id] ∉ map(c -> nodes_vars[c.id], filter(c -> mask(c) , order(n)[1:end-1])) , nodes_)

            @assert length(nodes_) > 0 spn2graphviz(spn,"no nodes satisfying")

            if verbose
                info("Try merge: $(mycount), $([x.id for x in nodes_])")
            end

            node = rand(nodes_)
            scope = keys(IxNodes[node.id])

            regular_vars = map(
                var -> isempty(mapreduce(child -> IxNodes[child.id][var], intersect, children(node))), scope)

            @assert sum(regular_vars) .== 1
            regular_var = collect(scope)[findfirst(regular_vars)]

            if verbose
                info("Variable, merge: $(regular_var), $(node.id)")
            end
            if makePlot
                spn2graphviz(spn, "$(mycount)_try_merge_$(regular_var), $(node.id).dot")
            end

            state = spn_merge!(mycount, spn, node, regular_var, IxNodes, maxId)
            if makePlot
                spn2graphviz(spn, "$(mycount)_done_merge_$(regular_var), $(node.id).dot")
            end

        end
    else

        # --
        # Split move
        #
        # --

        # var = rand([q for q in x.scope for x in nodes if isa(x,ProductNode) && length([y for y in children(x) if length([z for z in order(y) if isa(z,Leaf) && z.scope == q ]) >=2 ]) >=1 ])
        nodes_ = filter(n -> isa(n, ProductNode), nodes)
        nodes_ = filter(n -> length(n) > 1, nodes_)
        nodes_ = filter(n -> mapreduce(child -> any(map(length, unique(values(IxNodes[child.id]))) .>= 2), |, children(n)), nodes_)

        len = length(nodes_)

        if verbose
            info("Try split: $(mycount), $([x.id for x in nodes_])")
        end

        if len > 0

            node = rand(nodes_)

            if verbose
                info("Node id: $(node.id), its scope: $(node.scope)")
            end

            children_ = filter(c -> any(map(length, unique(values(IxNodes[c.id]))) .>= 2), children(node))
            #info("went through")
            C1 = rand(children_)
            #info("went through")
            C2 = rand([x for x in children(node) if x.id != C1.id])
            #info("went through")
            # info(C1.id," ",values(IxNodes[C1.id]))
            # info(map(length, values(IxNodes[C1.id])) .>= 2)
            # info(IxNodes[C1.id])
            var_id = rand(find(map(length, map( arr -> unique(arr), values(IxNodes[C1.id]))) .>= 2))
            # info("went through",var_id," ",keys(IxNodes[C1.id]))
            # info(collect(keys(IxNodes[C1.id]))[var_id])
            var = collect(keys(IxNodes[C1.id]))[var_id]
            # info("went through")
            if verbose
                info("Variable, node, C1, split: $(var), $(node.id), $(C1.id),
                leafs ids = $([z.id for z in order(C1) if isa(z,Leaf) && z.scope == var])")
            end
            # info("went through")
            rand_state = rand(IxNodes[C1.id][var])
            # info("went through")
            if verbose
                info("Variable, node, C1, split: $(var), $(node.id), $(C1.id),
                leafs ids = $([z.id for z in order(C1) if isa(z,Leaf) && z.scope == var ]),
                id of the indicator to split = $(rand_state)")
            end


            if verbose
            #    info("$([x.id for x in order(spn)])")
                for c_ in children(node)
                    info("BEFORE SPLIT: Id of the child =  $(c_.id)")
                end
            end
            if makePlot
                spn2graphviz(spn, "$(mycount)_try_split_$(var), $(node.id).dot")
            end

            # SPLIT
            state = spn_split!(spn, node, C1, C2, var, rand_state, mycount)

            if makePlot
                spn2graphviz(spn, "$(mycount)_did_split_$(var), $(node.id).dot")
            end

            if verbose
                for c_ in children(node)
                    info("AFTER SPLIT: Id of the child =  $(c_.id)")
                end
                #info("$([x.id for x in order(spn)])")
            end

        else
            warn("No nodes, satisfying condition")
            return -1
        end
    end

    if makePlot
        spn2graphviz(spn,"$(mycount)_structure_before_reduce($(now())).dot")
    end

    reduce_spn!(spn, mycount)

    if makePlot
        spn2graphviz(spn,"$(mycount)_structure_after_reduce($(now())).dot")
    end

    reindex!(spn)


    if verbose
        info("After reduce")
    end

    return state
end
