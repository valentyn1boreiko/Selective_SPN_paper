    using Distributions

function nextID()
    global globalID
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
        alphas = [param_Dirichlet(child) for child in children(sum)]
        nk = Int[counts[child.id] for child in children(sum)]

        sum.weights[:] = rand(Dirichlet(nk + alphas))
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
    if var in node.scope && node.id in states
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

function dismiss!(node::SumNode, var, states::Vector)

    to_remove = Vector{Int}(0)
    for (i,child) in enumerate(children(node))

        value = unique([x.id for x in order(child) if isa(x,Leaf) && var == x.scope])

        if length(intersect(var, child.scope)) >= 1 && length(intersect(states,value)) == length(value)
            #        println("removed ",child.id)
            push!(to_remove,i)
        end
    end

    sort!(to_remove, rev = true)
    for i in to_remove
        remove!(node,i)
    end

    try
        for child in children(node)
            dismiss!(child, var, states)
            info("fine")
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

function reduce_spn!(spn)
#TO implement: the retainment of the sum node over the IndicatorNode
    nodes = order(spn)

    # No nodes without children
    _nodes = [x for x in nodes if !isa(x,Leaf)]

    for node in _nodes
        if length(node) != 0
            to_remove = Vector{Int}(0)
            for (i,child) in enumerate(children(node))
                if  !isa(child,Leaf) && length(children(child)) == 0
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


    # Short Wire
    for node in _nodes
        if length(node) != 0
            to_remove = Vector{Int}(0)
            for (i,child) in enumerate(children(node))
                # soll man das lassen? !isa(children(child)[1],Leaf)
                if  !isa(child,Leaf) && length(child) == 1 && !isa(children(child)[1],Leaf)
                    push!(to_remove,i)
                end
            end
            sort!(to_remove, rev = true)
            for i in to_remove
                add!(node,children(children(node)[i])[1])
                remove!(node,i)
            end
        end
    end

    #
    # nodes_no_leaves = [x for x in order(spn) if !isa(x, Leaf)]
    # nodes_without_children = length([x.id for x in nodes_no_leaves if length(children(x)) == 0]) != 0
    # while nodes_without_children
    #     to_remove = Vector{Int}(0)


    #Collapse Products, Sums
    #spnplot(spn,"before_collapse")
    for node in nodes
        if !isa(node,Leaf) && length(children(node)) != 0
            to_remove = Vector{Int}(0)
            for (i,child) in enumerate(children(node))
                if  typeof(child) == typeof(node)
                    push!(to_remove,i)
                end
            end
            sort!(to_remove, rev = true)
            for i in to_remove
                for child in children(children(node)[i])
                    add!(node,child)
                end
                remove!(node,i)
            end
        end
    end


    #Clean scope
    for node in order(spn)

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

function construct_SPN(node::SumNode)
    _children = children(node)
    S = SumNode(nextID(), scope = unique_union(node.scope))
    for child in _children
        add!(S, construct_SPN(child))
    end

    return S
end

function construct_SPN(node::ProductNode)
    _children = children(node)
    P = ProductNode(nextID(), scope = unique_union(node.scope))
    for child in _children
        add!(P, construct_SPN(child))
    end
    return P
end

function construct_SPN(node::IndicatorNode)
    scope = node.scope
    val = node.value

    return IndicatorNode(nextID(), val, scope)
end

function spn_split!(node::ProductNode, C1, C2, var, states::Int, maxId)



    _children = children(node)
    len = length(_children)

    to_remove = Vector{Int}(0)

    C1_ind = findfirst(_children .== C1)
    push!(to_remove,C1_ind)
    C2_ind = findfirst(_children .== C2)
    push!(to_remove,C2_ind)

    info("Children id's: $([x.id for x in children(node)])")

    sort!(to_remove, rev = true)
    for i in to_remove
        remove!(node,i)
    end
    spn2graphviz(spn, "spn_without_children.dot")

    C1_strich = construct_SPN(C1)
    C2_strich = construct_SPN(C2)

    randState = rand(Ix_N(C1_strich,var))
    randstate = rand([x for x in order(C1_strich) if isa(x,Leaf) && var == x.scope && randState == x.value])

    all_states = [x.id for x in order(C1_strich) if isa(x,Leaf) && var == x.scope]
    info("States: ",all_states)

    @assert length(all_states) >= 2

    I_without_state = [x for x in all_states if x != randstate.id]

    #info("C1_strich kids before split", children(C1_strich))
    dismiss!(C1_strich, var, I_without_state)
    #info("C1_strich kids after split", children(C1_strich))
    spn2graphviz(C1_strich, "C1_strich_no_state.dot")

    dismiss!(C1, var, [x.id for x in order(C1) if isa(x,Leaf) && x.value == randstate.value])
    spn2graphviz(C1, "C1_no_states.dot")



    P_strich = ProductNode(nextID(), scope = union(C1.scope,C2.scope))
    info("C1 scope = $(C1.scope), C2 scope = $(C2.scope)")

    C1.parents = []
    C2.parents = []
    add!(P_strich, C1)
    add!(P_strich, C2)
    P_2strich = ProductNode(nextID(), scope = union(C1_strich.scope,C2_strich.scope))
    info("C1_strich scope = $(C1_strich.scope), C2_strich scope = $(C2_strich.scope)")
    add!(P_2strich, C1_strich)
    add!(P_2strich, C2_strich)

    S = SumNode(nextID(), scope = union(P_strich.scope,P_2strich.scope))
    add!(S, P_strich)
    add!(S, P_2strich)
    spn2graphviz(S, "S_ready.dot")
    add!(node,S)


    spn2graphviz(spn, "spn_ready.dot")
    return 0
end

function Ix_N(node,var)
    Ix = [x.value for x in order(node) if isa(x,Leaf) && var in x.scope]
    return unique(Ix)
end

function spn_merge!(spn, node::SumNode, variable)

    _children = children(node)
    len = length(_children)
    # len_ar = 1:len
    # child1_n = rand(len_ar)
    # child1 = _children[child1_n]
    # child2_n = rand(len_ar[1:end .!= child1_n])
    # child2 = _children[child2_n]
    # pairs_vec = []
    # len = length(_children)
    #
    #pairs = [(i,j) for i in 1:len for j in 1:len if (i>j && length(intersect(Ix_N(children(node)[i],variable),Ix_N(children(node)[j],variable))) == 0) ]
    IxN = [Ix_N(_children[i], variable) for i in 1:len]
    pairs = [(i,j) for i in 1:len for j in 1:i-1 if isempty(intersect(IxN[i], IxN[j]))]
    #
    # #println(pairs)
    # pairs_vec = vcat([i for (i,j) in pairs],[j for (i,j) in pairs])
    # pairs_vec = unique(pairs_vec)
    # pairs_vec = filter(x -> length(Ix_N(_children[x],variable)) >=1, pairs_vec)
    #
    # info("pairs_vec: $(pairs_vec)")
    #sum_nodes = [x for x in order(_children[pairs_vec[1]]) if isa(x,SumNode)]
    #  println("sum nodes : ",[x.id for x in sum_nodes])
    pair = rand(pairs)
    child1 = _children[pair[1]]
    child2 = _children[pair[2]]
    child1_n = findfirst(children(parents(child1)[1]) .== child1)
    child2_n = findfirst(children(parents(child2)[1]) .== child2)
    sum_nodes = [x for x in order(child1) if isa(x,SumNode)]

    for _node in sum_nodes
        info("going through sum node: id = $(_node.id), scope = $(_node.scope), length = $(length(_node.scope)), var = $(variable), var in scope = $(variable in _node.scope)")
        if length(_node.scope) == 1 && variable in _node.scope
            info("selected node: $(_node.id)")
    #         println("node scope 1, var in scope :", _node.id)
            for k in Ix_N(child2,variable)
                #@assert !(k in Ix_N(child1,variable)) "Merge condition was broken at node $(node.id) and child1 $(child1.id) and child2 $(child2.id) and variabe $(variable)"
                add!(_node, [x for x in order(child2) if isa(x,Leaf) && x.value == k && x.scope == variable][1])
            end
        end
    end

    remove!(node,child2_n)

    reduce_spn!(spn)
    return 0
end


# function isNodeSel(node::Leaf)
#     return true
# end
#
# function isNodeSel(node::ProductNode)
#     childrenSelective = true
#
#     for child in children(node)
#         childrenSelective &= isSel(child)
#     end
#
#     return childrenSelective
# end

# function get_support(_llhval,node_id)
# 	#Only for binary variables now
# 		return find(x -> x > -Inf, _llhval[:,node_id])
# end

# function isNodeSel(node::SumNode)
#
#     @assert length(node) > 0
#
#     ch_len = length(node)
#     _children = children(node)
#
#     if length(node) == 1
#         return isNodeSel(children(node)[1])
#     end
#     isSelective = true
#
#     for i in 1:ch_len
#         for j in 1:i-1
#             supp1 = get_support(spn,_children[i].id)
#             supp2 = get_support(spn,_children[j].id)
#             isSelective &= length(intersect(supp1,supp2)) == 0
#         end
#     end
#     # conditionsOnVariable = trues(length(node.scope))
#     # for (si,d) in enumerate(node.scope)
#     #
#     #     childStates = Vector{Vector}(length(children(node)))
#     #
#     #     for (ci, child) in enumerate(children(node))
#     #
#     #         nodes = filter(n -> isa(n, Leaf), order(child))
#     #         nodes = filter(n -> n.scope == d, nodes)
#     #
#     #         childStates[ci] = union([n.value for n in nodes])
#     #     end
#     #
#     #     # TODO: Change such that this code actually checks selectivity and not regular selectivity :D
#     #     baseChildState = sort(childStates[1])
#     #     if !all(baseChildState == sort(x) for x in childStates)
#     #         # check if all states are disjoint
#     #         for i in 2:length(childStates)
#     #             for j in 1:i-1
#     #                 p1 = length(reduce(union, childStates[i], childStates[j]))
#     #                 p2 = length(childStates[i]) + length(childStates[j])
#     #     #            println("Elemetns ", si, "Length = ", length(conditionsOnVariable))
#     #                 conditionsOnVariable[si] &= p1 == p2
#     #
#     #             end
#     #         end
#     #         @assert conditionsOnVariable[si] "Selectivity is brocken at node $(node.id) and variable $(d)"
#     #     else
#     #         conditionsOnVariable[si] = false
#     #     end
#     # end
#     #
#     # isSelective = all(isNodeSel(child) for child in children(node))
#     # isSelective &= sum(conditionsOnVariable) >= 1
#
#     return isSelective
# end


function isSel(spn)

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
    info("Nodes length $(length(nodes))")
    info("Nodes ids $([x.id for x in nodes])")
	for node in nodes
        info("node id $(node.id)")
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

function param_Dirichlet(node)
    return 1
end

function LBD_score(spn, Xval)

    (N, D) = size(Xval)

    #reindex!(spn)
    nodes = order(spn)
    maxId = maximum([node.id for node in nodes])
    _llhval = Matrix{Float64}(size(Xval, 1), maxId)
    fill!(_llhval, -Inf)

    for node in nodes
          eval!(node, Xval, _llhval)
    end

    counts = sum(_llhval .> -Inf, 1)
    #reverse!(counts)

    _nodes = filter(n -> isa(n, SumNode), nodes)

    LBD_score = zeros(length(_nodes))

    for (ni, node) in enumerate(_nodes)

        _children = children(node)
        sum1 = sum([param_Dirichlet(x) for x in _children])
        sum2 = sum([counts[x.id] for x in _children])
        sum3 = sum([(lgamma(param_Dirichlet(x) + counts[x.id]) - lgamma(param_Dirichlet(x))) for x in _children])

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

function regular_selective_S(spn)

    ### Wir wollen alle regular selective Knoten auswählen
    #ToDo - überprufen, dass alle andere Variable gleichen Support haben
    # Nehmen wir alle Summenknoten
    nodes = filter(n -> isa(n, SumNode), order(spn))

    # Nehmen wir aus dennen alle, für die es mind. 1 Variable gibt, für die es für jede 2 Kinder gilt (in underem Fall Anzahl von Paaren ist 1, generell aber - Int(factorial(length(children(x))) / (factorial(2) * factorial(length(children(x))-2)))), dass die beide mindestens ein Zustand von dieser Variable haben, Durchschnitt von Indizes von dieser Variable von dieser Kinder ist aber leer.
    nodes = filter(x -> length(filter(var -> length( [(i,j) for i in 1:length(children(x)) for j in 1:i-1 if (length(Ix_N(children(x)[i],var)) >= 1 && length(Ix_N(children(x)[j],var)) >= 1 && length(intersect(Ix_N(children(x)[i],var),Ix_N(children(x)[j],var))) == 0 )]) == 1, x.scope)) == length(x.scope), nodes)

    nodes = filter(x -> length([ch for ch in children(x) if isa(ch,Leaf)]) != length(children(x)), nodes)

    return nodes
end


function rand_move!(spn, counter; makePlot = false, verbose = false)

    if verbose
        info("After reindex: $([x.id for x in order(spn)])")
    end


    nodes = order(spn)

    # return state
    state = -1

    # Nehmen zufällig eine der Transformationen auf dem Graph
    method = rand(1:2)
    if method == 1
        # Merge
        ### Wir wollen alle regular selective Knoten auswählen
        nodes = regular_selective_S(spn)
        len = length(nodes)
        if verbose
            info("Try merge: $(counter), $(len)")
        end
        # for n_ in nodes
        #     info("Id of node =  $(n_.id)")
        #     for c_ in children(n_)
        #         info("Id of the child =  $(c_.id)")
        #     end
        # end
        if len > 0
            node = rand(nodes)

            #vars_ = [var for var in node.scope if length( [(i,j) for i in 1:length(children(node)) for j in 1:length(children(node)) if (i>j && length(intersect(Ix_N(children(node)[i],var),Ix_N(children(node)[j],var))) > 0)]) == 0 ]

            #vars = [var for var in vars_ if length([e for e in children(node) if length(Ix_N(e,var)) >=1]) >= 2]

            vars = [var for var in node.scope if length( [(i,j) for i in 1:length(children(node)) for j in 1:length(children(node)) if (i>j && length(Ix_N(children(node)[i],var)) >= 1 && length(Ix_N(children(node)[j],var)) >= 1 && length(intersect(Ix_N(children(node)[i],var),Ix_N(children(node)[j],var))) == 0 )]) == 1 ]

            try
                if verbose
                    info("all vars: ",vars)
                end
                var = rand(vars)
            catch
                for i in node.scope
                    info("states of the var $(i) in the children id $(children(node)[1].id) = $(Ix_N(children(node)[1],i))")
                end
                for i in node.scope
                    info("states of the var $(i) in the children id $(children(node)[2].id) = $(Ix_N(children(node)[2],i))")
                end
                warn("$([x.id for x in order(spn)])")
                warn("Node id = $(node.id), length children = $(length(children(node))) vars_ = $(vars_) vars = $(vars), node scope = $(node.scope)")
            end

            if verbose
                info("Variable, merge: $(var), $(node.id)")
            end
            if makePlot
                spn2graphviz(spn, "$(counter)_opt_internal_merge_$(var), $(node.id).dot")
            end
            #sum_nodes = [x for x in nodes if isa(x,SumNode) && var in x.scope]
            #node = rand(sum_nodes)#

            state = spn_merge!(spn, node, var)

            # nodes_no_leaves = [x for x in order(spn) if !isa(x, Leaf)]
            # println("merge! ",[x.id for x in nodes_no_leaves if length(children(x)) == 0])
        else
            warn("No nodes, satisfying condition")
            if makePlot
                spn2graphviz(spn, "$(counter)_opt_internal_merge_trial.dot")
            end
            return -1
        end
    else

        # var = rand([q for q in x.scope for x in nodes if isa(x,ProductNode) && length([y for y in children(x) if length([z for z in order(y) if isa(z,Leaf) && z.scope == q ]) >=2 ]) >=1 ])
        nodes = filter(n -> isa(n, ProductNode), order(spn))

        nodes = filter(x -> length(filter(var -> length([y for y in children(x) if length(unique([z.value for z in order(y) if isa(z,Leaf) && z.scope == var ])) >= 2 ]) >=1 , x.scope )) >=1 , nodes)

        len = length(nodes)

        if verbose
            info("Try split: $(counter), $([x.id for x in nodes])")
        end

        if len > 0

            node = rand(nodes)
            if verbose
                info("Node id: $(node.id), its scope: $(node.scope)")
            end
            vars = [var for var in node.scope if length( [ch for ch in children(node) if length(unique([z.value for z in order(ch) if isa(z,Leaf) && z.scope == var ])) >=2 ]) >=1 ]
            var = rand(vars)

            C1 = [ch for ch in children(node) if length(unique([z.value for z in order(ch) if isa(z,Leaf) && z.scope == var])) >= 2 ][1]
            C2 = [ch for ch in children(node) if  ch.id != C1.id][1]

            if verbose
                info("Variable, node, C1, split: $(var), $(node.id), $(C1.id), leafs ids = $([z.id for z in order(C1) if isa(z,Leaf) && z.scope == var ])")
            end
            if makePlot
                spn2graphviz(spn, "$(counter)_opt_internal_split_$(var), $(node.id), $(C1.id).dot")
            end
            # node = rand([x for x in nodes if isa(x,ProductNode) && var in x.scope])
            randState = rand(Ix_N(C1,var))
            randstate = rand([x.id for x in order(C1) if isa(x,Leaf) && var == x.scope && randState == x.value])
            if verbose
                info("Variable, node, C1, split: $(var), $(node.id), $(C1.id), leafs ids = $([z.id for z in order(C1) if isa(z,Leaf) && z.scope == var ]), id of the indicator to split = $(randstate)")
            end
            # info("Random Split Move on node: $(node.id), var: $(var), state: $(randstate)")
            maxId = maximum([x.id for x in order(spn)])
            if verbose
                info("$([x.id for x in order(spn)])")
                for c_ in children(node)
                    info("BEFORE SPLIT: Id of the child =  $(c_.id)")
                end
            end

            state = spn_split!(node, C1, C2, var, randstate, maxId)

            if verbose
                for c_ in children(node)
                    info("AFTER SPLIT: Id of the child =  $(c_.id)")
                end
                info("$([x.id for x in order(spn)])")
            end

        else
            warn("No nodes, satisfying condition")
            if makePlot
                spn2graphviz(spn, "$(counter)_opt_internal_split_trial.dot")
            end
            return -1
        end
    end

    reduce_spn!(spn)
    reindex!(spn)

    if makePlot
        spn2graphviz(spn, "$(counter)_opt_internal_spn_done.dot")
    end

    if verbose
        info("After reduce: $([x.id for x in order(spn)])")
    end

    return state

end
