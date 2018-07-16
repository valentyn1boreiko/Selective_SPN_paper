function buildSelectiveSum(dims)

	sumNode = SumNode(nextID(), scope = dims)

	# draw which dim will be used for conditioning
	di = rand(1:length(dims))

	# create product nodes (we assume binary data)
	prodNode1 = ProductNode(nextID(), scope = dims)
	prodNode2 = ProductNode(nextID(), scope = dims)

	# create selective spn under each product node
	sumChild1 = nothing
	sumChild2 = nothing
	if length(dims) > 1
		sumChild1 = buildSelectiveSum(setdiff(dims, dims[di]))
		sumChild2 = buildSelectiveSum(setdiff(dims, dims[di]))
	end

	# indicator leaves
	leaf1 = IndicatorNode(nextID(), 1, dims[di])
	leaf2 = IndicatorNode(nextID(), 2, dims[di])

	# connect nodes
	weights = rand(Dirichlet(2, 1.0))

	add!(sumNode, prodNode1, weights[1])
	add!(sumNode, prodNode2, weights[2])

	if length(dims) > 1
		add!(prodNode1, sumChild1)
	end
	add!(prodNode1, leaf1)

	if length(dims) > 1
		add!(prodNode2, sumChild2)
	end
	add!(prodNode2, leaf2)

	return sumNode
end

function buildSelectiveSum2(dims)

	sumNode = SumNode(nextID(), scope = dims)

	# draw which dim will be used for conditioning
	di = rand(1:length(dims))

	# create product nodes (we assume binary data)
	prodNode1 = ProductNode(nextID(), scope = dims)
    add!(sumNode, prodNode1, 1.0)
	#prodNode2 = ProductNode(nextID(), scope = dims)

	# create selective spn under each product node
	#sumChild1 = nothing
	#sumChild2 = nothing

    for di in dims

        sumNode_ = SumNode(nextID(), scope = [di])
        add!(prodNode1,sumNode_)

    	# indicator leaves
        leaf1 = IndicatorNode(nextID(), 1, dims[di])
        leaf2 = IndicatorNode(nextID(), 2, dims[di])

		weights = rand(Dirichlet(2, 1.0))
        add!(sumNode_,leaf1, weights[1])
        add!(sumNode_,leaf2, weights[2])

    end

    # connect nodes
    #weights = radnd(Dirichlet(2, 1.0))

	# add!(sumNode, prodNode1, weights[1])
	# add!(sumNode, prodNode2, weights[2])

	# if length(dims) > 1
	# 	add!(prodNode1, sumChild1)
	# end
	# add!(prodNode1, leaf1)
    #
	# if length(dims) > 1
	# 	add!(prodNode2, sumChild2)
	# end
	# add!(prodNode2, leaf2)

	return sumNode
end

function buildSelectiveSum3()
	global globalID = 0
    dims = [1,2]
	sumNode = SumNode(nextID(), scope = dims)

    prodNode1 = ProductNode(nextID(), scope = dims)

    sumNode1 = SumNode(nextID(), scope = [2])
    leaf1 = IndicatorNode(nextID(), 1, 2)

    sumNode2 = SumNode(nextID(), scope = [1])
    leaf2 = IndicatorNode(nextID(), 1, 1)
    leaf3 = IndicatorNode(nextID(), 2, 1)


    prodNode2 = ProductNode(nextID(), scope = dims)

    sumNode3 = SumNode(nextID(), scope = [1])
    leaf4 = IndicatorNode(nextID(), 1, 1)
    leaf5 = IndicatorNode(nextID(), 2, 1)

    sumNode4 = SumNode(nextID(), scope = [2])
    leaf6 = IndicatorNode(nextID(), 2, 2)


    weights = rand(Dirichlet(2, 1.0))


    add!(sumNode, prodNode1, weights[1])
    add!(sumNode, prodNode2, weights[2])

    add!(prodNode1, sumNode1)
    add!(sumNode1, leaf1, 1.)
    add!(prodNode1, sumNode2)
    weights = rand(Dirichlet(2, 1.0))
    add!(sumNode2, leaf2,weights[1])
    add!(sumNode2, leaf3,weights[2])


    add!(prodNode2, sumNode3)
    weights = rand(Dirichlet(2, 1.0))
    add!(sumNode3, leaf4, weights[1])
    add!(sumNode3, leaf5, weights[2])
    add!(prodNode2, sumNode4)
    add!(sumNode4, leaf6, 1.)

	return sumNode
end

function generateStartSelectiveSPN(D::Int)
	global globalID = 0
	spn = buildSelectiveSum2(collect(1:D))
	return spn
end

function generateRandomSelectiveSPN(D::Int)
	global globalID = 0
	spn = buildSelectiveSum(collect(1:D))
	return spn
end

function generateRandomSelectiveSPN2()
	global globalID = 0
	spn = buildSelectiveSum3()
	return spn
end

function drawData!(node::SumNode, X::Matrix, obs)

	N = length(obs)
	z = rand(Categorical(node.weights), N)

	for (ci, child) in enumerate(children(node))
		drawData!(child, X, obs[z .== ci])
	end
end

function drawData!(node::ProductNode, X::Matrix, obs)
	for child in children(node)
		drawData!(child, X, obs)
	end
end

function drawData!(node::IndicatorNode, X::Matrix, obs)
	X[obs,node.scope] = node.value
end
