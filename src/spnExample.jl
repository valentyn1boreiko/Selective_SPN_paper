using SumProductNetworks

# include plotting function from external script
include("plotSPN.jl")

# reading in the small dataset
X = convert(Array{Int}, readcsv("/home/valentyn/github/SPN/bd-spns/data/smallExample/nltcs.ts.data"))

# get size of the data set
(N, D) = size(X)

println(" Found data set with ", N, " samples and ", D, " dimensions.")

# Define function to construct a selective SPN where all variables are assumed independent.
# For now, we assume that the data is binary, i.e. x ∈ {0,1}
function constructSelectiveSPN(D::Int)

    # collect all dimensions
    dims = collect(1:D)

    # unique id
    id = 1

    # construct root node
    spn = SumNode(id, scope = collect(1:D))

    # increase id counter
    id += 1

    # add a single product node that splits up the scope into independent variables
    prod = ProductNode(id, scope = collect(1:D))
    add!(spn, prod, 1.0) # weight = 1.0

    # increase id counter
    id += 1

    # loop as long as with still have a dimenions to process
    while !isempty(dims)

        # get a dimenions from the stack
        d = pop!(dims)

        # split according to this dimension

        # 1. create a sum node over all states of this variable, i.e. 0 and 1
        sum = SumNode(id, scope = [d])

        # increase id counter
        id += 1

        # 2. create one indicator node for each state and add them to the sum
        add!(sum, IndicatorNode(id, 0, d)) # uses random weights
        id += 1

        add!(sum, IndicatorNode(id, 1, d)) # uses random weights
        id += 1

        # add the sum to the product node
        add!(prod, sum)

    end

    # as the weights of the internal sum nodes are random, we have to normalize the spn.
    SumProductNetworks.normalize!(spn)

    return spn
end

# call the function
spn = constructSelectiveSPN(D)

# now we can visualize the spn
spnplot(spn, "initialSelectiveNetwork")

# we can evaluate the model on the validation data
Xval = convert(Array{Int}, readcsv("../data/smallExample/nltcs.valid.data"))

println("LLH train:", mean(llh(spn, X)))
println("LLH validation:", mean(llh(spn, Xval)))
