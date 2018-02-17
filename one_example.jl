using Distributions, Combinatorics, SumProductNetworks, ValueHistories, Plots, Iterators

gr()

# read files
include("spn2graphviz.jl")
include("SelSPNLib.jl")

# read data
X = convert(Array{Int}, readcsv("data/smallExample/nltcs.ts.data"))
Xval = convert(Array{Int}, readcsv("data/smallExample/nltcs.valid.data"))
N = 16
spn = constructSelectiveSPN(X, N)

srand(123)
for i in 1:200

	rand_move!(spn, i, verbose = true)
	nodeIds = [node.id for node in order(spn)]
	@assert length(unique(nodeIds)) == length(nodeIds)
	spn2graphviz(spn, "$(i).dot")

end
