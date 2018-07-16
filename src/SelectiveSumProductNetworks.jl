module SelectiveSumProductNetworks

        using SumProductNetworks
        using Distributions
        using Combinatorics
        using Iterators
        using Combinatorics
        using Plots
        using ValueHistories

        include("spn2graphviz.jl")
        include("SelSPNLib.jl")
        include("generateData.jl")

        export optimize!
        export generateRandomSelectiveSPN
        export drawData!
        export spn2graphviz
        export generateStartSelectiveSPN
        export LBD_score
        export llh
        export nextID
        export param_Dirichlet
        export buildSelectiveSum3
        export children_of_parents

end
