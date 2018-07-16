using Colors, Compose, GraphPlot, LightGraphs, SumProductNetworks, Cairo



function nodetype2string(node::SPNNode)
    if isa(node, SumNode)
        return "S"
    elseif isa(node, ProductNode)
        return "P"
    elseif isa(node, NormalDistributionNode)
        return "N"
    else
        return "L"
    end
end

function spn2lightgraph(spn::SumNode)

    nodes = SumProductNetworks.order(spn)
    numVertex = length(nodes)

    G = DiGraph(numVertex)

    labels = Vector{String}(numVertex)
    types = Vector{}(numVertex)

    rootDepth = SumProductNetworks.depth(spn)

    for node in nodes
        if isa(node,Leaf)
            labels[node.id] = string(node.id,"_",node.scope,"_V = ",node.value)
        else
            labels[node.id] = string(node.id,"_",node.scope)
        end
        a=string()
        b=string(Int(floor(31/2)))
        c=string(30*2)
        types[node.id] = parse(Colorant,string((typeof(node)==IndicatorNode)?"green":((typeof(node)==SumNode)?"blue":((typeof(node)==ProductNode)?"red":(""))) ))
        println(types[node.id])
    end

    for node in filter(x -> !isa(x, Leaf), nodes)
        for (ci, child) in enumerate(children(node))
            add_edge!(G, node.id, child.id)
        end
    end

    return (G, labels, types)
end
function spnplot(spn::SumNode, filename; size = 26cm, title = "")

    reindex!(spn)
    (G, labels, types) = spn2lightgraph(spn)

    # colrs = linspace(colorant"blue2", colorant"aliceblue", maximum(types) + 1)
    nodefillc = types

    draw(PDF("$(filename).pdf", size, size),
         compose(gplot(G, nodelabel=labels, nodefillc = nodefillc, arrowlengthfrac=0.01),
                 Compose.text(0.0, -1.1, title))
        )
end

function main()

    info("Create test SPN")
    # create test SPN
    SPN = SumNode(1)
    SumProductNetworks.add!(SPN, ProductNode(2))
    SumProductNetworks.add!(SPN, ProductNode(3))

    lastId = 4
    for child in children(SPN)
        SumProductNetworks.add!(child, NormalDistributionNode(lastId, 1))
        SumProductNetworks.add!(child, NormalDistributionNode(lastId+1, 2))

        lastId += 2
    end

    info("Plotting SPN to testspn.pdf")
    spnplot(SPN, "testspn")
end

# uncomment to run the test script
# main()
