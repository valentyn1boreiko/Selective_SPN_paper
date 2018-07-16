function spn2graphviz(spn::Node, filename; title = "", printids = true, printtitle = false)

    gstring = ["digraph spn {"]

    if printtitle
        push!(gstring, "labelloc=t;")
        push!(gstring, "label=<<FONT POINT-SIZE=\"80\">$(title)</FONT>>;")
    end
    push!(gstring, "node [margin=0, fontsize=40, width=0.7, shape=circle, fixedsize=true];")

    nodes = SumProductNetworks.order(spn)
    internalNodes = filter(n -> isa(n, Node), nodes)

    for node in nodes
        idstring = ""
        if printids
            idstring = ", xlabel=<<FONT POINT-SIZE=\"20\">$(node.id)</FONT>>"
        end

        if isa(node, SumNode)
            push!(gstring, "$(node.id) [label=\"+\"$(idstring)];")
        elseif isa(node, ProductNode)
            push!(gstring, "$(node.id) [label=<&times;>$(idstring)];")
        elseif isa(node, IndicatorNode)
            push!(gstring, "$(node.id) [shape=doublecircle, label=\"X$(node.scope) = $(node.value)\", fontsize=15$(idstring)];")
        end
    end

    for node in internalNodes
        for child in children(node)
            push!(gstring, "$(node.id) -> $(child.id);")
        end
    end

    push!(gstring, "}")

    open(filename, "w") do f
        write(f, join(gstring, '\n'))
    end
end

