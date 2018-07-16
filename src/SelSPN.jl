using SelectiveSumProductNetworks
using ArgParse
using Plots

function parse_commandline()

    s = ArgParseSettings()

    @add_arg_table s begin
        # "--opt1"
        #     help = "an option with an argument"
        # "--opt2", "-o"
        #     help = "another option with an argument"
        #     a	rg_type = Int
        #     default = 0
        # "--flag1"
        #     help = "an option without argument, i.e. a flag"
        #     action = :store_true
        "dataDir"
            help = "Data folder"
            required = true

        "outDir"
            help = "Data folder"
            required = true

        "srcDir"
            help = "Src folder"
            required = true

        "data"
            help = "dataset"
            required = true

        #"valid.data"
        #    help = "Validational dataset"
        #    required = true

        "iters"
            help = "Number of iterations"
            required = true
            arg_type = Int

        "--gen_pictures"
            help = "Generate the structures found"
            default = false
            arg_type = Bool

        "--verbose"
            help = "Verbose mode"
            default = false
            arg_type = Bool

        "--t_0"
            help = "t_0"
            default = 220
            arg_type = Int

        "--t_n"
            help = "t_n"
            default = 0
            arg_type = Int

        "temp"
            help = "Temperature function"
            required = true

        "--A"
            help = "A in the Dirichlet parameter"
            default = 1.
            arg_type = Float64

        "param_Dir"
            help = "method of the Dirichlet parameter"
            required = true

        "--seed"
            help = "Random seed"
            required = true
            arg_type = Int

        "--cosh_param"
            help = "Random par of the cosh temperature"
            default = 60.
            arg_type = Float64

        "--rand_iter"
            help = "Random seed"
            arg_type = Int
            default = 100

        "--check_iter"
            help = "Random seed"
            arg_type = Int
            default = 10

    end

    return parse_args(s)
end

function main()

    parsed_args = parse_commandline()
    gr()

    outDir = parsed_args["outDir"]
    outputDir = joinpath(outDir, parsed_args["data"], replace(string(now()), ":", "-"))
    @assert !isdir(outputDir)
    mkpath(outputDir)

    info("Using the following output directory: ", outputDir)
    #logging(open(joinpath(outputDir,"logfile.log"), "w"))

    # Logging.configure(filename=joinpath(outputDir,"logfile.log"))

    # read files
    # include(joinpath(parsed_args["srcDir"],"spn2graphviz.jl"))
    # include(joinpath(parsed_args["srcDir"],"SelSPNLib.jl"))

    # read data
    inDirPath = joinpath(parsed_args["dataDir"])

    X = convert(Array{Int}, readcsv(joinpath(inDirPath, string(parsed_args["data"], ".ts.data"))))
    Xval = convert(Array{Int}, readcsv(joinpath(inDirPath, string(parsed_args["data"], ".valid.data"))))
    Xtest = convert(Array{Int}, readcsv(joinpath(inDirPath, string(parsed_args["data"], ".test.data"))))

    # remove zero variance features
    acceptedDims = find(var(X, 1) .> 0)

    X = X[:, acceptedDims]
    Xval = Xval[:, acceptedDims]
    Xtest = Xtest[:, acceptedDims]

    X += 1
    Xval += 1
    Xtest += 1

    # LBD_score(spn_data, X, A, param_Dir)

    # writecsv(joinpath(outDir,"history_bd_train_start.csv"), LBD_score(spn_data, X, A, param_Dir))
    # writecsv(joinpath(outDir,"history_bd_val_start.csv"), LBD_score(spn_data, Xval, A, param_Dir))
    # writecsv(joinpath(outDir,"history_llh_train_start.csv"), mean(llh(spn_data,X)))
    # writecsv(joinpath(outDir,"history_llh_val_start.csv"), mean(llh(spn_data,Xval)))
    # spn2graphviz(spn_data,joinpath(outDir,"optimize_start_data.dot"))

    (N, D) = size(X)
    info("Generating initial structure")
    spn = generateStartSelectiveSPN(D)

    spn2graphviz(spn,joinpath(outDir,"optimize_start.dot"))

    info("Start optimize")
    optimize!(spn, X, Xval, Xtest,
        parsed_args["iters"], parsed_args["temp"],
        parsed_args["t_0"], parsed_args["t_n"],
        parsed_args["A"], Symbol(parsed_args["param_Dir"]),
        parsed_args["seed"], outputDir, parsed_args["rand_iter"],
        parsed_args["check_iter"], parsed_args["cosh_param"],
        parsed_args["gen_pictures"], parsed_args["verbose"])

end

main()
