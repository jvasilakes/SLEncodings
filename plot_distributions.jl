import JSON
using ArgParse
using Glob
using Distributions
using StatsPlots
using Colors


function parse_cmdline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--indir"
            help = "Directory containing output files."
            arg_type = String
            required = true
        "--example-id"
            help = "ID of example to plot"
            arg_type = Int64
            required = true
    end
    return parse_args(s)
end


function slbeta(b, d, u)
    W = 2
    a = 0.5
    α = ((W * b)/u) + (W * a)
    β = ((W * d)/u) + ((1-a)W)
    return Beta(α, β)
end


function plot_example(indir, example_id; legend=:top)
    outputs_by_epoch = Dict()
    for fpath in glob("outputs*.json", indir)
        data = JSON.parsefile(fpath)
        epoch = split(split(fpath, '=')[end], '.')[1]
        epoch = parse(Int64, epoch)
        outputs_by_epoch[epoch] = data
    end

    # sort by epoch
    sorted_outputs = sort(collect(outputs_by_epoch), by=x->x[1])
    start = RGB(([239,139,99]./255)...)
    stop = RGB(([110,173,209]./255)...)
    colormap = range(start, stop=stop, length=length(sorted_outputs))
    initialized_plot = false
    plt = plot(size=(1000, 500))
    i = 1
    for (epoch, output) in sorted_outputs
        example = [e for e in values(output)
                   if e["example_id"] == example_id][1]

        if initialized_plot == false
            b = example["y"]["b"]
            d = example["y"]["d"]
            u = example["y"]["u"]
            dist = slbeta(b, d, u)
            plot!(dist, label='y', lw=3, legend=nothing, c=colormap[end], size=(1000,500))
            initialized_plot = true
        end

        b = example["model_output"]["b"]
        d = example["model_output"]["d"]
        u = example["model_output"]["u"]
        dist = slbeta(b, d, u)
        plot!(dist, label=epoch, lwd=2, legend=nothing, c=colormap[i], size=(1000,500))
        i += 1
    end
    display(plt)
end

function main()
    @show parsed_args = parse_cmdline()
    plot_example(parsed_args["indir"], parsed_args["example-id"])
    gui()
    readline()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
