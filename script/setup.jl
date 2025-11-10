using DrWatson

import Pkg
@quickactivate "UBItelematics-jl"

# use 'using Pkg; Pkg.add()' to add dependent packages

# locate R for RCall.jl
# ENV["R_HOME"] = "path/to/yout/R"

include(srcdir("UBItelematics.jl"))
