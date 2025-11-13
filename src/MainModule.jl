module UBItelematics

import Base: sum, size, length, zeros, 
import Base: ifelse, +, -, *, ==, /, ^
import Base: hcat, vcat, fill, reshape

using DataFrames
import DataFrames: DataFrame

using CategoricalArrays
import CategoricalArrays: levels

using StatsModels
import StatsModels: schema, apply_schema, modelcols, coefnames

using Statistics  
import Statistics: quantile

using Distributions
import Distributions: Normal, Poisson, pdf, cdf

using LRMoE
import fit_LRMoE

using Distributed

using Combinatorics
import Combinatorics: combinations

using Dates
import Dates: Year, Month, Day, value

# using PACKAGE
# import PACKAGE: FUNCTION, FUNCTION, ...

export safe_divide,
       compute_catg_prop,
       compute_catg_sum,
       cooks_distance,

       generate_LRMoE_data,
       bootstrap_LRMoE,
       LRMoE_CIp,
       calculate_posterior_class,

       gini,
       gini2,

       calculate_ncd_coeficients,

       generate_mc_data,
       generate_tripsperday,
       generate_lengthpertrip,
       generate_timestart,
       generate_simulated_data
       
       # MODULE DEFINED FUNCTIONS

include("common_utils")
# include all source files (under src/)



UBItelematics

end # module
