    function safe_divide(x,y)
        return ifelse.(y .== 0, 0, x ./ y)
    end


    function compute_catg_prop(x::AbstractVector)
        out = DataFrame(level = levels(x), prop = zeros(length(levels(x))))
        for i in 1:length(levels(x))
            out.prop[i] = sum(x .== out.level[i])/length(x)
        end
        return out
    end


    function compute_catg_sum(x::AbstractVector)
        out = DataFrame(level = levels(x), sum = zeros(length(levels(x))))
        for i in 1:length(levels(x))
            out.sum[i] = sum(x .== out.level[i])
        end
        return out
    end

    # Cook's distance https://en.wikipedia.org/wiki/Cook%27s_distance
    function cooks_distance(res::Vector, design_matrix::Matrix, dof::Int64, n::Int64)
        mse = res'*res/(n-dof)
        hii = zeros(n)
        for i in 1:n
        hii[i] = design_matrix[i,:]'*inv(design_matrix'*design_matrix)*design_matrix[i,:]
        end
        CooksDistance = res.^2/(dof*mse).*(hii./(1 .- hii).^2)
        return CooksDistance
    end

    # The function generate_LRMoE_data() is adopted from the demo of LRMoE.jl
    # Reference: https://actsci.utstat.utoronto.ca/LRMoE.jl/stable/
    # Original Authors: Spark C Tseung, Samson TC Fung, Andrei Badescu, Sheldon X Lin
    # License: MIT License
    function generate_LRMoE_data(fml, df) # given a formula and a dataframe, extract the needed matrix for LRMoE
            df_fml_schema = StatsModels.apply_schema(fml, StatsModels.schema(fml, df))
            # get y and X
            y, X = StatsModels.modelcols(df_fml_schema, df)
            X = hcat(fill(1, length(y)), X)
            # convert y to a matrix, which is needed for LRMoE
            y = reshape(y, length(y), 1)
            # keep track of the column names
            y_col, X_col = StatsModels.coefnames(df_fml_schema)
            X_col = ["Intercept"; X_col]
            return y, X, y_col, X_col
    end 

    # Use LRMoE_CIp() to get the confidence intervals and pvalues of parameter alpha.
    function LRMoE_CIp(Bootstrap_alpha::AbstractMatrix, col::Integer)
        # Wald method
        CI = round.(quantile(Bootstrap_alpha[:,col], [0.025,0.975]),digits=4)
        temp = cdf(Normal(), abs(LRMoE_model.model_fit.α[1,col]/std(Bootstrap_alpha[:,col])))
        pvalue = round((1 - temp)*2, digits = 4)
        return CI, pvalue
    end
