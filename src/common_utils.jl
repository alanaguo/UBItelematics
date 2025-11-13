    function safe_divide(x::Real, y::Real)
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
    function generate_LRMoE_data(fml::FormulaTerm, df::DataFrame) # given a formula and a dataframe, extract the needed matrix for LRMoE
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

    function bootstrap_LRMoE(
        data::DataFrame,
        fml::FormulaTerm,
        α_init::Matrix,
        model_init,
        N::Int,
        bootstrap_size::Int
    )
        Bootstrap_alpha = zeros(N*2, size(α_init,2))
        
        @distributed for i in 1:N
            Bootstrap_data = data[rand(1:size(data, 1), bootstrap_size), :]
            y_bt, X_bt, y_col_bt, X_col_bt = generate_LRMoE_data(fml, Bootstrap_data)
            LRMoE_model_bootstrap = fit_LRMoE(
                y_bt, X_bt, α_init, model_init.ll_best;
                exact_Y=true, ϵ=0.01, ecm_iter_max=1000, print_steps=100
            )
            Bootstrap_alpha[i, :] = LRMoE_model_bootstrap.model_fit.α[1, :]
            Bootstrap_alpha[N+i, :] = LRMoE_model_bootstrap.model_fit.α[2, :]
            println("$ith bootstrap")
        end
        
        return Bootstrap_alpha
    end

    function LRMoE_CIp(Bootstrap_alpha_comp::AbstractMatrix, LRMoE_model::LRMoE.LRMoESTDFit, col::Integer) 
        CI = round.(quantile(Bootstrap_alpha_comp[:,col], [0.025,0.975]),digits=4)
        # Wald method
        temp = cdf(Normal(), abs(LRMoE_model.model_fit.α[1,col]/std(Bootstrap_alpha_comp[:,col])))
        pvalue = round((1 - temp)*2, digits = 4)
        return CI, pvalue
    end

    function calculate_posterior_class(y::Matrix, X::Matrix, alpha::Matrix, comp_dist::Matrix)
        prob_component = predict_class_prior(X, alpha).prob
        n_comp = size(comp_dist)[2]
        prob_component_post = copy(prob_component)
        denominator = ((prob_component[:,1].*pdf.(comp_dist[1,1], y)) .+ (prob_component[:,2].*pdf.(comp_dist[1,2], y)) .+ (prob_component[:,3].*pdf.(comp_dist[1,3], y)))
            for j in 1:n_comp
            prob_component_post[:,j] = (prob_component[:,j].*pdf.(comp_dist[1,j], y))./denominator
            end
        return prob_component_post
    end

        function gini(y::Vector)
            combs = combinations(y, 2)  
            # gmd
            gmd = sum(abs(x[1] - x[2]) for x in combs)  
            gmd = gmd * 2 / (length(y)^2-length(y))
            # gini
            gini = gmd/mean(y)/2
            return gini
        end

        function gini2(y::Vector)
            combs = combinations(y, 2)  
            # gmd
            gmd = sum(abs(x[1] - x[2]) for x in combs)  
            gmd = gmd * 2 / (length(y)^2-length(y))
            # gini
            gini = gmd/median(y)/2
            return gini
        end



