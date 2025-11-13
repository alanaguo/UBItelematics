using DataFrames, GLM, StatsPlots, Colors

function GLMtest(data::DataFrame, formula_list::AbstractArray)    
    mu = mean(data.accidents)
    claim_fit = compute_catg_sum(data.accidents)
    rename!(claim_fit, :level => "Claim_count", :sum => "Observations")

    pglm_t = glm(formula_list[2], data, Poisson(mu), LogLink())
    coef_tt = filter(row -> row.Variable == "log(total_triplength)", significance_table(pglm_t)).Coefficient
    
    pglm_d = glm(formula_list[3], data, Poisson(mu), LogLink())
    coef_td = filter(row -> row.Variable == "log(total_tripdistance)", significance_table(pglm_d)).Coefficient
    
    data.Exposure = coef_tt .* log.(data.total_triplength)
    pglmtt = glm(formula_list[1], data, Poisson(mu), LogLink(); offset = data.Exposure)
    pglmtt_predicted = predict_glm_distribution(pglmtt)

    data.Exposure = coef_td .* log.(data.total_tripdistance)
    pglmtd = glm(formula_list[1], data, Poisson(mu), LogLink(); offset = data.Exposure)
    pglmtd_predicted = predict_glm_distribution(pglmtd)

    coef_table = summarize_GLMs(vcat(significance_table(pglmtt), significance_table(pglmtd)))
    replace!(coef_table.Significance, "***" => "*** (p < 0.001)", "**" => "** (p < 0.01)", "*" => "* (p < 0.05)")
    sort!(coef_table, order(:Coefficient_mean))
    
    plot = @df coef_table scatter(:Coefficient_mean, :Variable, group = :Significance,
        shape = :diamond, palette = [RGB(223/255,122/255,94/255), RGB(130/255,178/255,154/255), RGB(242/255,204/255,142/255)],
        title = "Coefficients of benchmark GLMs", legend = :bottomleft, ms = 8,
        markerstrokecolor = :white, markerstrokewidth = 1)
    vline!([0], linewidth = 2.5, linecolor = :grey, linestyle = :dash, alpha = 2/3, label = false)

    compare_table = compare_GLMs([pglm_t, pglm_d, pglmtt, pglmtd])
    compare_table.model = ["pglm_t", "pglm_d", "pglmtt", "pglmtd"]
    for i in 2:6
        compare_table[:,i] = round.(compare_table[:,i], digits = 2)
    end

    predict_table = hcat(claim_fit[:,1:2], (pglmtt_predicted=pglmtt_predicted, pglmtd_predicted=pglmtd_predicted))

    return plot, coef_table, compare_table, predict_table, coef_tt, coef_td
end


function GLMtest_logistic(data::DataFrame, formula_list::AbstractArray)   
    claim_fit = compute_catg_sum(data.claim)
    rename!(claim_fit, :level => "Claim_status", :sum => "Observations")

    glm_t = glm(formula_list[2], data, Bernoulli(), LogitLink())
    coef_tt = filter(row -> row.Variable == "log(total_triplength)", significance_table(glm_t)).Coefficient
    
    glm_d = glm(formula_list[3], data, Bernoulli(), LogitLink())
    coef_td = filter(row -> row.Variable == "log(total_tripdistance)", significance_table(glm_d)).Coefficient
    
    data.Exposure = coef_tt .* log.(data.total_triplength)
    glmtt = glm(formula_list[1], data, Bernoulli(), LogitLink(); offset = data.Exposure)
    glmtt_predicted = predict_logisticglm_distribution(glmtt)

    data.Exposure = coef_td .* log.(data.total_tripdistance)
    glmtd = glm(formula_list[1], data, Bernoulli(), LogitLink(); offset = data.Exposure)
    glmtd_predicted = predict_logisticglm_distribution(glmtd)

    coef_table = summarize_GLMs(vcat(significance_table(glmtt), significance_table(glmtd)))
    replace!(coef_table.Significance, "***" => "*** (p < 0.001)", "**" => "** (p < 0.01)", "*" => "* (p < 0.05)")
    sort!(coef_table, order(:Coefficient_mean))
    
    plot = @df coef_table scatter(:Coefficient_mean, :Variable, group = :Significance,
        shape = :diamond, palette = [RGB(223/255,122/255,94/255), RGB(130/255,178/255,154/255), RGB(242/255,204/255,142/255)],
        title = "Coefficients of benchmark GLMs", legend = :bottomright, ms = 8,
        markerstrokecolor = :white, markerstrokewidth = 1)
    vline!([0], linewidth = 2.5, linecolor = :grey, linestyle = :dash, alpha = 2/3, label = false)

    compare_table = compare_GLMs([glm_t, glm_d, glmtt, glmtd])
    compare_table.model = ["glm_t", "glm_d", "glmtt", "glmtd"]
    for i in 2:6
        compare_table[:,i] = round.(compare_table[:,i], digits = 2)
    end

    predict_table = hcat(claim_fit[:,1:2], (glmtt_predicted=glmtt_predicted, glmtd_predicted=glmtd_predicted))

    return plot, coef_table, compare_table, predict_table, coef_tt, coef_td
end

# predict for mixture of Poisson model
function predict_HGLM(model::GeneralizedLinearMixedModel, data::DataFrame) # original data
    λ_pred =  MixedModels.predict(model)      
    return round.(sum.(map(n -> map(λ -> pdf.(Poisson(λ), n), λ_pred), [0,1,2])), digits = 2)
end

function pca(X::AbstractMatrix)
    n, m = size(X)
    cov_matrix = (X' * X) / n  
    eig = eigen(cov_matrix)   
    P = eig.vectors
    return vec(P' * X'[1,:])  
end
