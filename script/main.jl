using DrWatson
@quickactivate "UBItelematics-jl"

using CSV, DataFrames, JLD2, CategoricalArrays, Statistics, Dates, Random, StableRNGs: StableRNG
using Distributions, GLM, StatsBase, StatsModels, HypothesisTests, LogExpFunctions, ExpectationMaximization, LRMoE
using Plots, StatsPlots, ColorSchemes
using RCall, Resample

using .UBItelematics


data # original data

# model 1: Poisson model(R)
                           
    @rput data

    R"""
    poisson_model <- glm(accidents ~ age + I(age^2) + car_response + log(total_triplength) 
        + avr_harshacceleration + avr_harshbraking + avr_intensiveturning + avr_speed + max_speed 
        + topup_amount + notripday_prop + peakprop + fatigueprop, 
        data = data, family = poisson(link = "log"))
    summary(poisson_model)
    """
    
    R"""
    coef(poisson_model)
    """
    
    R"""
    c(AIC(poisson_model),BIC(poisson_model))
    """
    
    R"""
    poisson_model_predicted = fitted(poisson_model)
    """
    
    @rget poisson_model_predicted
    
    R"""
    confint(poisson_model)
    """


# model 2: ZIP model


    @rput data

    R"""
    zip_model <- pscl::zeroinfl(accidents ~ 
    age + I(age^2) + car_response
    # driving style
    + avr_harshbraking + avr_intensiveturning + avr_harshacceleration
    + avr_speed + max_speed
    # driving habits
    + topup_amount + notripday_prop + peakprop + fatigueprop 
    | 0 + log(total_triplength),  
    data = data, family = poisson(link = 'log'))

    summary(zip_model)
    """


    R"""
    c(AIC(zip_model),BIC(zip_model))
    """

    R"""
    zip_model_predicted = fitted(zip_model)
    """

    @rget zip_model_predicted

    R"""
    zip_lambda = predict(zip_model, type = "count")
    zip_p0 = predict(zip_model, type = "zero")
    """
    @rget zip_lambda
    @rget zip_p0

    zip_lambda.*(1 .-p0) == zip_model_predicted # true

    R"""
    confint(zip_model)
    """



# LRMoE model
   
    formula_LRMoE = @formula(accidents ~ 
        log(total_triplength) +
        # driver's profile
        age + age^2 + car_response
        # driving style
        + avr_harshbraking + avr_intensiveturning + avr_harshacceleration
        + avr_speed + max_speed
        # driving habits
        + topup_amount + notripday_prop + peakprop + fatigueprop 
        )

    y, X, y_col, X_col = generate_LRMoE_data(formula_LRMoE, data)
    
    n_comp = 3
    model_init = cmm_init(y, X, n_comp, ["discrete"]; exact_Y = true, n_random = 1)

    α_init = model_init.α_init  

    LRMoE_model = fit_LRMoE(y, X,  α_init, model_init.ll_best;
    exact_Y=true, ϵ = 0.001, ecm_iter_max=2000, print_steps=100)
    summary(LRMoE_model)

    LRMoE_model_predicted = predict_mean_prior(X, LRMoE_model.model_fit.α, LRMoE_model.model_fit.comp_dist;)


# Dynamic pricing model

    ## pricing basis

    simulated_data = load(datadir("simulated_data.jld2"), "simulated_data") 

    simulated_data_3m = filter(row->row.DayID.<=91,simulated_data)
    simulated_data_re = filter(row->row.DayID.>91,simulated_data)

    formula_nm = @formula(accidents ~ 
    log(total_triplength) +
    # driver's profile
    age + age^2
    # driving style
    + avr_hbrk + avr_turn
    # driving habits
    + notripday_prop + peakprop
    )

    y_nm, X_nm, y_col_nm, X_col_nm = generate_LRMoE_data(formula_nm, simulated_data_3m)

    n_comp = 3
    model_init = cmm_init(y_nm, X_nm, n_comp, ["discrete"]; exact_Y = true)
    model_init.ll_best[1,:] 

    LRMoE_nm = fit_LRMoE(y_nm, X_nm, model_init.α_init, 
    [GammaCountExpert(1,13);;GammaCountExpert(0.001,0.5);;GammaCountExpert(0.2, 8)], #model_init.ll_best;
    exact_Y=true, ϵ = 0.05, ecm_iter_max=2000, print_steps=10)
    summary(LRMoE_nm) 

    ## model results

    # parameters α
    hcat(X_col_nm, round.(LRMoE_nm.model_fit.α'[:,:], digits =4))

    # δ
    pdf.(LRMoE_nm.model_fit.comp_dist, 0) 
    # μ (annually)
    mean.(LRMoE_nm.model_fit.comp_dist).*4
    # σ
    sqrt.(var.(LRMoE_nm.model_fit.comp_dist))
    # π
    mean(predict_class_prior(X_nm, LRMoE_nm.model_fit.α).prob, dims = 1)


    ## pure premium calculation
    LRMoE_pers_nm = predict_mean_prior(X_nm, alpha, comp_dist)
    j=1
    nm_pers_m3=zeros(size(data,1))
    for i in 1:size(data,1)
        index = collect(j:1:j+sum(simulated_data_3m.DriverID .== i)-1)    
        nm_pers_m3[i] = mean(LRMoE_pers_nm[index,:]) # first 3m premium base is calculated by true using days
        j = j+sum(simulated_data_3m.DriverID .== i)
        println("$i")
    end

    ## 1st step, compute pure risk premium person by person => nm_premium_matrix
    nm_premium_matrix = fill(0.0, (size(data,1),maximum(data.length_of_record)))     
    @distributed for i in 1:size(data,1) 
        tempdf = simulated_data_re[findall(simulated_data_re.DriverID.==i),:]
        y_temp = zeros(size(tempdf,1),1)
        y_temp[:,1] = tempdf.accidents                 
        X_temp = generate_LRMoE_data(formula_nm,tempdf)[2]
        replace!(X_temp, -Inf => log(0.000001))
        prob_component_post = posterior_class(y_temp, X_temp, alpha, comp_dist)
        mean_posterior = sum(prob_component_post[:,:]'.*mean.(comp_dist)[1,:],dims = 1)[1,:]
        nm_premium_matrix[i,Integer.(tempdf.DayID)] = mean_posterior
        println("driver: $i")
    end

    # 2nd step: compute daily-scaled pure risk premium person by person => scaled_nm_premium_matrix
    scaled_nm_premium_matrix= zeros(size(data,1),maximum(data.length_of_record))
    @distributed for n in 1:size(data,1)
    temp = vcat(repeat([nm_pers_m3[n]/91],91), nm_premium_matrix[n,92:maximum(data.length_of_record)]./length(findall(simulated_data_3m.DriverID.==n)))
    scaled_nm_premium_matrix[n,:] = temp
    println("$n")
    end

    # jldsave(datadir("scaled_nm_premium_matrix.jld2"); DataFrame(scaled_nm_premium_matrix, :auto))


# Poisson model as pricing basis and current commercial auto-insurance NCD system as premium updating system
    # Default: current NCD system dynamics + Poisson GLM with only traditional covariates and policyperiod as offset (no update)

        NCDglm = glm(@formula(accidents ~ PolicyPeriod + age + age^2 + gender + car_brand + license_seniority), 
            data, Poisson(), LogLink())

    # pure risk premium calculation
        NCDpremiumbase = GLM.predict(NCDglm, data)
        NCDcoefficients = calculate_NCDcoefficients(data)

        NCDpremiums = NCDpremiumbase .* NCDcoefficients

    # time measure
    time_observe_NCD = hcat(fill(1,size(data,1)), 
    ifelse.(data.length_of_record./365 .>2, 1, data.length_of_record.%365/365), 
    ifelse.(data.length_of_record./365 .>2, data.length_of_record.%365/365, 0))

    year_observe_NCD = ifelse.(time_observe_NCD.>0, 1, 0)


