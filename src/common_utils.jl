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

    function calculate_ncd_coefficients(data::DataFrame)
        # Check for required columns to prevent runtime errors
        required_cols = [:accidents, :timeofaccidents, :install_date, :secondaccident]
        @assert all(col in names(data) for col in required_cols) "Missing required columns: $(setdiff(required_cols, names(data)))"
    
        n_rows = nrow(data)
        # Initialize result array with dimensions [row count, 4 columns]
        # Column meanings: [Initial value, Year 1, Year 2, Year 3]
        ncd_coefficients = zeros(n_rows, 4)  
        
        # Precompute time differences to avoid repeated calculations in loop
        time_diffs = data.timeofaccidents .- data.install_date
        second_accident_diffs = data.secondaccident .- data.install_date
        
        # Define time thresholds as constants for clarity and reusability
        day0 = Day(0)
        day365 = Day(365)
        day730 = Day(730)
    
        for i in 1:n_rows
            accident_count = data.accidents[i]
            time_diff = time_diffs[i]
            
            if accident_count == 0
                # No accidents: apply standard discount progression
                ncd_coefficients[i, :] = [1.0, 0.8, 0.7, 0.6]
                
            elseif accident_count == 1
                # Single accident: determine discount based on when it occurred
                if time_diff < day0
                    # Accident occurred before installation date (year -1)
                    ncd_coefficients[i, :] = [1.0, 0.8, 0.7, 0.6]
                elseif time_diff < day365
                    # Accident occurred in year 0 (0-365 days after installation)
                    ncd_coefficients[i, :] = [1.0, 1.0, 0.8, 0.7]
                else  # Inferred: time_diff < day730 from original logic
                    # Accident occurred in year 1 (365-730 days after installation)
                    ncd_coefficients[i, :] = [1.0, 0.8, 1.0, 0.8]
                end
                
            elseif accident_count == 2
                # Two accidents: check timing of both incidents
                second_time_diff = second_accident_diffs[i]
                
                if time_diff < day0 && second_time_diff < day0
                    # Both accidents occurred before installation date
                    ncd_coefficients[i, :] = [1.2, 0.8, 0.7, 0.6]
                elseif time_diff < day0 && second_time_diff > day0
                    # One accident before installation, one after
                    ncd_coefficients[i, :] = [1.0, 1.0, 0.8, 0.7]
                else
                    # Both accidents in year 0 (per original comment: 43/45 cases)
                    ncd_coefficients[i, :] = [1.0, 1.2, 0.8, 0.7]
                end
            else
                # Handle unaccounted accident counts (>2)
                @warn "Row $i has $accident_count accidents - no defined logic, using default values"
                ncd_coefficients[i, :] = [1.0, 1.0, 1.0, 1.0]
            end
        end
    
        return ncd_coefficients
    end


    function generate_mc_data(transitionmatrix::Matrix, length::Integer)
        
        initial_state = 1 # set to be 0-10
        num_states = size(transitionmatrix, 1)
        current_state = initial_state   
    
        generated_data = Vector()
        push!(generated_data, initial_state)
    
            for t in 1:length-1
                next_state = sample(1:num_states, weights(transitionmatrix[current_state, :]))
                push!(generated_data, next_state)
                current_state = next_state
            end
        
    
        return generated_data
    end
    
    
    
    function generate_tripsperday(avr_numtrip::Real, tripdays::Integer, notripdays::Integer)
        avr_numtrip_dist = truncated(
            Normal(avr_numtrip, 0.377059*avr_numtrip),lower=1
            )
            
        tripsperday = vcat((round.(rand(avr_numtrip_dist,tripdays))),
                       zeros(notripdays)
                        )
    tripsperday = Integer.(tripsperday)
    shuffle!(tripsperday)
    return tripsperday
    end
    
    function generate_lengthpertrip(avr_triplength::Real, num_trip::Integer, fatiguedriving::Integer)
    avr_triplength_dist = truncated(Exponential(avr_triplength.*60), lower = 1, upper = 240)
    
            lengthpertrip = vcat(rand(avr_triplength_dist,
                                num_trip-fatiguedriving),
                         repeat([240], fatiguedriving)) # 240~360: setting for fatigued trip
            shuffle!(lengthpertrip)
    return lengthpertrip
    end
    
    function generate_timestart(peakprop::Real, morningpeak_prop::Real, eveningpeak_prop::Real, tripsperday::Vector)
    Timestart_dist = MixtureModel([
              truncated(Normal(13.5627, 4.6431), lower = 0, upper = 6.9999), # latenight
              Uniform(7,9.9999), # morningpeak
              truncated(Normal(13.5627, 4.6431), lower = 10, upper = 16.9999), # daytime
              Uniform(17,19.9999), # eveningpeak
              truncated(Normal(13.5627, 4.6431), lower = 20, upper = 23.9999)], # night
             [0.0964*(1 - peakprop), # latenight
              morningpeak_prop*0.5, # morningpeak_prop
              0.7729*(1 - peakprop), # daytime
              eveningpeak_prop*0.5, # eveningpeak
              0.1307*(1 - peakprop) # night
              ]
            )
    
            Timestart = Integer.(floor.(rand(Timestart_dist, sum(tripsperday)))) 
    return Timestart
    end
    
    
    
    function generate_simulated_data(start::Integer,stop::Integer, smdata::DataFrame, transitionmatrix::Matrix)
        df = DataFrame()
    
         for n in start:stop
    
            # dists 
            # avr_numtrip #####################################################################################
            avr_numtrip_dist = truncated(Normal(smdata.avr_numtrip[n], 0.377059*smdata.avr_numtrip[n]),lower=1)
            
            tripsperday = vcat((round.(rand(avr_numtrip_dist,smdata.tripdays[n]))),
                               zeros(smdata.notripdays[n])
                                )
            tripsperday = Integer.(tripsperday)
            shuffle!(tripsperday)
    
            # avr_triplength (minutes) ###########################################################################
            avr_triplength_dist = truncated(Exponential(smdata.avr_triplength[n].*60), lower = 1, upper = 239)
    
            lengthpertrip = vcat(rand(avr_triplength_dist,
                                    sum(tripsperday)-smdata.fatiguedriving[n]),
                         repeat([240], smdata.fatiguedriving[n])) # 240~360: setting for fatigued trip
            shuffle!(lengthpertrip)
    
            # timestamp (timestart) ###########################################################################
            # Alter variable: peakprop = peakdays/tripdays => peakprop = peaktrips/numtrips
            Timestart_dist = MixtureModel([
              truncated(Normal(13.5627, 4.6431), lower = 0, upper = 6.9999), # latenight
              Uniform(7,9.9999), # morningpeak
              truncated(Normal(13.5627, 4.6431), lower = 10, upper = 16.9999), # daytime
              Uniform(17,19.9999), # eveningpeak
              truncated(Normal(13.5627, 4.6431), lower = 20, upper = 23.9999)], # night
             [0.0964*(1 - smdata.peakprop[n]), # latenight
              smdata.morningpeak_prop[n]*0.5, # morningpeak_prop
              0.7729*(1 - smdata.peakprop[n]), # daytime
              smdata.eveningpeak_prop[n]*0.5, # eveningpeak
              0.1307*(1 - smdata.peakprop[n]) # night
              ]
            )
    
            Timestart = Integer.(floor.(rand(Timestart_dist, sum(tripsperday)))) 
    
        # Start generation ####################################################################################
        
            # clear IDs
            id = 1 # total trip id of a person
            dayid = 1
            tripid = 1 # tripid in a day, used for loop
    
            # clear variables
            num_obs_trip = 0
    
            for dayid in 1:smdata.length_of_record[n] # generate per day
                if tripsperday[dayid] > 0 # if it is a trip day:
                    for tripid in 1:tripsperday[dayid] # generate by trip
                        
                        num_obs_trip = length(collect(1:1:Integer(round(lengthpertrip[id] .* 6)))) # observation every 10 seconds
    
                        Starttime = DateTime("$(Date("0000-01-01")+Day(dayid-1))T$(Timestart[id]):$(rand(0:59)):$(rand(1:59))")
                        Timestamp = collect(Starttime:Second(10):Starttime + Second(10*(num_obs_trip-1))) # 10 seconds per observation
    
                        DriverID = fill(n, num_obs_trip)
    
                        DayID = fill(dayid, num_obs_trip)
    
                        TripID = fill(tripid, num_obs_trip)
    
                        TripCode = fill("$n-$dayid-$tripid", num_obs_trip)
    
                             speed_ratio = smdata.avr_speed[n]/3.31331
    
                        Speed = generate_mc_data(transitionmatrix, num_obs_trip).*speed_ratio 
                        Speed = Speed .- rand(Normal(speed_ratio/2.5,1), num_obs_trip)
                        Speed = ifelse.(Speed .<0, 0, Speed)
    
                            hbrk_ratio = smdata.harshbraking[n]/smdata.total_triplength[n]/360 # per observation
                            
                            turn_ratio = smdata.intensiveturning[n]/smdata.total_triplength[n]/360 # per observation
    
                        Hbrk = rand(Categorical([1-hbrk_ratio, hbrk_ratio]), num_obs_trip).-1
    
                        Turn = rand(Categorical([1-turn_ratio, turn_ratio]), num_obs_trip).-1
    
                        df_new = DataFrame(
                            DriverID = DriverID,
                            DayID = DayID,
                            TripID = TripID, 
                            TripCode = TripCode,
                            Timestamp = Timestamp, 
                            Speed = Speed,
                            Hbrk = Hbrk,
                            Turn = Turn
                            )
    
                        append!(df,df_new)
                    end # trip
                    id += 1
                end # day
            end # person
        end # all drivers
        return df
    end # function
