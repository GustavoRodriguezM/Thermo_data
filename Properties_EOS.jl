using Clapeyron, PyCall, Plots, MAT, JSON
using Base.Filesystem
using Statistics
using PyCall: PyError


traceback = pyimport("traceback")
include("iPCSAFT.jl")
include("TVTPR.jl")
include("VTPR.jl")
include("tcPRC.jl")

bisect = pyimport("bisect").bisect
optimize = pyimport("scipy.optimize")
CoolProp = pyimport("CoolProp")
scipy = pyimport("scipy")
matplotlib = pyimport("matplotlib")
plt = pyimport("matplotlib.pyplot")
np = pyimport("numpy")
curve_fit = pyimport("scipy.optimize").curve_fit
pe = pyimport("matplotlib.patheffects")
cm = matplotlib.cm
colors = pyimport("matplotlib.colors")
sp_interp = pyimport("scipy.interpolate")

CESs = ["tcPRC","cPR","ADPCSAFT", "BACKSAFT", "Berthelot", "CKSAFT", "Clausius", "CPA", "CPPCSAFT", "PR","DAPT", "EPPR78", "GEPCSAFT" , "GEPCSAFT" , "HeterogcPCPSAFT", "HomogcPCPSAFT", "iPCSAFT", "KU", "LJSAFT","ogSAFT", "PatelTeja", "PCPSAFT", "PCSAFT", "pharmaPCSAFT", "PR78","PSRK", "PTV", "QCPR", "OPCSAFT", "RK", "RKPR","SAFTgammaMie","SAFTVRMie", "SAFTVRMie15", "SAFTVRQMie", "SAFTVRSMie", "SAFTVRSW", "sCKSAFT","sCPA", "softSAFT2016","sPCSAFT", "SRK", "structSAFTgammaMie", "tcPR", "tcPRW" ,"tcRK", "TVTPR", "gcsPCSAFT","TWUSRK", "UMRPR", "vdW", "VTPR"] 


SaftVR = read("Saft_VR_mie.json", String)
SaftVRp = JSON.parse(SaftVR)

CIDD = read("CIDs.json", String)
CID = JSON.parse(CIDD)

# Determine the master folder path
#Master_folder = joinpath(@__DIR__,"0.6.10")
Master_folder = @__DIR__

point_distribution = "log"

function check_and_load_matfiles(compound, CES)
    mat_dir = joinpath(Master_folder, "NPZ_files", CES, "$(CES)_$(compound)")
    files = [
        "$(CES) $compound density.mat",
        "$(CES) $compound residual entropy.mat",
        "$(CES) $compound fugacity coefficient.mat",
        "$(CES) $compound Hv.mat",
        "$(CES) $compound pv.mat"
    ]
    all_exist = all(f -> isfile(joinpath(mat_dir, f)), files)
    data = nothing

    if all_exist
        println("CES MAT files exist for $compound. Loading data...")
        data = (
            matread(joinpath(mat_dir, "$(CES) $compound density.mat")),
            matread(joinpath(mat_dir, "$(CES) $compound residual entropy.mat")),
            matread(joinpath(mat_dir, "$(CES) $compound fugacity coefficient.mat")),
            matread(joinpath(mat_dir, "$(CES) $compound Hv.mat")),
            matread(joinpath(mat_dir, "$(CES) $compound pv.mat"))
        )
    end

    return data
end

function vapor_pressure(T, Tc, Pc, param_values)
    tau = 1 .- T ./ Tc
    params = vcat(collect(param_values), zeros(4 - length(param_values)))
    ln_p = (Tc ./ T) .* (params[1] .* tau .+ params[2] .* tau.^1.5 .+ params[3] .* tau.^2 .+ params[4] .* tau.^4)
    return Pc .* np.exp.(ln_p)
end

function vapor_pressure_alt(T, Tc, Pc, param_values)
    params = vcat(collect(param_values), zeros(6 - length(param_values)))
    ln_p = params[1] .+ params[2]./(params[3] .+ T) .+ params[4] .* T .+ params[5] .* T .^2 .+ params[6] .* np.log(T)
    return np.exp.(ln_p)
end

function rho_liquid(T, Tc, rho_c, param_values)
    tau = 1 .- T ./ Tc
    params = vcat(collect(param_values), zeros(4 - length(param_values)))
    ln_rho = (params[1] .* tau.^(2/6) .+ params[2] .* tau.^(3/6) .+ params[3] .* tau.^(7/6) .+ params[4] .* tau.^(9/6))
    return rho_c .* np.exp.(ln_rho)
end

function rho_altern(T, Tc, rho_c, param_values)
    params = vcat(collect(param_values), zeros(4 - length(param_values)))
    rho = rho_c * abs(params[1]) .^ (.-(abs.(1 .- T / Tc)).^params[2] )
    return rho
end

function rho_vapor(T, Tc, rho_c, param_values)
    tau = 1 .- T ./ Tc
    params = vcat(collect(param_values), zeros(5 - length(param_values)))
    ln_rho = (params[1] .* tau.^(2/6) .+ params[2] .* tau.^(4/6) .+ params[3] .* tau.^(7/6) .+
              params[4] .* tau.^(13/6) .+ params[5] .* tau.^(25/6))
    return rho_c .* np.exp.(ln_rho)
end

function adjuster_(x,y,fun,N,method)
    params = Float64[]
    AAD = 10000000
    for num_param in 1:N
        guess = ones(num_param)
        if length(params) > 0
            guess[1:length(params)] .= params
        end
        result = curve_fit(fun, x, y, guess, maxfev=10000,nan_policy="omit",method=method)
        params = collect(result[1])
        pred = fun(x, params...)
        AAD = sum(abs.(pred .- y)) / length(y)
    end
    return params,AAD
end

function adjuster(x,y,fun,N)
    try
        return adjuster_(x,y,fun,N,"lm")
    catch
        try
            return adjuster_(x,y,fun,N,"trf")
        catch
            return adjuster_(x,y,fun,N,"dogbox")
        end
    end
end 


function fit_vapor_pressure(Tc, Pc, T, P_sat, CES, subs; print_AAD=false, plot=false)
    mat_dir = joinpath(Master_folder, "Figures", CES, "$(CES)_$(subs)")
    ensure_directory_exists(mat_dir)
    AAD,params,Psat_fun = nothing, nothing, nothing
    wrapper(T, p...) = vapor_pressure(T, Tc, Pc, p)
    wrapper_alt(T, p...) = vapor_pressure_alt(T, Tc, Pc, p)

    plt.plot(T, P_sat, "y-", label="p_sat")
    plt.legend()

    mask_below_Tc = T .< Tc    
    mask_T_min = T .> Tc * 0
    is_not_nan = np.isfinite(P_sat)

    P_sat = P_sat[mask_below_Tc][mask_T_min][is_not_nan]
    T = T[mask_below_Tc][mask_T_min][is_not_nan]

    P_sat = np.append(P_sat, [Pc])
    T = np.append(T, [Tc])
    
    try
        params,AAD = adjuster(T,P_sat,wrapper,4)
        plt.plot(T, wrapper(T, params...), "k--", label="p_sat fit")
        Psat_fun(T) = wrapper(T, params...)
        test = Psat_fun(Tc)
        if test == 0 || !isfinite(test)
            throw("Wrong Wagner adjust")
        end
    catch
        try
            params,AAD = adjuster(T,P_sat,wrapper_alt,6)
            plt.plot(T, wrapper_alt(T, params...), "k--", label="p_sat fit")
            Psat_fun(T) = wrapper_alt(T, params...)
            #println("Second Psat = $(params)")
        catch
            plt.savefig(joinpath(mat_dir, "vp fits.png"), dpi=600)
            plt.close()
        end
    end
    plt.savefig(joinpath(mat_dir, "vp fits.png"), dpi=600)
    plt.close()

    return Psat_fun
end

function fit_densities(Tc, rho_c, T, rho_sat_vapor, rho_sat_liquid, CES, subs; print_AAD=false, plot=false)

    f_liq,f_vap,liquid_params,vapor_params = nothing, nothing, nothing, nothing

    wrap_liq(T, p...) = rho_liquid(T, Tc, rho_c, p)
    wrap_vap(T, p...) = rho_vapor(T, Tc, rho_c, p)
    wrap_alt(T, p...) = rho_altern(T, Tc, rho_c, p)

    mask_below_Tc = T .< Tc
    mask_T_min = T .> Tc * 0
    is_not_nan = np.isfinite(rho_sat_liquid) .& np.isfinite(rho_sat_vapor)

    T = T[mask_below_Tc][mask_T_min][is_not_nan]
    rho_sat_liquid = rho_sat_liquid[mask_below_Tc][mask_T_min][is_not_nan]
    rho_sat_vapor = rho_sat_vapor[mask_below_Tc][mask_T_min][is_not_nan]
    mat_dir = joinpath(Master_folder, "Figures", CES, "$(CES)_$(subs)")


    try
        liquid_params,AAD = adjuster(T,rho_sat_liquid,wrap_liq,4)
        f_liq = T -> wrap_liq(T, liquid_params...)
    catch
        liquid_params,AAD = adjuster(T,rho_sat_liquid,wrap_alt,2)
        f_liq = T -> wrap_alt(T, liquid_params...)
    end

    try
        vapor_params,AAD = adjuster(T,rho_sat_vapor,wrap_vap,5)
        f_vap = T -> wrap_vap(T, vapor_params...)

    catch
        vapor_params,AAD = adjuster(T,rho_sat_vapor,wrap_alt,2)
        f_vap = T -> wrap_alt(T, vapor_params...)
    end

    if plot
        plt.plot(T,rho_sat_liquid,  "g--", label="liquid")
        plt.plot(T,rho_sat_vapor, "b--", label="vapor")
        plt.plot(T,wrap_liq(T, liquid_params...), "g", label="liquid fit")
        plt.plot(T,wrap_vap(T, vapor_params...), "b", label="vapor fit")
        plt.legend()
        plt.savefig(joinpath(mat_dir, "Density fits.png"))
        plt.close()
    end

    return f_liq, f_vap
end

function ensure_directory_exists(dir_path::String)
    if !ispath(dir_path)
        mkpath(dir_path)
    end
end

function average_surrounding(matrix, i, j)
    sum = 0.0
    count = 0
    for di in -1:1
        for dj in -1:1
            if !(di == 0 && dj == 0) && i + di >= 1 && i + di <= size(matrix, 1) && j + dj >= 1 && j + dj <= size(matrix, 2)
                val = matrix[i + di, j + dj]
                if !isnan(val) && val != 0
                    sum += val
                    count += 1
                end
            end
        end
    end
    return sum / count
end

function SAFTVR_get(Name)
    Prop = CID[Name]
    Saft_parameters = SaftVRp[Prop["CID"]]
    return Saft_parameters, parse(Float64, Prop["Mw"]), parse(Float64, Prop["n_H"]), parse(Float64, Prop["n_e"])
end

function Initiator(CES, Name)
    if CES == "SAFTVRMie"
        Parameters, Mw, H, e = SAFTVR_get(Name)
        if Parameters["epsilonAB"] != "\u2014\u2014"
            a = float(Parameters["epsilonAB"])
            b = float(Parameters["kAB "])
        else
            a = 0
            b = 0
        end
        model = SAFTVRMie([Name]; userlocations=(;
            Mw = [Mw],

            segment = [float(Parameters["m"])],
            sigma = [float(Parameters["sigma"])],
            epsilon = [float(Parameters["epsilon"])],
            lambda_a = [float(Parameters["i_a"])],
            lambda_r = [float(Parameters["i_r"])],
            n_H = [(H)],
            n_e = [(e)],
            epsilon_assoc = Dict(((Name, "e"), (Name, "H")) => a),
            bondvol = Dict(((Name, "e"), (Name, "H")) => b * 10^-30)
        ))

    elseif CES == "CPPCSAFT"
        Param, Mw, H1, e1 = SAFTVR_get(Name)
        pd = pyimport("pandas")
        df = pd.read_excel("CPPCSAFT_parameters.xlsx", index_col=0, decimal=",")
        H,e = nothing,nothing
        segment = float( py"$df.loc[$Name, \"segment\"]")
        sigma_val = float(py"$df.loc[$Name, \"sigma\"]")
        epsilon_val = float(py"$df.loc[$Name, \"epsilon\"]")
        epsilon_assoc_val = convert(Float64, py"$df.loc[$Name, \"epsilon_assoc\"]")
        bondvol_val = convert(Float64, py"$df.loc[$Name, \"bondvol\"]")
        scheme = py"$df.loc[$Name, \"Association\"]"

        if scheme=="4C"
            H = 2
            e = 2
        elseif scheme=="2B"
            H = 1
            e = 1
        elseif scheme=="1A"
            H = 0
            e = 1
        else
            H = 0
            e = 0
        end
        model = CPPCSAFT([Name]; userlocations=(;
            Mw = [Mw],
            segment = [segment],
            sigma = [sigma_val],
            epsilon = [epsilon_val],
            n_H = [(H)],
            n_e = [(e)],
            epsilon_assoc = Dict(((Name, "e"), (Name, "H")) => epsilon_assoc_val),
            bondvol = Dict(((Name, "e"), (Name, "H")) => bondvol_val) * 10^-30))

    elseif CES=="VTPR"
        model =PR([Name]; alpha=TwuAlpha, translation=RackettTranslation)
        
    else
        model = eval(Meta.parse("$(CES)([\"$(Name)\"])"))
    end
    return model
end


function limit_creator(handle,point_distribution,N,T_shift)
    pc = CoolProp.AbstractState.p_critical(handle)
    Tmax = CoolProp.AbstractState.Tmax(handle)
    
    handle.update(CoolProp.QT_INPUTS, 0, CoolProp.AbstractState.Tmin(handle))

    pmin = max(0.001 * pc, handle.p())
    pmax = CoolProp.AbstractState.pmax(handle)

    if pmin == handle.p()
        Tmin = CoolProp.AbstractState.Tmin(handle) + T_shift
    else
        try
            handle.update(CoolProp.PQ_INPUTS, pmin*100, 1)
            Tmin = max(CoolProp.AbstractState.Tmin(handle), handle.T()-50) + T_shift
        catch
            Tmin = CoolProp.AbstractState.Tmin(handle)+T_shift
        end
    end
    
    if point_distribution=="linear"
        P = collect(LinRange(pmin, pmax, N))
    elseif point_distribution=="log"
        P = exp10.(LinRange(log10(pmin), log10(pmax), N))
    end

    T = LinRange(Tmin, Tmax, N)
    return np.array(T),np.array(P)
end

function Error_spliter(Matrix, vp_CES, N, P, T, Tc_CES, Pc_CES, Tsat_CES , Psat_adj)

    nan_index = findall(!, isfinite.(vp_CES))
    
    if !isempty(nan_index)
        nan_index = np.where(np.isnan(vp_CES))[1]

        for index in nan_index
            Tsel = Tsat_CES[index + 1]
            Ppre = Psat_adj(Tsel)

            vp_CES[index + 1] = Psat_adj(Tsel)
        end
    end

    Pc_ind = bisect(P, Pc_CES) + 1
    Tc_ind = bisect(T, Tc_CES) + 1
    vp_ind = [bisect(P, i) + 1 for i in vp_CES]

    mask_supercritical = fill(false, size(Matrix))
    mask_liquid = fill(false, size(Matrix))
    mask_gas = fill(false, size(Matrix))

    mask_supercritical[1:N, Pc_ind+1:N] .= true
    mask_gas[Tc_ind:N, 1:Pc_ind] .= true

    for (vp_i, T_CES) in zip(vp_ind, Tsat_CES)
        Tsat_ind = bisect(T, T_CES)
        mask_gas[Tsat_ind, 1:vp_i] .= true
        mask_liquid[Tsat_ind, vp_i+1:Pc_ind] .= true
    end

    regions = Dict(
        "supercritical" => mask_supercritical,
        "gas" => mask_gas,
        "liquid" => mask_liquid
    )

    return regions
end


function interpolate_zone(Rho_CES,T,P; method::String="linear")
    valid_mask = np.isfinite(Rho_CES)
    nan_mask = .!np.isfinite(Rho_CES)

    points = np.column_stack((np.ravel(T[valid_mask]), np.ravel(P[valid_mask])))
    points_all = np.column_stack((np.ravel(T), np.ravel(P)))
    xi = np.column_stack((np.ravel(T[nan_mask]), np.ravel(P[nan_mask])))
    values = np.ravel(Rho_CES[valid_mask])
    interpolated_ = sp_interp.griddata(points,values,points_all, method=method)
    complete_matrix = reshape(interpolated_, size(Rho_CES))
    return complete_matrix 
end

function neighbor_filler(Tsat, Rho_sat_liq_CES, Rho_sat_vap_CES, pv_CES, model)
    is_nan = .!isfinite.(Rho_sat_liq_CES) .| .!isfinite.(Rho_sat_vap_CES)
    
    filled = falses(length(Tsat))
    queue = np.where(is_nan)[1]
    queue = queue .+ 1

    while !isempty(queue)
        i = pop!(queue)

        # Skip if already filled
        if !is_nan[i] || filled[i]
            continue
        end

        # Try using neighbor(s) as initial guesses
        initial_p = nothing
        initial_vl = nothing
        initial_vv = nothing

        # Priority: use closest filled neighbors
        for offset in (-1, 1)
            j = i + offset
            if j >= 1 && j <= length(Tsat) && isfinite(Rho_sat_liq_CES[j]) && isfinite(Rho_sat_vap_CES[j])
                initial_p = pv_CES[j]
                initial_vl = Rho_sat_liq_CES[j]
                initial_vv = Rho_sat_vap_CES[j]
                break
            end
        end

        if initial_p === nothing
            continue
        end

        try
            pv, vl, vv = saturation_pressure(
                model,
                Tsat[i],
                IsoFugacitySaturation(p0 = initial_p, vl = initial_vl, vv = initial_vv)
            )

            pv_CES[i] = pv
            Rho_sat_liq_CES[i] = vl
            Rho_sat_vap_CES[i] = vv
            filled[i] = true
            is_nan[i] = false

            # Add neighboring NaNs to queue
            for offset in (-1, 1)
                j = i + offset
                if j >= 1 && j <= length(Tsat) && is_nan[j] && !(j in queue)
                    push!(queue, j)
                end
            end
        catch
            continue  # skip if saturation_pressure fails
        end
    end

    return pv_CES, Rho_sat_liq_CES, Rho_sat_vap_CES
end


function interpolator(Rho_CES::Array{<:Real,2}, T::Vector{<:Real}, P::Vector{<:Real},vp_CES , Tc_CES, Pc_CES , Tsat_CES, Psat_adj; method::String="linear")
    Pgrid, Tgrid = np.meshgrid(P, T)
    regions = Error_spliter(Rho_CES, vp_CES, 500,P,T , Tc_CES, Pc_CES, Tsat_CES,Psat_adj)
    sup_matrix = interpolate_zone(Rho_CES[regions["supercritical"]],Tgrid[regions["supercritical"]],Pgrid[regions["supercritical"]]; method=method)
    gas_matrix = interpolate_zone(Rho_CES[regions["gas"]],Tgrid[regions["gas"]],Pgrid[regions["gas"]]; method=method)
    liq_matrix = interpolate_zone(Rho_CES[regions["liquid"]],Tgrid[regions["liquid"]],Pgrid[regions["liquid"]]; method=method)
    inter = np.zeros(np.shape(Rho_CES))
    inter[regions["supercritical"]] = sup_matrix
    inter[regions["gas"]] = gas_matrix
    inter[regions["liquid"]] = liq_matrix
    
    return inter
end


function fitter_filler(model,Tsat,Rho_sat_liq_CES,Rho_sat_vap_CES,Phi_sat_liq_CES,Phi_sat_vap_CES,Sres_sat_liq_CES,Sres_sat_vap_CES,Hv_CES,pv_CES,Vc,Tc, Pc, CES, compound)
    Rho_sat_liq_CES[Rho_sat_liq_CES .< 1/Vc] .= NaN
    Rho_sat_vap_CES[Rho_sat_vap_CES .> 1/Vc] .= NaN



    P_fit,liq, vap = nothing, nothing, nothing
    if any(np.isnan(Rho_sat_liq_CES)) || any(np.isnan(Rho_sat_vap_CES)) || any(np.isinf(Rho_sat_liq_CES)) || any(np.isinf(Rho_sat_vap_CES))
        P_fit = fit_vapor_pressure(Tc, Pc, Tsat, pv_CES, CES, compound; print_AAD=false, plot=true)
        #println(Tc, 1/Vc, Tsat, Rho_sat_vap_CES, Rho_sat_liq_CES, CES)
        liq, vap = fit_densities(Tc, 1/Vc, Tsat, Rho_sat_vap_CES, Rho_sat_liq_CES, CES, compound; print_AAD=false, plot=true) 

        for (t_ind, t) in enumerate(Tsat)
            idx = t_ind
            
            if (!(isnan(Rho_sat_liq_CES[idx]) || isnan(Rho_sat_vap_CES[idx]) || isinf(Rho_sat_liq_CES[idx]) || isinf(Rho_sat_vap_CES[idx])))
                continue
            else
                (pv, vl, vv) = saturation_pressure(model, t, IsoFugacitySaturation(p0 = P_fit(t), vl = liq(t), vv = vap(t)))
                if isnan(vv)
                    (pv, vl, vv) = saturation_pressure(model, t, ChemPotVSaturation(vl = liq(t), vv = vap(t)))
                    if isnan(vv)
                        i_used = (t_ind - 1 > 0) ? (t_ind - 1) : (t_ind + 1)
                        (pv, vl, vv) = saturation_pressure(model, t, IsoFugacitySaturation(p0 = pv_CES[i_used], vl = liq(t), vv = vap(t)))                    
                        if isnan(vv)
                            try 
                                (pv, vl, vv) = saturation_pressure(model, t, IsoFugacitySaturation(p0 = pv_CES[t_ind+1], vl = liq(t), vv = vap(t)))                    
                            catch BoundsError
                                (pv, vl, vv) = saturation_pressure(model, t, IsoFugacitySaturation(p0 = Pc*(t/Tc), vl = liq(t), vv = vap(t))) 
                                
                            end
                            if isnan(vv)
                                (pv, vl, vv) = saturation_pressure(model, t, IsoFugacitySaturation(p0 =  pv_CES[t_ind-1], vl = 1/Rho_sat_liq_CES[t_ind-1], vv = 1/Rho_sat_vap_CES[t_ind-1]))      
                            end
                        end
                    end
                end
            end

            Rho_sat_liq_CES[idx] = 1 / vl
            Rho_sat_vap_CES[idx] = 1 / vv
            Phi_sat_liq_CES[idx] = Clapeyron.VT_fugacity_coefficient(model, vl, t, [1.])[1]
            Phi_sat_vap_CES[idx] = Clapeyron.VT_fugacity_coefficient(model, vv, t, [1.])[1]
            Sres_sat_liq_CES[idx] = Clapeyron.VT_entropy_res(model, vl, t, [1.])
            Sres_sat_vap_CES[idx] = Clapeyron.VT_entropy_res(model, vv, t, [1.])
            hv = Clapeyron.VT_enthalpy(model, vv, t, [1.])
            hl = Clapeyron.VT_enthalpy(model, vl, t, [1.])
            Hv_CES[idx] = hv - hl
            pv_CES[idx] = pv
        end
    end
    return Rho_sat_liq_CES,Rho_sat_vap_CES,Phi_sat_liq_CES,Phi_sat_vap_CES,Sres_sat_liq_CES,Sres_sat_vap_CES,Hv_CES,pv_CES,P_fit,liq, vap
end

function interpolator_filler(Rho_CES,T, P,pv_CES,Tc, Pc, Tsat,CES, compound,model,Sres_CES,Phi_CES)
    if any(np.isnan(Rho_CES)) || any(np.isinf(Rho_CES))
        P_fit = fit_vapor_pressure(Tc, Pc, Tsat, pv_CES, CES, compound; print_AAD=false, plot=true)
        Rho_pre_linear = interpolator(Rho_CES, T, P,pv_CES , Tc, Pc , Tsat,P_fit,method="linear")
        Rho_pre_cubic = interpolator(Rho_CES, T, P,pv_CES , Tc, Pc , Tsat,P_fit,method="cubic")
        Rho_pre_near = interpolator(Rho_CES, T, P,pv_CES , Tc, Pc , Tsat,P_fit,method="nearest")
        Tmis, Pmis,Tmis_idx,Pmis_idx = [],[],[],[]
        for (t_idx, t) in enumerate(T)
            for (p_idx, p) in enumerate(P)
                if  isfinite(Rho_CES[t_idx, p_idx]) 
                    continue
                elseif !np.isfinite(Rho_CES[t_idx, p_idx])
                    volume_calc = volume(model, p, t, vol0 = 1 /Rho_pre_near[t_idx,p_idx] )
                    if isnan(volume_calc)
                        volume_calc = volume(model, p, t, vol0 = 1 /Rho_pre_linear[t_idx,p_idx])
                        if isnan(volume_calc)
                            volume_calc = volume(model, p, t, vol0 = 1 /Rho_pre_cubic[t_idx,p_idx])
                            if isnan(volume_calc)
                                Delta_T = T[2] - T[1]
                                Delta_P = P[2] - P[1]
                                continue_T_up = true 
                                continue_T_do = true 
                                continue_P_up = true 
                                continue_P_do = true 

                                T_up_final = nothing
                                T_do_final = nothing
                                P_up_final = nothing
                                P_do_final = nothing

                                volume_P_up = 1/Rho_CES[t_idx, p_idx + 1]
                                volume_P_do = 1/Rho_CES[t_idx, p_idx - 1]
                                volume_T_up = 1/Rho_CES[t_idx + 1, p_idx]
                                volume_T_do = 1/Rho_CES[t_idx - 1, p_idx]

                                P_up = Rho_CES[t_idx, p_idx + 1]
                                P_do = Rho_CES[t_idx, p_idx - 1]
                                T_up = Rho_CES[t_idx + 1, p_idx]
                                T_do = Rho_CES[t_idx - 1, p_idx]
                                N = 100000000
                                for i in 1:N
                                    if continue_T_up
                                        volume_T_up = volume(model, p, t+ i*Delta_T/N)
                                        if isnan(volume_T_up)
                                            volume_T_up = volume(model, p, t+ i*Delta_T/N, vol0 = T_up)
                                        end
                                        if isfinite(volume_T_up)
                                            continue_T_up = false
                                            T_up_final = t+ i*Delta_T/N
                                        end
                                    end
                                    if continue_T_do
                                        volume_T_do = volume(model, p, t - i*Delta_T/N)
                                        if isnan(volume_T_do)
                                            volume_T_do = volume(model, p, t - i*Delta_T/N, vol0 = T_do)
                                        end

                                        if isfinite(volume_T_do)
                                            continue_T_do = false
                                            T_do_final = t - i*Delta_T/N
                                        end
                                    end
                                    
                                    if continue_P_up
                                        volume_P_up = volume(model, p + i*Delta_P/N, t)
                                        if isnan(volume_P_up)
                                            volume_P_up = volume(model, p + i*Delta_P/N, t, vol0 = P_up)
                                        end

                                        if isfinite(volume_P_up)
                                            continue_P_up = false
                                            P_up_final = p + i*Delta_P/N
                                        end
                                    end
                                    
                                    if continue_P_do
                                        volume_P_do = volume(model, p - i*Delta_P/N, t)

                                        if isnan(volume_P_do)
                                            volume_P_do = volume(model, p - i*Delta_P/N, t, vol0 = P_do)
                                        end

                                        if isfinite(volume_P_do)
                                            continue_P_do = false
                                            P_do_final = p - i*Delta_P/N
                                        end
                                    end
                                end
                                print("t = ",t,", p =",p)
                                print("t up")
                                print("i = ",(T_up_final-t)/Delta_T)
                                print(volume_T_up)
                                print("t do")
                                print("i = ",-(T_do_final-t)/Delta_T)
                                print(volume_T_do)
                                print("P up")
                                print("i = ",(P_up_final-p)/Delta_P)
                                print(volume_P_up)
                                print("P do")
                                print("i = ",-(P_do_final-p)/Delta_P)
                                print(volume_P_do)

                                
                                volume_average = (volume_P_do + volume_P_up + volume_T_do + volume_T_up)/4
                                volume_calc = volume(model, p, t, vol0 = volume_average )
                                if isnan(volume_calc)
                                    known_grid =  np.array([
                                        [T_up_final, p],
                                        [T_do_final, p],
                                        [t, P_up_final],
                                        [t, P_do_final]])
                                    
                                    known_points = np.array([volume_T_up, volume_T_do, volume_P_up, volume_P_do])
                                    sought_point = np.array([[t, p]])
                                    results_nearest = griddata(known_grid, known_points, sought_point, method="nearest")
                                    volume_calc = volume(model, p, t, vol0 = results_nearest )
                                    
                                    if isnan(volume_calc)
                                        results_linear = griddata(known_grid, known_points, sought_point, method="linear")
                                        volume_calc = volume(model, p, t, vol0 = results_linear )
                                        if isnan(volume_calc)
                                            results_cubic = griddata(known_grid, known_points, sought_point, method="cubic")
                                            volume_calc = volume(model, p, t, vol0 = results_cubic )
                                        end

                                    end
                                end


                            end
                        end
                    end

                    density_value = 1 / volume_calc 
                    if !np.isfinite(density_value)
                        push!(Tmis,t)
                        push!(Pmis,p)
                        push!(Tmis_idx,t_idx)
                        push!(Pmis_idx,p_idx)
                    end
                    Rho_CES[t_idx, p_idx] = density_value
                    Sres_CES[t_idx, p_idx] = Clapeyron.VT_entropy_res(model, 1 / density_value, t, [1.])
                    Phi_CES[t_idx, p_idx] = Clapeyron.VT_fugacity_coefficient(model, 1 / density_value, t, [1.])[1]
                
                end
            end
        end

        if any(np.isnan(Rho_CES)) || any(np.isinf(Rho_CES))
            Rho_pre = interpolator(Rho_CES, T, P,pv_CES , Tc, Pc , Tsat, P_fit)
            for (t,p,t_idx,p_idx) in zip(Tmis,Pmis,Tmis_idx,Pmis_idx)
                predicted = Rho_pre_cubic[t_idx,p_idx]
                if !np.isfinite(predicted)
                    predicted = Rho_pre_linear[t_idx,p_idx]
                    if !np.isfinite(predicted)
                        predicted = Rho_pre_near[t_idx,p_idx]
                        if !np.isfinite(predicted)
                            continue
                        end
                    end                
                end
                density_value = 1 / volume(model, p, t, vol0 = 1 / predicted)
                Rho_CES[t_idx, p_idx] = density_value
                Sres_CES[t_idx, p_idx] = Clapeyron.VT_entropy_res(model, 1 / density_value, t, [1.])
                Phi_CES[t_idx, p_idx] = Clapeyron.VT_fugacity_coefficient(model, 1 / density_value, t, [1.])[1]

            end
        end
    end
    return Rho_CES, Sres_CES, Phi_CES
end

function density_CES(compound, CES; T_shift = 0.0)

    N = 500

    handle = CoolProp.AbstractState("HEOS", compound)
    model = Initiator(CES, compound)

 
    T,P = limit_creator(handle,point_distribution,N,T_shift)

    (Tc, Pc, Vc) = crit_pure(model)
    Tsat = T[T .< Tc] 
    
    #println("Tc = $(Tc)")
    #println("Pc = $(Pc)")
    Rho_CES, Phi_CES, Sres_CES, Rho_sat_liq_CES, Phi_sat_liq_CES, Sres_sat_liq_CES, Rho_sat_vap_CES, Phi_sat_vap_CES, Sres_sat_vap_CES, pv_CES, Hv_CES = fill(NaN, length(T), length(P)), zeros(Float64, length(T), length(P)), zeros(Float64, length(T), length(P)), zeros(length(Tsat)), zeros(length(Tsat)), zeros(length(Tsat)), zeros(length(Tsat)), zeros(length(Tsat)), zeros(length(Tsat)), zeros(length(Tsat)), zeros(length(Tsat))
    for (t_idx, t) in enumerate(Tsat)
        (pv, vl, vv) = saturation_pressure(model, t)
        if vl > vv
            vl, vv = vv, vl
        end

        hl = Clapeyron.VT_enthalpy(model, vl, t, [1.])
        hv = Clapeyron.VT_enthalpy(model, vv, t, [1.])

        Rho_sat_liq_CES[t_idx] = 1 / vl
        Rho_sat_vap_CES[t_idx] = 1 / vv
        Phi_sat_liq_CES[t_idx] = Clapeyron.VT_fugacity_coefficient(model, vl, t, [1.])[1]
        Phi_sat_vap_CES[t_idx] = Clapeyron.VT_fugacity_coefficient(model, vv, t, [1.])[1]
        Sres_sat_liq_CES[t_idx] = Clapeyron.VT_entropy_res(model, vl, t, [1.])
        Sres_sat_vap_CES[t_idx] = Clapeyron.VT_entropy_res(model, vv, t, [1.])
        Hv_CES[t_idx] = hv - hl
        pv_CES[t_idx] = pv
        
    end
    #pv_CES, Rho_sat_liq_CES, Rho_sat_vap_CES = neighbor_filler(Tsat, Rho_sat_liq_CES, Rho_sat_vap_CES, pv_CES, model)
    #Rho_sat_liq_CES,Rho_sat_vap_CES,Phi_sat_liq_CES,Phi_sat_vap_CES,Sres_sat_liq_CES,Sres_sat_vap_CES,Hv_CES,pv_CES,P_fit,liq, vap = fitter_filler(model,Tsat,Rho_sat_liq_CES,Rho_sat_vap_CES,Phi_sat_liq_CES,Phi_sat_vap_CES,Sres_sat_liq_CES,Sres_sat_vap_CES,Hv_CES,pv_CES,Vc,Tc, Pc, CES, compound)
    for (t_idx, t) in enumerate(T)
        #println("Pc = ",Pc)
        for (p_idx, p) in enumerate(P)
            #println("p_idx = ",p_idx," p = ",p)
            if p < Pc && t < Tc
                if p < pv_CES[t_idx]
                    density_value = 1 / volume(model, p, t; phase = :vapor) 
                else
                    density_value = 1 / volume(model, p, t; phase = :liquid)
                end
            else
                density_value = 1 / volume(model, p, t)
            end
            """
            if isnan(density_value)
                try
                    density_value = 1 / volume(model, p, t,vol0 = 1 /Rho_CES[t_idx, p_idx-1])
                catch
                    density_value = 1 / volume(model, p, t,vol0 = 1 /Rho_CES[t_idx, p_idx+1])
                end
            end
            """
            #println("density_value = $(density_value)")
            Rho_CES[t_idx, p_idx] = density_value
            Sres_CES[t_idx, p_idx] = Clapeyron.VT_entropy_res(model, 1 / density_value, t, [1.])
            Phi_CES[t_idx, p_idx] = Clapeyron.VT_fugacity_coefficient(model, 1 / density_value, t, [1.])[1]
        end
    end

    #Rho_CES, Sres_CES, Phi_CES = interpolator_filler(Rho_CES,T, P,pv_CES,Tc, Pc, Tsat,CES, compound,model,Sres_CES,Phi_CES )

    print(np.where(np.isnan(Rho_CES)))
    mat_dir = joinpath(Master_folder, "NPZ_files", CES, "$(CES)_$(compound)")
    println("Tsat = ",Tsat)
    println("pv_CES = ", pv_CES)
    ensure_directory_exists(mat_dir)
    matwrite(joinpath(mat_dir, "$(CES) $compound density.mat"), Dict("Rho_CES" => Rho_CES, "Rho_sat_liq" => Rho_sat_liq_CES, "Rho_sat_vap" => Rho_sat_vap_CES,"T" =>T,"P" =>P,"Tc" =>Tc,"Pc" => Pc,"Vc" => Vc,"T_shift" =>T_shift))
    matwrite(joinpath(mat_dir, "$(CES) $compound residual entropy.mat"), Dict("Sres_CES" => Sres_CES, "Sres_sat_liq_CES" => Sres_sat_liq_CES, "Sres_sat_vap_CES" => Sres_sat_vap_CES,"T" =>T,"P" =>P,"Tc" =>Tc,"Pc" => Pc,"Vc" => Vc,"T_shift" =>T_shift))
    matwrite(joinpath(mat_dir, "$(CES) $compound fugacity coefficient.mat"), Dict("Phi_CES" => Phi_CES, "Phi_sat_liq_CES" => Phi_sat_liq_CES, "Phi_sat_vap_CES" => Phi_sat_vap_CES,"T" =>T,"P" =>P,"Tc" =>Tc,"Pc" => Pc,"Vc" => Vc,"T_shift" =>T_shift))
    matwrite(joinpath(mat_dir, "$(CES) $compound Hv.mat"), Dict("Hv_CES" => Hv_CES,"Tsat" =>Tsat,"T_shift" =>T_shift))
    matwrite(joinpath(mat_dir, "$(CES) $compound pv.mat"), Dict("pv_CES" => pv_CES,"Tsat" =>Tsat,"T_shift" =>T_shift))
    
    return T, P, Rho_CES, Sres_CES, Hv_CES, pv_CES, Rho_sat_liq_CES, Rho_sat_vap_CES, Sres_sat_liq_CES, Sres_sat_vap_CES, Phi_CES, Phi_sat_liq_CES, Phi_sat_vap_CES, Tsat, Tc, Pc
end

function density_CP(compound; T_shift = 0.0)
    N = 500

    handle = CoolProp.AbstractState("HEOS", compound)

    T,P = limit_creator(handle,point_distribution,N,T_shift)
    Tc = CoolProp.AbstractState.T_critical(handle)
    Pc = CoolProp.AbstractState.p_critical(handle)

    Tsat = T[T .< Tc] 

    Rho_CP, Phi_CP, Sres_CP, Rho_sat_liq_CP, Phi_sat_liq_CP, Sres_sat_liq_CP, Rho_sat_vap_CP, Phi_sat_vap_CP, Sres_sat_vap_CP, pv_CP, Hv_CP = zeros(Float64, length(T), length(P)), zeros(Float64, length(T), length(P)), zeros(Float64, length(T), length(P)), zeros(length(Tsat)), zeros(length(Tsat)), zeros(length(Tsat)), zeros(length(Tsat)), zeros(length(Tsat)), zeros(length(Tsat)), zeros(length(Tsat)), zeros(length(Tsat))

    for t in T
        if t in Tsat

            handle.update(CoolProp.QT_INPUTS, 0, t)
            hl = CoolProp.CoolProp.AbstractState.hmolar(handle)
            pv_CP[t .== Tsat] .= CoolProp.CoolProp.AbstractState.p(handle)
            Rho_sat_liq_CP[t .== Tsat] .= CoolProp.CoolProp.AbstractState.rhomolar(handle)
            Phi_sat_liq_CP[t .== Tsat] .= CoolProp.CoolProp.AbstractState.fugacity_coefficient(handle, 0)
            Sres_sat_liq_CP[t .== Tsat] .= CoolProp.CoolProp.AbstractState.smolar_residual(handle)

            handle.update(CoolProp.QT_INPUTS, 1, t)
            hv = CoolProp.CoolProp.AbstractState.hmolar(handle)
            Hv_CP[t .== Tsat] .= hv - hl
            Rho_sat_vap_CP[t .== Tsat] .= CoolProp.CoolProp.AbstractState.rhomolar(handle)
            Phi_sat_vap_CP[t .== Tsat] .= CoolProp.CoolProp.AbstractState.fugacity_coefficient(handle, 0)
            Sres_sat_vap_CP[t .== Tsat] .= CoolProp.CoolProp.AbstractState.smolar_residual(handle)
        end

        for pr in P
            handle.update(CoolProp.PT_INPUTS, pr, t)
            Rho_CP[T .== t, P .== pr] .= CoolProp.CoolProp.AbstractState.rhomolar(handle)
            Sres_CP[T .== t, P .== pr] .= CoolProp.CoolProp.AbstractState.smolar_residual(handle)
            Phi_CP[T .== t, P .== pr] .= CoolProp.CoolProp.AbstractState.fugacity_coefficient(handle, 0)
        end
    end

    mat_dir = joinpath(Master_folder, "NPZ_files", "CoolProp", "CoolProp_$(compound)")
    ensure_directory_exists(mat_dir)
    #print(Rho_sat_liq_CP)

    matwrite(joinpath(mat_dir, "Coolprop $compound density.mat"), Dict("Rho_CP" => Rho_CP, "Rho_sat_liq" => Rho_sat_liq_CP, "Rho_sat_vap" => Rho_sat_vap_CP, "T" => T, "P" => P,"Tc" =>Tc,"Pc" => Pc,"T_shift" =>T_shift,"Tc" => Tc, "Pc" => Pc))
    matwrite(joinpath(mat_dir, "Coolprop $compound residual entropy.mat"), Dict("Sres_CP" => Sres_CP, "Sres_sat_liq_CP" => Sres_sat_liq_CP, "Sres_sat_vap_CP" => Sres_sat_vap_CP, "T" => T, "P" => P,"Tc" =>Tc,"Pc" => Pc,"T_shift" =>T_shift,"Tc" => Tc, "Pc" => Pc))
    matwrite(joinpath(mat_dir, "Coolprop $compound fugacity coefficient.mat"), Dict("Phi_CP" => Phi_CP, "Phi_sat_liq_CP" => Phi_sat_liq_CP, "Phi_sat_vap_CP" => Phi_sat_vap_CP, "T" => T, "P" => P,"Tc" =>Tc,"Pc" => Pc,"T_shift" =>T_shift,"Tc" => Tc, "Pc" => Pc))
    matwrite(joinpath(mat_dir, "Coolprop $compound Hv.mat"), Dict("Hv_CP" => Hv_CP, "Tsat" => Tsat,"T_shift" =>T_shift,"Tc" => Tc, "Pc" => Pc))
    matwrite(joinpath(mat_dir, "Coolprop $compound pv.mat"), Dict("pv_CP" => pv_CP, "Tsat" => Tsat,"T_shift" =>T_shift,"Tc" => Tc, "Pc" => Pc))
    
    return T, P, Rho_CP, Sres_CP, Hv_CP, pv_CP, Rho_sat_liq_CP, Rho_sat_vap_CP, Sres_sat_liq_CP, Sres_sat_vap_CP, Phi_CP, Phi_sat_liq_CP, Phi_sat_vap_CP, Tsat, Tc, Pc
end


function graph_(CES, Name, subs, eos_data, exp_data, sat_liq_exp, sat_vap_exp,
               sat_liq_eos, sat_vap_eos, vp_cool, vp_eos, Tsat_cool, Tsat_eos,
               Tc_cool, Tc_eos, Pc_cool, Pc_eos, T, P)

    eos_style = ":"
    coo_style = "-"
    eos_color = "k"
    coo_color = "k"
    P_fit = fit_vapor_pressure(Tc_eos, Pc_eos, Tsat_eos, sat_vap_eos, CES, Name; print_AAD=false, plot=false)
    mat_dir = joinpath(Master_folder, "Figures", CES, "$(CES)_$(subs)")
    ensure_directory_exists(mat_dir)

    Tmin = minimum(T)
    Pmin = minimum(P)

    handle = CoolProp.AbstractState("HEOS", subs)
    lowest_T(temperature) = sat_prop(handle, temperature, "p", 1) - Pmin
    try
        Tlow = scipy.optimize.fsolve(lowest_T, (Tc_cool + Tmin) / 2)[1]
    catch
        Tlow = Tmin
    end
    Plin, Tlin = P, T
    P, T = np.meshgrid(P, T)

    levels = LinRange(0, 30, 11)
    cmap = matplotlib.cm.get_cmap("RdYlGn_r")
    cmap[:set_over]("red")
    norm = matplotlib.colors.BoundaryNorm(levels, ncolors = cmap[:N], clip = false)

    plt.figure(2^3)
    plt.title("$(CES) $(Name) error for $(subs)")
    plt.yscale("log")
    Error = (abs.(eos_data .- exp_data) .* 100 ./ abs.(exp_data))
    contour = plt.contourf(T ./ Tc_cool, P ./ Pc_cool, Error, levels = levels, cmap = cmap, extend = "max")
    plt.colorbar(contour, label = "Error (%)")
    plt.grid()
    plt.xlabel("\$T_{r}\$ [-]")
    plt.ylabel("\$P_{r}\$ [-]")
    plt.gca()[:set_ylim](bottom = 0.01)
    

    if Tc_cool > Tc_eos
        Tsat = Tsat_eos
        sat_liq_exp = sat_liq_exp[Tsat_cool .<= Tc_eos]
        sat_vap_exp = sat_vap_exp[Tsat_cool .<= Tc_eos]
        errors_liq = abs.(sat_liq_exp .- sat_liq_eos) *100 ./ sat_liq_exp
        errors_vap = abs.(sat_vap_exp .- sat_vap_eos) *100 ./ sat_vap_exp
        errors = (errors_liq .+ errors_vap) ./ 2
        Tsam = Tsat
        Vpsam = vp_eos



        plt.fill_between(Tsat_eos ./ Tc_cool, vp_eos ./ Pc_cool, vp_cool[Tsat_cool .<= Tc_eos] ./ Pc_cool, color="w")
        plt.plot(Tsat_cool ./ Tc_cool, vp_cool ./ Pc_cool, linestyle = coo_style, linewidth = 1.5, color = coo_color)

    elseif Tc_cool < Tc_eos
        Tsat = Tsat_cool
        sat_liq_eos = sat_liq_eos[Tsat_eos .<= Tc_cool]
        sat_vap_eos = sat_vap_eos[Tsat_eos .<= Tc_cool]
        errors_liq = abs.(sat_liq_exp .- sat_liq_eos) *100 ./ sat_liq_exp
        errors_vap = abs.(sat_vap_exp .- sat_vap_eos) *100 ./ sat_vap_exp
        errors = (errors_liq .+ errors_vap) ./ 2
        Tsam = Tsat
        Vpsam = vp_cool

        # Interpolate vp_cool to match Tsat_eos
        #println(Tsat_cool, Tc_cool, vp_cool , Pc_cool)
        plt.fill_between(Tsat_cool ./ Tc_cool, vp_cool ./ Pc_cool, vp_eos[Tsat_eos .<= Tc_cool] ./ Pc_cool, color="white", zorder=0)
        plt.plot(Tsat_eos ./ Tc_cool, vp_eos ./ Pc_cool, linestyle = eos_style, linewidth = 1.5, color = eos_color)

    else
        Tsat = Tsat_cool
        errors_liq = abs.(sat_liq_exp .- sat_liq_eos) *100 ./ sat_liq_exp
        errors_vap = abs.(sat_vap_exp .- sat_vap_eos) *100 ./ sat_vap_exp
        errors = (errors_liq .+ errors_vap) ./ 2
        Tsam = Tsat
        Vpsam = vp_cool
        plt.plot(Tsat_eos ./ Tc_cool, vp_eos ./ Pc_cool, linestyle = eos_style, linewidth = 1.5, color = eos_color)
    end
    
    norm = colors.BoundaryNorm(np.array(levels), cmap.N)
    plt.plot(Tsam./ Tc_cool, Vpsam./ Pc_cool, color=eos_color, linewidth=2.5)
    for i in 1:length(Tsam)-1
        x = [Tsam[i]/Tc_cool, Tsam[i+1]/Tc_cool]
        y = [Vpsam[i]/Pc_cool, Vpsam[i+1]/Pc_cool]
        color_value = errors[i]
        color = cmap(norm(color_value))
        
        plt.plot(x, y, color=color, linewidth=1.5)
        
    end
    
    plt.axvline(x = 1, linestyle = coo_style, linewidth = 1.5, color = coo_color)
    plt.axhline(y = 1, linestyle = coo_style, linewidth = 1.5, color = coo_color)
    plt.axvline(x = Tc_eos/Tc_cool, linestyle = eos_style, linewidth = 1.5, color = eos_color)
    plt.axhline(y = Pc_eos/Pc_cool, linestyle = eos_style, linewidth = 1.5, color = eos_color)
    
    plt.savefig(joinpath(mat_dir, "$(CES) $(Name) $(subs) big.png"), dpi=1500)
    plt.close()

    plt.figure(3^7)
    plt.title("$(CES) $(Name) error for $(subs)")
    plt.yscale("log")
    contour = plt.contourf((T ./ Tc_cool), (P ./ Pc_cool), Error, levels=levels, cmap=cmap, extend="max")
    plt.colorbar(contour, label = "Error (%)")
    plt.grid()
    plt.xlabel("\$T_{r}\$ [-]")
    plt.ylabel("\$P_{r}\$ [-]")
    plt.gca()[:set_ylim](bottom = 0.01)

    if Tc_cool > Tc_eos
        plt.plot(Tsat_cool ./ Tc_cool, vp_cool ./ Pc_cool, linestyle = coo_style, linewidth = 1.5, color = coo_color)
        plt.fill_between(Tsat_eos ./ Tc_cool, vp_eos ./ Pc_cool, vp_cool[Tsat_cool .<= Tc_eos] ./ Pc_cool, color="w")

    elseif Tc_cool < Tc_eos
        plt.fill_between(Tsat_cool ./ Tc_cool, vp_cool ./ Pc_cool, vp_eos[Tsat_eos .<= Tc_cool] ./ Pc_cool, color="white")
        plt.plot(Tsat_eos ./ Tc_cool, vp_eos ./ Pc_cool, linestyle = eos_style, linewidth = 1.5, color = eos_color)
        
    else
        plt.plot(Tsat_eos ./ Tc_cool, vp_eos ./ Pc_cool, linestyle = eos_style, linewidth = 1.5, color = eos_color)
    end

    norm = colors.BoundaryNorm(np.array(levels), cmap.N)
    plt.plot(Tsam./ Tc_cool, Vpsam./ Pc_cool, color=eos_color, linewidth=2.5)
    for i in 1:length(Tsam)-1
        x = [Tsam[i]/Tc_cool, Tsam[i+1]/Tc_cool]
        y = [Vpsam[i]/Pc_cool, Vpsam[i+1]/Pc_cool]
        color_value = errors[i]
        color = cmap(norm(color_value))

        plt.plot(x, y, color=color, linewidth=1.5)
    end

    
    plt.axvline(x = 1, linestyle = coo_style, linewidth = 1.5, color = coo_color)
    plt.axhline(y = 1, linestyle = coo_style, linewidth = 1.5, color = coo_color)
    plt.axvline(x = Tc_eos/Tc_cool, linestyle = eos_style, linewidth = 1.5, color = eos_color)
    plt.axhline(y = Pc_eos/Pc_cool, linestyle = eos_style, linewidth = 1.5, color = eos_color)
    
    
    plt.xlim(0.7, 1.2)
    plt.ylim(0.01, 10)
    plt.savefig(joinpath(mat_dir, "$(CES) $(Name) $(subs) small.png"), dpi=1500)
    plt.close()

    
    #plt.title("$(CES) $(Name) vs Temperature at Constant Pressures for $(subs)")
    #plt.xlabel("Temperature [K]")
    #plt.ylabel("$(Name)")
    """plt.grid()
    indices = round.(Int, range(1, 500, length=15))

    plt.axvline(x = Tc_eos, linestyle = eos_style, linewidth = 1.5, color = eos_color)
    plt.axvline(x = Tc_cool, linestyle = coo_style, linewidth = 1.5, color = coo_color)
    int_data = interpolator(eos_data, Tlin, Plin,vp_eos , Tc_eos, Pc_eos , Tsat_eos,P_fit,method="nearest")
    error_ = (eos_data-int_data)*100/eos_data

    for i in indices
        Isobars = error_[:,i]
        if Plin[i]<=Pc_eos
    """
    #        plt.plot(Tlin, Isobars, label="$(Plin[i])",color="b")
    #    elseif Plin[i]>Pc_eos
    #        plt.plot(Tlin, Isobars, label="$(Plin[i])",color="r")
    #    end
    #end
    #plt.legend()
    #plt.savefig(joinpath(mat_dir, "$(CES) $(subs) $(Name)vs_T.png"), dpi=1500)
    #plt.close()
    
end

CESs = ["cPR","ADPCSAFT", "BACKSAFT", "Berthelot", "CKSAFT", "Clausius", "CPA", "CPPCSAFT", "PR","DAPT", "EPPR78", "GEPCSAFT" , "GEPCSAFT" , "HeterogcPCPSAFT", "HomogcPCPSAFT", "iPCSAFT", "KU", "LJSAFT","ogSAFT", "PatelTeja", "PCPSAFT", "PCSAFT", "pharmaPCSAFT", "PR78","PSRK", "PTV", "QCPR", "OPCSAFT", "RK", "RKPR","SAFTgammaMie","SAFTVRMie", "SAFTVRMie15", "SAFTVRQMie", "SAFTVRSMie", "SAFTVRSW", "sCKSAFT","sCPA", "softSAFT2016","sPCSAFT", "SRK", "structSAFTgammaMie", "tcPR", "tcRK", "TVTPR", "gcsPCSAFT","TWUSRK", "UMRPR", "vdW", "VTPR"] 
#CESs = ["iPCSAFT","RK","SRK","tcRK","PSRK","PR","PR78","cPR","tcPR","tcPRW","QCPR","VTPR","PatelTeja","PTV","PCSAFT","PCPSAFT","ADPCSAFT","SAFTVRMie","SAFTVRQMie","DAPT"]

compounds = ["n-Nonane", "MethylLinolenate", "DimethylCarbonate", "R21", "DiethylEther", "trans-2-Butene", "R245fa", "ParaDeuterium", "OrthoDeuterium", "Isohexane", "R365MFC", "n-Dodecane", "R410A", "Deuterium", "D4", "R13", "MD2M", "n-Hexane", "Methane", "Ethane", "CarbonylSulfide", "EthylBenzene", "CarbonMonoxide", "Isopentane", "Xenon", "cis-2-Butene", "R152A", "Oxygen", "EthyleneOxide", "R1234ze(E)", "n-Octane", "R404A", "R236EA", "CycloHexane", "n-Heptane", "R22", "R113", "n-Pentane", "MethylLinoleate", "R11", "SulfurDioxide", "R23", "Helium", "R32", "R227EA", "R407C", "HydrogenSulfide", "Air", "R245ca", "Novec649", "R143a", "D5", "R507A", "R134a", "Dichloroethane", "ParaHydrogen", "R1233zd(E)", "Acetone", "n-Decane", "HeavyWater", "MethylPalmitate", "n-Propane", "R115", "R1234yf", "R236FA", "Ethylene", "R116", "MD4M", "Benzene", "Methanol", "SulfurHexafluoride", "o-Xylene", "R125", "Fluorine", "R1234ze(Z)", "CarbonDioxide", "IsoButane", "n-Butane", "NitrousOxide", "DimethylEther", "RC318", "Toluene", "IsoButene", "MethylStearate", "Ammonia", "Argon", "R218", "R41", "Neon", "Propyne", "CycloPropane", "R12", "Nitrogen", "Water", "MethylOleate", "R161", "D6", "SES36", "HFE143m", "n-Undecane", "R123", "HydrogenChloride", "m-Xylene", "R141b", "R124", "1-Butene", "Propylene", "R14", "p-Xylene", "Cyclopentane", "MDM", "Hydrogen", "Neopentane", "Ethanol", "OrthoHydrogen", "R114", "Krypton", "MD3M", "R1243zf", "MM", "R142b", "R40", "R13I1"]


function graph(args...; kwargs...)
    graph_(args...; kwargs...)
end



function process_comp(comp,EOS)

    for compound in [comp]
  
        ces = EOS
        T_shift = 0.0
        failed_loading = false
    
        mat_dir_cool = joinpath(Master_folder, "NPZ_files", "CoolProp", "CoolProp_$(comp)")
        mat_dir_eos = joinpath(Master_folder, "NPZ_files", EOS, "$(EOS)_$(comp)")
        # <-- lines that were missing -->
        cp_files = [
            joinpath(mat_dir_cool, "Coolprop $compound density.mat"),
            joinpath(mat_dir_cool, "Coolprop $compound residual entropy.mat"),
            joinpath(mat_dir_cool, "Coolprop $compound fugacity coefficient.mat"),
            joinpath(mat_dir_cool, "Coolprop $compound Hv.mat"),
            joinpath(mat_dir_cool, "Coolprop $compound pv.mat")
        ]
        mat_data_cp = all(isfile, cp_files) ? [matread(f) for f in cp_files] : nothing
        #print("mat_data_cp = ",mat_data_cp)
        #mat_data_cp = nothing
        ces_files = [
            joinpath(mat_dir_eos, "$(EOS) $compound density.mat"),
            joinpath(mat_dir_eos, "$(EOS) $compound residual entropy.mat"),
            joinpath(mat_dir_eos, "$(EOS) $compound fugacity coefficient.mat"),
            joinpath(mat_dir_eos, "$(EOS) $compound Hv.mat"),
            joinpath(mat_dir_eos, "$(EOS) $compound pv.mat")
        ]
        #mat_data_ces = all(isfile, ces_files) ? [matread(f) for f in ces_files] : nothing
        mat_data_ces = nothing
               

        T, P, Tsat = nothing, nothing, nothing
        Rho_CP, Sres_CP, Hv_CP, pv_CP, Rho_sat_liq_CP, Rho_sat_vap_CP, Sres_sat_liq_CP, Sres_sat_vap_CP, Phi_CP, Phi_sat_liq_CP, Phi_sat_vap_CP, Tsat_cool, Tc_cool, Pc_cool = nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing
        Rho_CES, Sres_CES, Hv_CES, pv_CES, Rho_sat_liq_CES, Rho_sat_vap_CES, Sres_sat_liq_CES, Sres_sat_vap_CES, Phi_CES, Phi_sat_liq_CES, Phi_sat_vap_CES = nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing
        
        
        if mat_data_cp === nothing
            fail = true
            print("we are here")
            while fail
                try
                    T, P, Rho_CP, Sres_CP, Hv_CP, pv_CP, Rho_sat_liq_CP, Rho_sat_vap_CP, Sres_sat_liq_CP, Sres_sat_vap_CP, Phi_CP, Phi_sat_liq_CP, Phi_sat_vap_CP, Tsat_cool, Tc_cool, Pc_cool = density_CP(compound; T_shift = T_shift)
                    fail = false
                catch
                    T_shift += 1
                end
            end
        else
            T, P, Rho_CP = mat_data_cp[1]["T"], mat_data_cp[1]["P"], mat_data_cp[1]["Rho_CP"]
            Sres_CP = mat_data_cp[2]["Sres_CP"]
            Hv_CP = mat_data_cp[4]["Hv_CP"]
            pv_CP = mat_data_cp[5]["pv_CP"]
            Rho_sat_liq_CP = mat_data_cp[1]["Rho_sat_liq"]
            Rho_sat_vap_CP = mat_data_cp[1]["Rho_sat_vap"]
            Sres_sat_liq_CP = mat_data_cp[2]["Sres_sat_liq_CP"]
            Sres_sat_vap_CP = mat_data_cp[2]["Sres_sat_vap_CP"]
            Phi_CP = mat_data_cp[3]["Phi_CP"]
            Phi_sat_liq_CP = mat_data_cp[3]["Phi_sat_liq_CP"]
            Phi_sat_vap_CP = mat_data_cp[3]["Phi_sat_vap_CP"]
            Tsat_cool = mat_data_cp[4]["Tsat"]
            T_shift = mat_data_cp[4]["T_shift"]
            Tc_cool = mat_data_cp[1]["Tc"]
            Pc_cool = mat_data_cp[1]["Pc"]
            #println(T_shift)




        end

        if mat_data_ces === nothing
            try
                println("Calculating CES")
                T, P, Rho_CES, Sres_CES, Hv_CES, pv_CES, Rho_sat_liq_CES, Rho_sat_vap_CES, Sres_sat_liq_CES, Sres_sat_vap_CES, Phi_CES, Phi_sat_liq_CES, Phi_sat_vap_CES, Tsat_ces, Tc_ces, Pc_ces  = density_CES(compound, ces; T_shift = T_shift)
                println("Done calculating CES")
            catch e
                println("CES Failed:")
                showerror(stdout, e)
                println("\nStacktrace:")
                display(stacktrace(catch_backtrace()))
                failed_loading = true
            end
        else
            Rho_CES = mat_data_ces[1]["Rho_CES"]
            Sres_CES = mat_data_ces[2]["Sres_CES"]
            Hv_CES = mat_data_ces[4]["Hv_CES"]
            pv_CES = mat_data_ces[5]["pv_CES"]
            Rho_sat_liq_CES = mat_data_ces[1]["Rho_sat_liq"]
            Rho_sat_vap_CES = mat_data_ces[1]["Rho_sat_vap"]
            Sres_sat_liq_CES = mat_data_ces[2]["Sres_sat_liq_CES"]
            Sres_sat_vap_CES = mat_data_ces[2]["Sres_sat_vap_CES"]
            Phi_CES = mat_data_ces[3]["Phi_CES"]
            Phi_sat_liq_CES = mat_data_ces[3]["Phi_sat_liq_CES"]
            Phi_sat_vap_CES = mat_data_ces[3]["Phi_sat_vap_CES"]
            Tsat_ces = mat_data_ces[4]["Tsat"]
            Tc_ces = mat_data_ces[1]["Tc"]
            Pc_ces = mat_data_ces[1]["Pc"]
            #println("Tsat = ",length(Tsat_ces))
            #println("pv_CES = ",length(pv_CES))
        end

        if !failed_loading
            mat_dir = joinpath(Master_folder, "Figures", EOS, "$(EOS)_$(comp)")
            ensure_directory_exists(mat_dir)

            println("Starting to calculate the graphs")
            graph(EOS, "Residual molar entropy", compound, Sres_CES, Sres_CP, Sres_sat_liq_CP, Sres_sat_vap_CP, Sres_sat_liq_CES, Sres_sat_vap_CES,  pv_CP,pv_CES,Tsat_cool,Tsat_ces,Tc_cool,Tc_ces,Pc_cool,Pc_ces, T, P)

            graph(EOS, "Density", compound, Rho_CES, Rho_CP, Rho_sat_liq_CP, Rho_sat_vap_CP, Rho_sat_liq_CES, Rho_sat_vap_CES, pv_CP,pv_CES,Tsat_cool,Tsat_ces,Tc_cool,Tc_ces,Pc_cool,Pc_ces, T, P)

            graph(EOS, "Fugacity Coefficient", compound, Phi_CES, Phi_CP, Phi_sat_liq_CP, Phi_sat_vap_CP, Phi_sat_liq_CES, Phi_sat_vap_CES, pv_CP,pv_CES,Tsat_cool,Tsat_ces,Tc_cool,Tc_ces,Pc_cool,Pc_ces, T, P)
            println("Done with calculating the graphs")

            if Tc_ces > Tc_cool
                Tsat = Tsat_cool
                pv_EXP = pv_CP
                pv_EOS = pv_CES[Tsat_ces .<Tc_cool]
                Hv_EXP = Hv_CP
                Hv_EOS = Hv_CES[Tsat_ces .<Tc_cool]

            elseif Tc_ces < Tc_cool
                Tsat = Tsat_ces
                pv_EXP = pv_CP[Tsat_cool .<=Tc_ces]
                pv_EOS = pv_CES
                Hv_EXP = Hv_CP[Tsat_cool .<=Tc_ces]
                Hv_EOS = Hv_CES

            elseif Tc_ces == Tc_cool
                Tsat = Tsat_cool
                pv_EXP = pv_CP
                pv_EOS = pv_CES
                Hv_EXP = Hv_CP
                Hv_EOS = Hv_CES
            end
           

            #println("Tsat = ",Tsat)
            #println("pv_CES = ",pv_CES)
            #println("pv_EXP = ",pv_EXP)
            #println("pv_error = ",abs.(pv_EXP .- pv_EOS) .* 100 ./ pv_EXP)
            plt = plot(Tsat, abs.(pv_EXP .- pv_EOS) .* 100 ./ pv_EXP, xlabel = "Temperature [K]", ylabel = "Pressure Error", title = "Vapor Pressure $compound", xlims = (minimum(Tsat), maximum(Tsat)))
            savefig(plt, joinpath(mat_dir, "Vapor Pressure $(EOS) $compound.png"))
        
            plt = plot(Tsat, abs.(Hv_EXP .- Hv_EOS) .* 100 ./ Hv_EXP, xlabel = "Temperature [K]", ylabel = "Enthalpy Error", title = "Enthalpy $compound", xlims = (minimum(Tsat), maximum(Tsat)))
            savefig(plt, joinpath(mat_dir, "Enthalpy $(EOS) $compound.png"))
        end
    end
end

compounds = ["n-Nonane", "MethylLinolenate", "DimethylCarbonate", "R21", "DiethylEther", "trans-2-Butene", "R245fa", "ParaDeuterium", "OrthoDeuterium", "Isohexane", "R365MFC", "n-Dodecane", "R410A", "Deuterium", "D4", "R13", "MD2M", "n-Hexane", "Methane", "Ethane", "CarbonylSulfide", "EthylBenzene", "CarbonMonoxide", "Isopentane", "Xenon", "cis-2-Butene", "R152A", "Oxygen", "EthyleneOxide", "R1234ze(E)", "n-Octane", "R404A", "R236EA", "CycloHexane", "n-Heptane", "R22", "R113", "n-Pentane", "MethylLinoleate", "R11", "SulfurDioxide", "R23", "Helium", "R32", "R227EA", "R407C", "HydrogenSulfide", "Air", "R245ca", "Novec649", "R143a", "D5", "R507A", "R134a", "Dichloroethane", "ParaHydrogen", "R1233zd(E)", "Acetone", "n-Decane", "HeavyWater", "MethylPalmitate", "n-Propane", "R115", "R1234yf", "R236FA", "Ethylene", "R116", "MD4M", "Benzene", "Methanol", "SulfurHexafluoride", "o-Xylene", "R125", "Fluorine", "R1234ze(Z)", "CarbonDioxide", "IsoButane", "n-Butane", "NitrousOxide", "DimethylEther", "RC318", "Toluene", "IsoButene", "MethylStearate", "Ammonia", "Argon", "R218", "R41", "Neon", "Propyne", "CycloPropane", "R12", "Nitrogen", "Water", "MethylOleate", "R161", "D6", "SES36", "HFE143m", "n-Undecane", "R123", "HydrogenChloride", "m-Xylene", "R141b", "R124", "1-Butene", "Propylene", "R14", "p-Xylene", "Cyclopentane", "MDM", "Hydrogen", "Neopentane", "Ethanol", "OrthoHydrogen", "R114", "Krypton", "MD3M", "R1243zf", "MM", "R142b", "R40", "R13I1"]

CESs = ["cPR","ADPCSAFT", "BACKSAFT", "Berthelot", "CKSAFT", "Clausius", "CPA", "CPPCSAFT", "PR","DAPT", "EPPR78", "GEPCSAFT" , "GEPCSAFT" , "HeterogcPCPSAFT", "HomogcPCPSAFT", "iPCSAFT", "KU", "LJSAFT","ogSAFT", "PatelTeja", "PCPSAFT", "PCSAFT", "pharmaPCSAFT", "PR78","PSRK", "PTV", "QCPR", "OPCSAFT", "RK", "RKPR","SAFTgammaMie","SAFTVRMie", "SAFTVRMie15", "SAFTVRQMie", "SAFTVRSMie", "SAFTVRSW", "sCKSAFT","sCPA", "softSAFT2016","sPCSAFT", "SRK", "structSAFTgammaMie", "tcPR", "tcPRW" ,"tcRK", "TVTPR", "gcsPCSAFT","TWUSRK", "UMRPR", "vdW", "VTPR"] 
CESs_reversed = [
    "VTPR","vdW","UMRPR","TWUSRK","gcsPCSAFT","TVTPR","tcRK","tcPRW","tcPR",
    "structSAFTgammaMie","SRK","sPCSAFT","softSAFT2016","sCPA","sCKSAFT",
    "SAFTVRSW","SAFTVRSMie","SAFTVRQMie","SAFTVRMie15","SAFTVRMie","SAFTgammaMie",
    "RKPR","RK","OPCSAFT","QCPR","PTV","PSRK","PR78","pharmaPCSAFT","PCSAFT",
    "PCPSAFT","PatelTeja","ogSAFT","LJSAFT","KU","iPCSAFT","HomogcPCPSAFT",
    "HeterogcPCPSAFT","GEPCSAFT","GEPCSAFT","EPPR78","DAPT","PR","CPPCSAFT",
    "CPA","Clausius","CKSAFT","Berthelot","BACKSAFT","ADPCSAFT","cPR"
]

CESs_midfirst = [
    "PSRK","PR78","pharmaPCSAFT","PCSAFT","PCPSAFT","PatelTeja","ogSAFT","LJSAFT",
    "KU","iPCSAFT","HomogcPCPSAFT","HeterogcPCPSAFT","GEPCSAFT","GEPCSAFT","EPPR78",	
    "DAPT","PR","CPPCSAFT","CPA","Clausius","CKSAFT","Berthelot","BACKSAFT",
    "ADPCSAFT","cPR","VTPR","vdW","UMRPR","TWUSRK","gcsPCSAFT","TVTPR","tcRK",
    "tcPRW","tcPR","structSAFTgammaMie","SRK","sPCSAFT","softSAFT2016","sCPA",
    "sCKSAFT","SAFTVRSW","SAFTVRSMie","SAFTVRQMie","SAFTVRMie15","SAFTVRMie",
    "SAFTgammaMie","RKPR","RK","OPCSAFT","QCPR","PTV"
]
for EOS in [ARGS[2]]
    for subs in [ARGS[1]]
        path = joinpath(Master_folder, "NPZ_files", EOS, "$(EOS)_$(subs)")
        #if !ispath(path)
        if true
            #model = eval(Meta.parse("$(EOS)([\"$(subs)\"])"))
            try
                println(path)
                println("$(EOS) with $(subs)")
                process_comp(subs,EOS)
            catch e
                
                println("Python Error:")
                showerror(stdout, e)
                println("\nStacktrace:")
                display(stacktrace(catch_backtrace()))
                continue
            end
        end
    end
end
