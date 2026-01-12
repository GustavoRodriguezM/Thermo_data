using Clapeyron, PyCall, Plots, MAT, JSON, Printf
using Base.Filesystem
using Statistics
using PyCall: PyError


traceback = pyimport("traceback")
include("iPCSAFT.jl")
include("TVTPR.jl")
include("VTPR.jl")
include("tcPRC.jl")
include("tcRK.jl")
include("tcPR.jl")
include("cPR.jl")

bisect = pyimport("bisect").bisect
optimize = pyimport("scipy.optimize")
CoolProp = pyimport("CoolProp")
scipy = pyimport("scipy")
matplotlib = pyimport("matplotlib")
plt = pyimport("matplotlib.pyplot")
np = pyimport("numpy")
curve_fit = pyimport("scipy.optimize").curve_fit
least_squares = pyimport("scipy.optimize").least_squares
pe = pyimport("matplotlib.patheffects")
cm = matplotlib.cm
colors = pyimport("matplotlib.colors")
cons = pyimport("scipy.constants")
sp_interp = pyimport("scipy.interpolate")
CESs = ["tcPRC","cPR","ADPCSAFT", "BACKSAFT", "Berthelot", "CKSAFT", "Clausius", "CPA", "CPPCSAFT", "PR","DAPT", "EPPR78", "GEPCSAFT" , "GEPCSAFT" , "HeterogcPCPSAFT", "HomogcPCPSAFT", "iPCSAFT", "KU", "LJSAFT","ogSAFT", "PatelTeja", "PCPSAFT", "PCSAFT", "pharmaPCSAFT", "PR78","PSRK", "PTV", "QCPR", "OPCSAFT", "RK", "RKPR","SAFTgammaMie","SAFTVRMie", "SAFTVRMie15", "SAFTVRQMie", "SAFTVRSMie", "SAFTVRSW", "sCKSAFT","sCPA", "softSAFT2016","sPCSAFT", "SRK", "structSAFTgammaMie", "tcPR", "tcPRW" ,"tcRK", "TVTPR", "gcsPCSAFT","TWUSRK", "UMRPR", "vdW", "VTPR"] 
nan_index = ([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12], [179, 180, 181, 182, 183, 179, 180, 181, 182, 183, 180, 181, 182, 183, 184, 180, 181, 182, 183, 184, 181, 182, 183, 184, 181, 182, 183, 184, 185, 181, 182, 183, 184, 185, 182, 183, 184, 185, 186, 182, 183, 184, 185, 186, 182, 183, 184, 185, 186, 183, 184, 185, 186, 187, 183, 184, 185, 186, 187, 183, 184, 185, 186, 187, 188])


SaftVR = read("Saft_VR_mie.json", String)
SaftVRp = JSON.parse(SaftVR)

CIDD = read("CIDs.json", String)
CID = JSON.parse(CIDD)

# Determine the master folder path
#Master_folder = joinpath(@__DIR__,"0.6.10")
Master_folder = @__DIR__

point_distribution = "log"
setprecision(512)
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
    println("param_values = $(param_values)")
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

function adjuster__(x,y,fun,N,method)
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



function adjuster_(x, y, fun, N, method="trf")
    params = []
    AAD = 1e7

    for num_param in 1:N
        guess = ones(num_param)
        if length(params) > 0
            guess[1:length(params)] .= params
        end
        # residuals to minimize
        function residuals(p)
            return fun(x, p...) - y
        end
        res = least_squares(residuals, guess, method=method, max_nfev=10000)
        params = res["x"]
        pred = fun(x, params...)
        AAD = np.mean(np.abs(pred - y)*100/y)
    end
    return params, AAD
end


function adjuster(x,y,fun,N)
    return adjuster_(x,y,fun,N,"trf")
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
    
    
    params,AAD = adjuster(T,P_sat,wrapper,4)
    plt.plot(T, wrapper(T, params...), "k--", label="p_sat fit")
    Psat_fun(T) = wrapper(T, params...)
    test = Psat_fun(Tc)
    if test == 0 || !isfinite(test)
        throw("Wrong Wagner adjust")
    end
    
    plt.savefig(joinpath(mat_dir, "vp fits.png"), dpi=600)
    plt.close()

    return Psat_fun,AAD
end

function fit_densities(Tc, rho_c, T, rho_sat_vapor, rho_sat_liquid, CES, subs; print_AAD=false, plot=false)

    f_liq,f_vap,liquid_params,vapor_params = nothing, nothing, nothing, nothing

    wrap_liq(T, p...) = rho_liquid(T, Tc, rho_c, p)
    wrap_vap(T, p...) = rho_vapor(T, Tc, rho_c, p)


    mask_below_Tc = T .< Tc
    mask_T_min = T .> Tc * 0
    is_not_nan = np.isfinite(rho_sat_liquid) .& np.isfinite(rho_sat_vapor)

    T = T[mask_below_Tc][mask_T_min][is_not_nan]
    rho_sat_liquid = rho_sat_liquid[mask_below_Tc][mask_T_min][is_not_nan]
    rho_sat_vapor = rho_sat_vapor[mask_below_Tc][mask_T_min][is_not_nan]
    mat_dir = joinpath(Master_folder, "Figures", CES, "$(CES)_$(subs)")


    
    liquid_params,AAD_liq = adjuster(T,rho_sat_liquid,wrap_liq,4)
    f_liq = T -> wrap_liq(T, liquid_params...)


    vapor_params,AAD_vap = adjuster(T,rho_sat_vapor,wrap_vap,5)
    f_vap = T -> wrap_vap(T, vapor_params...)

    

    if plot
        plt.plot(T,rho_sat_liquid,  "g--", label="liquid")
        plt.plot(T,rho_sat_vapor, "b--", label="vapor")
        plt.plot(T,wrap_liq(T, liquid_params...), "g", label="liquid fit")
        plt.plot(T,wrap_vap(T, vapor_params...), "b", label="vapor fit")
        plt.legend()
        plt.savefig(joinpath(mat_dir, "Density fits.png"))
        plt.close()
    end

    return f_liq, f_vap, AAD_liq, AAD_vap
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
        try
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
                H = 1
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
                bondvol = Dict(((Name, "e"), (Name, "H")) => bondvol_val * 10^-30)))
        catch
            model = eval(Meta.parse("$(CES)([\"$(Name)\"])"))
        end
    elseif CES in ["cPR","tcRK","tcPR","sCPA","iPCSAFT","HomogcPCPSAFT","BACKSAFT","PCPSAFT","SAFTVRSW","pharmaPCSAFT","SAFTgammaMie","sPCSAFT", "CKSAFT","softSAFT2016","PCSAFT","GEPCSAFT","HeterogcPCPSAFT", "ogSAFT"]
        model = eval(Meta.parse("$(CES)([\"$(Name)\"])"))
    
    elseif CES=="VTPR" 
        bef = PR(Name,alpha = TwuAlpha)
        
        handle = CoolProp.AbstractState("HEOS", Name)
        Tc = CoolProp.AbstractState.T_critical(handle)
        Pc = CoolProp.AbstractState.p_critical(handle)
        R = cons.R 
        Vc = 1 / CoolProp.AbstractState.rhomolar_critical(handle)
        Zc = Pc*Vc/(R*Tc)
        c_cor =  - 0.252*R*Tc/Pc*(1.5448Zc - 0.4024)        
        

        handle.update(CoolProp.QT_INPUTS, 0, 0.7 * Tc)
        vexp = 1 / CoolProp.CoolProp.AbstractState.rhomolar(handle)

        handle.update(CoolProp.QT_INPUTS, 1, 0.7 * Tc)
        
        dnd = tcPR([Name])
        
        (pv, vcalc, vv) = saturation_pressure(bef, 0.7 * Tc, ChemPotVSaturation(vl = vexp, vv = 1 / CoolProp.CoolProp.AbstractState.rhomolar(handle) ))

        println("vexp = $(vexp),  vcalc = $(vcalc)")
        c = -(vexp - vcalc)    
        println("c experimental = $(c), c correlation = $(c_cor), c tcPR = $(dnd.translation.params.v_shift[1])")
        translation = ConstantTranslation([Name];userlocations = (;v_shift = [c]))
        model = PR([Name]; alpha=TwuAlpha, translation=translation, userlocations = (;Tc = [Tc],Pc = [Pc], Mw = [dnd.params.Mw[1]]))
    else
        handle = CoolProp.AbstractState("HEOS", Name)
        Tc = CoolProp.AbstractState.T_critical(handle)
	    Pc = CoolProp.AbstractState.p_critical(handle)
        acentricfactor = CoolProp.AbstractState.acentric_factor(handle)
        Vc = 1/CoolProp.AbstractState.rhomolar_critical(handle)
        par = Dict("Tc" => Tc, "Pc" => Pc, "acentricfactor" => acentricfactor, "Vc" => Vc )
        user = "userlocations = (;"
        model = eval(Meta.parse("$(CES)([\"$(Name)\"])"))

        for (i,name) in enumerate(fieldnames(typeof(model.params)))
            param = getfield(model.params, name)
            if !(string(name) in ["Tc","Pc","acentricfactor","Vc","epsilon_assoc","bondvol"])
                add = "$(string(name)) = [$(param.values[1])]"
            elseif string(name) in ["epsilon_assoc","bondvol"]
                try
                    add = "$(string(name)) = Dict(((\"$(Name)\", \"e\"), (\"$(Name)\", \"H\")) => $(param.values[1])) "
                catch
                    continue
                end
            else 
                add = "$(string(name)) = [$(par[string(name)])]"
            end
            if i == length(fieldnames(typeof(model.params)))
                last = ""
            else
                last = ", "
            end
            user = user * add * last
        end
        if CES=="PR" && Name == "MethylLinoleate"
            user = user * ", Vc = [$(Vc)]"
        end
        user = user * " )"
        println(user)
        model = eval(Meta.parse("$(CES)([\"$(Name)\"];"* user* ")"))

    end
    return model
end



function limit_creator(handle, point_distribution, N, T_shift,compound)

    pc = CoolProp.AbstractState.p_critical(handle)
    Tc = CoolProp.AbstractState.T_critical(handle)  # <- critical temperature

    
    Tmin = CoolProp.AbstractState.Tmin(handle) + T_shift

    println("Tmin bef = $(Tmin)")
    if compound=="Water"
        T_shift = T_shift + 3
    elseif compound=="Methanol"
        T_shift = T_shift + 19
    end
    Tmax = CoolProp.AbstractState.Tmax(handle)
    Tmin = CoolProp.AbstractState.Tmin(handle) + T_shift
    
    

    pmin = pc/1000
    pmax = CoolProp.AbstractState.pmax(handle)
    if compound=="Nitrogen"
        Tmin = 66.151
        pmax = 9.58051424301807e8
    elseif compound=="Neon"
        Tmin = 26.56
        pmax = 7.731780478418967e8
    elseif compound=="Hydrogen"
        Tmin = 15.957
        pmax = 2.7882549791824746e8
    elseif compound=="Helium"
        Tmin = 4.1768
        pmax = 4.389640024673441e7
    end
    println("Tmin aft = $(Tmin)")
    T = LinRange(Tmin, Tmax, N)
    if point_distribution == "linear"
        P = collect(LinRange(pmin, pmax, N))
    elseif point_distribution == "log"
        P = exp10.(LinRange(log10(pmin), log10(pmax), N))
    else
        error("Unknown point distribution")
    end

    return collect(T), P
end



using Printf

function same_sign_digits(nums::Vector{Float64}; n::Int=6)
    # Return true if fewer than 2 numbers (trivial case)
    if length(nums) < 2
        return true
    end

    # Build the format string dynamically, e.g., "%.5e"
    fmt = "%." * string(n-1) * "e"

    # Reference mantissa from the first number
    sa = Printf.format(Printf.Format(fmt), nums[1])
    mant_ref = split(sa, 'e')[1]

    # Compare all others
    for x in nums[2:end]
        if isnan(x) || (x == 0 && nums[1] != 0)
            return false
        end
        sx = Printf.format(Printf.Format(fmt), x)
        mant_x = split(sx, 'e')[1]
        if mant_x != mant_ref
            return false
        end
    end

    return true
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


    nan_index_liq = Set(np.where(np.isnan(Rho_sat_liq_CES))[1])
    nan_index_vap = Set(np.where(np.isnan(Rho_sat_vap_CES))[1])
    nan_index_vp = Set(np.where(np.isnan(pv_CES))[1])

    nan_index = union(nan_index_liq, nan_index_vap, nan_index_vp)
    nan_index = Set(x + 1 for x in nan_index)
    P_fit,liq, vap = nothing, nothing, nothing
    println(nan_index,isempty(nan_index))
    if !isempty(nan_index)
        P_fit, AARD_vp = fit_vapor_pressure(Tc, Pc, Tsat, pv_CES, CES, compound; print_AAD=false, plot=true)
        liq, vap, AARD_liq, AARD_vap = fit_densities(Tc, 1/Vc, Tsat, Rho_sat_vap_CES, Rho_sat_liq_CES, CES, compound; print_AAD=false, plot=true) 
    else
        return Rho_sat_liq_CES,Rho_sat_vap_CES,Phi_sat_liq_CES,Phi_sat_vap_CES,Sres_sat_liq_CES,Sres_sat_vap_CES,Hv_CES,pv_CES,P_fit,liq, vap
    end
    for ind in nan_index
        t = Tsat[ind]
        #(pv, vl, vv) = saturation_pressure(model, t, ChemPotDensitySaturation( vl = liq(t), vv = vap(t)))
        #(pv, _, __) = saturation_pressure(model, t, IsoFugacitySaturation(p0 = P_fit(t), vl = liq(t), vv = vap(t) ))

        vv,vl = NaN,NaN
        if AARD_vap<=0.1
            vv = vap(t)
        end

        if AARD_liq<=0.1 
            vl = liq(t)
        end

        if AARD_vp<=0.1
            pv = P_fit(t)
        else
            (pv, _, __) = saturation_pressure(model, t, IsoFugacitySaturation(p0 = P_fit(t), vl = liq(t), vv = vap(t) ))
        end
        Rho_sat_liq_CES[ind] = 1 / vl
        Rho_sat_vap_CES[ind] = 1 / vv
        Phi_sat_liq_CES[ind] = Clapeyron.VT_fugacity_coefficient(model, vl, t, [1.])[1]
        Phi_sat_vap_CES[ind] = Clapeyron.VT_fugacity_coefficient(model, vv, t, [1.])[1]
        Sres_sat_liq_CES[ind] = Clapeyron.VT_entropy_res(model, vl, t, [1.])
        Sres_sat_vap_CES[ind] = Clapeyron.VT_entropy_res(model, vv, t, [1.])
        hv = Clapeyron.VT_enthalpy(model, vv, t, [1.])
        hl = Clapeyron.VT_enthalpy(model, vl, t, [1.])
        Hv_CES[ind] = hv - hl
        pv_CES[ind] = pv
    end

    return Rho_sat_liq_CES,Rho_sat_vap_CES,Phi_sat_liq_CES,Phi_sat_vap_CES,Sres_sat_liq_CES,Sres_sat_vap_CES,Hv_CES,pv_CES,P_fit,liq, vap
end



function coolprop_int(handle,T,P,Matrix,type,range_left,range_right,Number,cp_cl)

    for i in 1:500
        rho_line = Matrix[:,i]

        if !np.any(np.isnan(rho_line))
            continue
        end
        nan_indexes = np.where(np.isnan(rho_line))[1]

        for nan_i in nan_indexes

            nan_ind = nan_i + 1
            Tnan = T[nan_ind]

            if type=="density"
                low = rho_line[min(nan_ind+5,500)]
                high = rho_line[max(1,nan_ind-5)]
            elseif type=="entropy"
                low = rho_line[max(1,nan_ind-5)]
                high = rho_line[min(500,nan_ind+5)]
            elseif type=="fugacity"
                low = 0
                high = 1
            end
            
            T_x = collect(LinRange(T[max(nan_ind-range_left,1)], T[min(nan_ind+range_right,500)], Number))
            rho_y = similar(T_x)

            for j in 1:length(T_x)
                
                try
                    if cp_cl=="coolprop"
                        handle.update(CoolProp.PT_INPUTS,P[i],T_x[j])
                        if type=="density"
                            den = CoolProp.CoolProp.AbstractState.rhomolar(handle)
                        elseif type=="entropy"
                            den = CoolProp.CoolProp.AbstractState.smolar_residual(handle)
                        elseif type=="fugacity"
                            den = CoolProp.CoolProp.AbstractState.fugacity_coefficient(handle, 0)
                        end
                    elseif cp_cl=="clapeyron"

                        if type=="density"
                            den = 1 / volume(handle,P[i],T_x[j])
                        elseif type=="entropy"
                            dens = 1 / volume(handle,P[i],T_x[j])
                            den = Clapeyron.VT_entropy_res(handle, BigFloat(1 / dens), T_x[j], [1.])
                        elseif type=="fugacity"
                            dens = 1 / volume(handle,P[i],T_x[j])
                            den = Clapeyron.VT_fugacity_coefficient(handle, BigFloat(1 / dens), T_x[j], [1.])[1]

                        end
                    end
                    rho_y[j] = den
                catch e
                    rho_y[j] = NaN
                end
            end
            low_mask = low.<rho_y
            high_mask = high.>rho_y
            mask_rho = .!isnan.(rho_y) .& low_mask .& high_mask
            T_rho = T_x[mask_rho]

            rho_clean = rho_y[mask_rho]
            println("rho_clean = $(size(rho_clean)),  $(size(T_rho)), P[i]")
            rho_spl = sp_interp.make_interp_spline(T_rho, rho_clean, k=1)

            rho_def = rho_spl(Tnan)
            Matrix[nan_ind,i] = rho_def[1]

        end
    end
    return Matrix
end




function interpolator_filler(model,Rho_CES,Sres_CES, Phi_CES,T, P) 
    nan_index = np.where(np.isnan(Rho_CES)) 
    Delta_T = T[2] - T[1] 
    Delta_P = P[2] - P[1] 
    println(Delta_T) 
    println(Delta_P) 
    rotator = Dict("n" => "e", "e" =>"s","s"=>"w","w" =>"n") 

    println("nan index = $(nan_index)") 
    for (t_id,p_id) in zip(nan_index...) 
        t_idx=t_id+1 
        p_idx=p_id+1 

        println(t_idx,"   ",p_idx)
        t = T[t_idx] 
        p = P[p_idx] 
        x0 = Rho_CES[t_idx,p_idx] 
        println("x0 = $(x0)") 
        if t_idx==1 && p_idx<500
            nor = (0,1)
            sou = (1,1)
            wes = (2,1)
            eas = (1,0)
            println("western face")
        elseif p_idx == 500 && t_idx < 500
            nor = (1,-1)
            sou = (0,-1)
            wes = (1,-2)
            eas = (1,0)
            println("north face")
        elseif t_idx == 500 && p_idx > 1
            nor = (-1,-1)
            sou = (0,-1)
            wes = (-1,0)
            eas = (-2,-1)
            println("eastern face")
        elseif p_idx == 1 && t_idx > 1
            nor = (0,1)
            sou = (-1,1)
            wes = (-1,0)
            eas = (-1,2)
            println("soutern face")
        else
            nor = (0,1)
            sou = (0,-1)
            wes = (1,0)
            eas = (-1,0)
        end

        xo_n = 1/Rho_CES[t_idx + nor[1],p_idx + nor[2]] 
        xo_s = 1/Rho_CES[t_idx + sou[1],p_idx + sou[2] ] 
        xo_w = 1/Rho_CES[t_idx + wes[1],p_idx + wes[2]] 
        xo_e = 1/Rho_CES[t_idx + eas[1],p_idx + eas[2]] 
        N = 1000 
        X = Dict("n"=>xo_n, "s"=>xo_s, "e"=>xo_e, "w"=>xo_w) 
        V = Dict("n"=> fill(NaN, N+1), "s"=> fill(NaN, N+1), "e"=> fill(NaN, N+1), "w"=> fill(NaN, N+1)) 
        V["n"][1] = xo_n 
        V["s"][1] = xo_s
        V["w"][1] = xo_w
        V["e"][1] = xo_e 
        Vn,Vs,Vw,Ve = Nothing,Nothing,Nothing,Nothing 
        for i in 1:N 
            Vn = volume(model, p + nor[2] *Delta_P/2^i , t+ nor[1]* Delta_T/2^i ,vol0 = X["n"]) 
            Vs = volume(model, p + sou[2] *Delta_P/2^i , t+ sou[1]* Delta_T/2^i,vol0 = X["s"]) 
            Ve = volume(model, p + eas[2] *Delta_P/2^i , t+ eas[1]* Delta_T/2^i,vol0 = X["e"]) 
            Vw = volume(model, p + wes[2] *Delta_P/2^i , t+ wes[1]* Delta_T/2^i,vol0 = X["w"]) 
            V["n"][i+1] = Vn 
            V["s"][i+1] = Vs 
            V["w"][i+1] = Vw 
            V["e"][i+1] = Ve 

            g = Dict("n"=>Vn, "s"=>Vs, "e"=>Ve, "w"=>Vw)
            for d in ["n", "s", "e", "w"] 
                if !isnan(V[d][i]) 
                    X[d] = V[d][i] 
                else nd = d 
                    vals = Float64[] 
                    attempts = 0 
                    while isempty(vals) && attempts < 4 
                        nd = rotator[nd] 
                        vals = filter(!isnan, V[nd]) 
                        attempts += 1 
                    end 
                    if !isempty(vals) 
                        X[d] = last(vals) 
                    end 
                end 
            end 
            if all(x -> x == g["n"], values(g))
                println("Breaking early: all X values are identical at iteration $i")
                break
            end



            if i > 1
                no_change = all(d -> V[d][i] == V[d][i-1], ["n", "s", "e", "w"])
                if no_change
                    println("Breaking early at iteration $i: all directions stopped changing.")
                    break
                end
            end



        end 

        # Condition 1: all four current X values are the same
        println(V," ",t," ",p) 
        Vn_def, Vs_def, Ve_def, Vw_def = NaN, NaN, NaN, NaN
        try
            Vn_def = last(filter(!isnan, V["n"])) 
        catch
            Vn_def = NaN
        end
        try
            Vs_def = last(filter(!isnan, V["s"])) 
        catch
            Vs_def = NaN
        end
        try
            Ve_def = last(filter(!isnan, V["e"])) 
        catch
            Ve_def = NaN
        end
        try
            Vw_def = last(filter(!isnan, V["w"])) 
        catch
            Vw_def = NaN
        end
        

        V0 = (Vn_def + Vs_def + Ve_def + Vw_def)/4 
        Vdef = volume(model, p , t ,vol0 = V0) 

        vals = [Vn_def, Vs_def, Ve_def, Vw_def]

        for v in vals
            Vdef = volume(model, p , t ,vol0 = v)
            if !isnan(Vdef)
                break
            end
        end
        if same_sign_digits(vals)
            println("All four values agree up to 1e-6 precision. Using simple average only.")
            Vdef = mean(vals)
        end
        if isnan(Vdef)
            all_equal = all(v -> v == Vn_def, [Vs_def, Ve_def, Vw_def])
            if all_equal
                println("All four values identical and volume() returned NaN assigning directly.")
                Vdef = Vn_def
            else
                println("Using griddata interpolation with iteration-based coordinates.")

                # Determine the real positions from each iteration depth
                in = findlast(!isnan,V["n"]) - 1
                is = findlast(!isnan, V["s"]) - 1
                iw = findlast(!isnan, V["w"]) - 1
                ie = findlast(!isnan, V["e"]) - 1
                Tn = T[t_idx] +  nor[1] *Delta_T/2^in
                Ts = T[t_idx] +  sou[1] *Delta_T/2^is
                Tw = T[t_idx] +  wes[1] *Delta_T/2^iw 
                Te = T[t_idx] +  eas[1] *Delta_T/2^ie 
                Pn = P[t_idx] +  nor[2] *Delta_P/2^in
                Ps = P[t_idx] +  sou[2] *Delta_P/2^is
                Pw = P[t_idx] +  wes[2] *Delta_P/2^iw
                Pe = P[t_idx] +  eas[2] *Delta_P/2^ie
                known_grid = np.array([[Tn, Pn], [Ts, Ps], [Tw, Pw], [Te, Pe]]) 
                known_points = np.array([Vn_def, Vs_def, Vw_def, Ve_def]) 
                sought_point = np.array([[t, p]]) 
                Vdef_interp = sp_interp.griddata(known_grid, known_points, sought_point, method="cubic")
                if Vdef_interp === nothing || isnan(Vdef_interp[1])
                    println("Griddata failed, using simple average fallback.")
                    Vdef = V0
                else
                    Vdef = Vdef_interp[1]
                end
            end
        end

        # --- Assign interpolated results ---
        Rho_CES[t_idx, p_idx] = 1 / Vdef
        Sres_CES[t_idx, p_idx] = Clapeyron.VT_entropy_res(model, Vdef, t, [1.])
        Phi_CES[t_idx, p_idx] = Clapeyron.VT_fugacity_coefficient(model, Vdef, t, [1.])[1]
    end


    return Rho_CES, Sres_CES, Phi_CES 
end



function density_CES(compound, CES; T_shift = 0.0)

    N = 500

    handle = CoolProp.AbstractState("HEOS", compound)
    model = Initiator(CES, compound)

 
    T,P = limit_creator(handle,point_distribution,N,T_shift,compound)

    (Tc, Pc, Vc) = crit_pure(model)
    Tsat = T[T .< Tc] 
    
    println("Tc = $(Tc)")
    println("Pc = $(Pc)")
    Rho_CES, Phi_CES, Sres_CES, Rho_sat_liq_CES, Phi_sat_liq_CES, Sres_sat_liq_CES, Rho_sat_vap_CES, Phi_sat_vap_CES, Sres_sat_vap_CES, pv_CES, Hv_CES = fill(NaN, length(T), length(P)), zeros(Float64, length(T), length(P)), zeros(Float64, length(T), length(P)), zeros(length(Tsat)), zeros(length(Tsat)), zeros(length(Tsat)), zeros(length(Tsat)), zeros(length(Tsat)), zeros(length(Tsat)), zeros(length(Tsat)), zeros(length(Tsat))
    for (t_idx, t) in enumerate(Tsat)
        
        (pv, vl, vv) = saturation_pressure(model, BigFloat(t))
        if vl > vv
            vl, vv = vv, vl
        end

        hl = Clapeyron.VT_enthalpy(model, BigFloat(vl), BigFloat(t), [1.])
        hv = Clapeyron.VT_enthalpy(model, BigFloat(vv), BigFloat(t), [1.])

        Rho_sat_liq_CES[t_idx] = 1 / vl
        Rho_sat_vap_CES[t_idx] = 1 / vv
        Phi_sat_liq_CES[t_idx] = Clapeyron.VT_fugacity_coefficient(model, BigFloat(vl), BigFloat(t), [1.])[1]
        Phi_sat_vap_CES[t_idx] = Clapeyron.VT_fugacity_coefficient(model, BigFloat(vv), BigFloat(t), [1.])[1]
        Sres_sat_liq_CES[t_idx] = Clapeyron.VT_entropy_res(model, BigFloat(vl), BigFloat(t), [1.])
        Sres_sat_vap_CES[t_idx] = Clapeyron.VT_entropy_res(model, BigFloat(vv), BigFloat(t), [1.])
        Hv_CES[t_idx] = hv - hl
        pv_CES[t_idx] = pv
        
    end


    n_cut = Int(floor(0.4 * length(Tsat)))     
    idx_cut = 1:n_cut
    Rho_sat_vap_CES[idx_cut] .= NaN
    #pv_CES, Rho_sat_liq_CES, Rho_sat_vap_CES = neighbor_filler(Tsat, Rho_sat_liq_CES, Rho_sat_vap_CES, pv_CES, model)
    Rho_sat_liq_CES,Rho_sat_vap_CES,Phi_sat_liq_CES,Phi_sat_vap_CES,Sres_sat_liq_CES,Sres_sat_vap_CES,Hv_CES,pv_CES,P_fit,liq, vap = fitter_filler(model,Tsat,Rho_sat_liq_CES,Rho_sat_vap_CES,Phi_sat_liq_CES,Phi_sat_vap_CES,Sres_sat_liq_CES,Sres_sat_vap_CES,Hv_CES,pv_CES,Vc,Tc, Pc, CES, compound)
    println("pv_CES =  $(pv_CES)")
    for (t_idx, t) in enumerate(T)
        #println("Pc = ",Pc)
        for (p_idx, p) in enumerate(P)
            #println("p_idx = ",p_idx," p = ",p)
            if p < Pc && t < Tc
                if p < pv_CES[t_idx]
                    density_value = 1 / volume(model, BigFloat(p), BigFloat(t); phase = :vapor) 
                else
                    density_value = 1 / volume(model, BigFloat(p), BigFloat(t); phase = :liquid)
                end
            else
                density_value = 1 / volume(model, BigFloat(p), BigFloat(t))
            end

            if isnan(density_value)
                try
                    density_value = 1 / volume(model, BigFloat(p), BigFloat(t),vol0 = BigFloat(1 /Rho_CES[t_idx, p_idx-1]))
                catch
                    density_value = 1 / volume(model, BigFloat(p), BigFloat(t),vol0 = BigFloat(1 /Rho_CES[t_idx, p_idx+1]))
                end
            end

            #println("density_value = $(density_value)")
            Rho_CES[t_idx, p_idx] = density_value
            Sres_CES[t_idx, p_idx] = Clapeyron.VT_entropy_res(model, BigFloat(1 / density_value), BigFloat(t), [1.])
            Phi_CES[t_idx, p_idx] = Clapeyron.VT_fugacity_coefficient(model, BigFloat(1 / density_value), BigFloat(t), [1.])[1]
        end
    end
    rows, cols = nan_index
    for (ro, co) in zip(rows, cols)
        row = ro + 1   # only if indices are 0-based
        col = co + 1
        Rho_CES[row, col] = NaN
    end
    Rho_CES, Sres_CES, Phi_CES = interpolator_filler(model,Rho_CES,Sres_CES, Phi_CES,T, P)
    
    T_crit = (T .>0.998*Tc) .& (T .< 1.001*Tc)
    P_crit = (P .> 0.97*Pc) .& (P .< 0.99*Pc)
    if compound=="CycloHexane" && CES=="CPA" 
        Rho_CES[T_crit, P_crit] .= NaN
        Sres_CES[T_crit, P_crit] .= NaN
        Phi_CES[T_crit, P_crit] .= NaN

        Phi_CES = coolprop_int(model,T,P,Phi_CES,"fugacity",3,3,10000,"clapeyron")
        Rho_CES = coolprop_int(model,T,P,Rho_CES,"density",3,3,10000,"clapeyron")
        Sres_CES = coolprop_int(model,T,P,Sres_CES,"entropy",3,3,10000,"clapeyron") 
    end

    

    #print(np.where(np.isnan(Rho_CES)))
    mat_dir = joinpath(Master_folder, "NPZ_files", CES, "$(CES)_$(compound)")
    #println("Tsat = ",Tsat)
    #println("pv_CES = ", pv_CES)
    ensure_directory_exists(mat_dir)
    matwrite(joinpath(mat_dir, "$(CES) $compound density.mat"), Dict("Rho_CES" => Rho_CES, "Rho_sat_liq" => Rho_sat_liq_CES, "Rho_sat_vap" => Rho_sat_vap_CES,"T" =>T,"P" =>P,"Tc" =>Tc,"Pc" => Pc,"Vc" => Vc,"T_shift" =>T_shift))
    matwrite(joinpath(mat_dir, "$(CES) $compound residual entropy.mat"), Dict("Sres_CES" => Sres_CES, "Sres_sat_liq_CES" => Sres_sat_liq_CES, "Sres_sat_vap_CES" => Sres_sat_vap_CES,"T" =>T,"P" =>P,"Tc" =>Tc,"Pc" => Pc,"Vc" => Vc,"T_shift" =>T_shift))
    matwrite(joinpath(mat_dir, "$(CES) $compound fugacity coefficient.mat"), Dict("Phi_CES" => Phi_CES, "Phi_sat_liq_CES" => Phi_sat_liq_CES, "Phi_sat_vap_CES" => Phi_sat_vap_CES,"T" =>T,"P" =>P,"Tc" =>Tc,"Pc" => Pc,"Vc" => Vc,"T_shift" =>T_shift))
    matwrite(joinpath(mat_dir, "$(CES) $compound Hv.mat"), Dict("Hv_CES" => Hv_CES,"Tsat" =>Tsat,"T_shift" =>T_shift))
    matwrite(joinpath(mat_dir, "$(CES) $compound pv.mat"), Dict("pv_CES" => pv_CES,"Tsat" =>Tsat,"T_shift" =>T_shift))
    
    return T, P, Rho_CES, Sres_CES, Hv_CES, pv_CES, Rho_sat_liq_CES, Rho_sat_vap_CES, Sres_sat_liq_CES, Sres_sat_vap_CES, Phi_CES, Phi_sat_liq_CES, Phi_sat_vap_CES, Tsat, Tc, Pc
end


function Tshifter(handle,T,P)
    i = 0
    cont = true

    while cont
        t = CoolProp.AbstractState.Tmin(handle) + i
        try
            for p in P
                handle.update(CoolProp.PT_INPUTS, p, t)
                den = CoolProp.CoolProp.AbstractState.rhomolar(handle)
            end
            cont = false
        catch
            i = i + 1
        end
    end
    return i
end
function density_CP(compound)
    N = 500
    

    handle = CoolProp.AbstractState("HEOS", compound)
    
    T,P = limit_creator(handle,point_distribution,N,0,compound)
    T_shift = Tshifter(handle,T,P)
    T,P = limit_creator(handle,point_distribution,N,T_shift,compound)
    Tc = CoolProp.AbstractState.T_critical(handle)
    Pc = CoolProp.AbstractState.p_critical(handle)
    PP, TT = np.meshgrid(P, T)
    Tsat = T[T .< Tc] 

    T_crit = (T .>= Tc) .& (T .< 1.003*Tc)
    P_crit = (P .> 1.020*Pc) .& (P .< 1.033*Pc)
    Rho_CP, Phi_CP, Sres_CP, Rho_sat_liq_CP, Phi_sat_liq_CP, Sres_sat_liq_CP, Rho_sat_vap_CP, Phi_sat_vap_CP, Sres_sat_vap_CP, pv_CP, Hv_CP = zeros(Float64, length(T), length(P)), zeros(Float64, length(T), length(P)), zeros(Float64, length(T), length(P)), zeros(length(Tsat)), zeros(length(Tsat)), zeros(length(Tsat)), zeros(length(Tsat)), zeros(length(Tsat)), zeros(length(Tsat)), zeros(length(Tsat)), zeros(length(Tsat))

    for i in 1:N
        t = T[i]
        if i<=length(Tsat)
            handle.update(CoolProp.QT_INPUTS, 0, t)
            hl = CoolProp.CoolProp.AbstractState.hmolar(handle)
            pv_CP[i] = CoolProp.CoolProp.AbstractState.p(handle)
            Rho_sat_liq_CP[i] = CoolProp.CoolProp.AbstractState.rhomolar(handle)
            Phi_sat_liq_CP[i] = CoolProp.CoolProp.AbstractState.fugacity_coefficient(handle, 0)
            Sres_sat_liq_CP[i] = CoolProp.CoolProp.AbstractState.smolar_residual(handle)

            handle.update(CoolProp.QT_INPUTS, 1, t)
            hv = CoolProp.CoolProp.AbstractState.hmolar(handle)
            Hv_CP[i] = hv - hl
            Rho_sat_vap_CP[i] = CoolProp.CoolProp.AbstractState.rhomolar(handle)
            Phi_sat_vap_CP[i] = CoolProp.CoolProp.AbstractState.fugacity_coefficient(handle, 0)
            Sres_sat_vap_CP[i] = CoolProp.CoolProp.AbstractState.smolar_residual(handle)
        end
        for j in 1:N
            p = P[j]
            try
                handle.update(CoolProp.PT_INPUTS, p, t)
                Rho_CP[i, j] = CoolProp.CoolProp.AbstractState.rhomolar(handle)
                Sres_CP[i, j] = CoolProp.CoolProp.AbstractState.smolar_residual(handle)
                Phi_CP[i, j] = CoolProp.CoolProp.AbstractState.fugacity_coefficient(handle, 0)

            catch
                Rho_CP[i, j] = NaN
                Sres_CP[i, j] = NaN
                Phi_CP[i, j] = NaN
            end
        end
    end
    if compound in ["R22","R123","R152A"]
        Rho_CP[T_crit, P_crit] .= NaN
        Sres_CP[T_crit, P_crit] .= NaN
        Phi_CP[T_crit, P_crit] .= NaN
    end

    Phi_CP = coolprop_int(handle,T,P,Phi_CP,"fugacity",3,3,10000,"coolprop")
    Rho_CP = coolprop_int(handle,T,P,Rho_CP,"density",3,3,10000,"coolprop")
    Sres_CP = coolprop_int(handle,T,P,Sres_CP,"entropy",3,3,10000,"coolprop")

    mat_dir = joinpath(Master_folder, "NPZ_files", "CoolProp", "CoolProp_$(compound)")
    ensure_directory_exists(mat_dir)

    matwrite(joinpath(mat_dir, "Coolprop $compound density.mat"), Dict("Rho_CP" => Rho_CP, "Rho_sat_liq" => Rho_sat_liq_CP, "Rho_sat_vap" => Rho_sat_vap_CP, "T" => T, "P" => P,"Tc" =>Tc,"Pc" => Pc,"T_shift" =>T_shift,"Tc" => Tc, "Pc" => Pc))
    matwrite(joinpath(mat_dir, "Coolprop $compound residual entropy.mat"), Dict("Sres_CP" => Sres_CP, "Sres_sat_liq_CP" => Sres_sat_liq_CP, "Sres_sat_vap_CP" => Sres_sat_vap_CP, "T" => T, "P" => P,"Tc" =>Tc,"Pc" => Pc,"T_shift" =>T_shift,"Tc" => Tc, "Pc" => Pc))
    matwrite(joinpath(mat_dir, "Coolprop $compound fugacity coefficient.mat"), Dict("Phi_CP" => Phi_CP, "Phi_sat_liq_CP" => Phi_sat_liq_CP, "Phi_sat_vap_CP" => Phi_sat_vap_CP, "T" => T, "P" => P,"Tc" =>Tc,"Pc" => Pc,"T_shift" =>T_shift,"Tc" => Tc, "Pc" => Pc))
    matwrite(joinpath(mat_dir, "Coolprop $compound Hv.mat"), Dict("Hv_CP" => Hv_CP, "Tsat" => Tsat,"T_shift" =>T_shift,"Tc" => Tc, "Pc" => Pc))
    matwrite(joinpath(mat_dir, "Coolprop $compound pv.mat"), Dict("pv_CP" => pv_CP, "Tsat" => Tsat,"T_shift" =>T_shift,"Tc" => Tc, "Pc" => Pc))
    
    return T, P, Rho_CP, Sres_CP, Hv_CP, pv_CP, Rho_sat_liq_CP, Rho_sat_vap_CP, Sres_sat_liq_CP, Sres_sat_vap_CP, Phi_CP, Phi_sat_liq_CP, Phi_sat_vap_CP, Tsat, Tc, Pc, T_shift
end


function graph_(CES, Name, subs, eos_data, exp_data, sat_liq_exp, sat_vap_exp,
               sat_liq_eos, sat_vap_eos, vp_cool, vp_eos, Tsat_cool, Tsat_eos,
               Tc_cool, Tc_eos, Pc_cool, Pc_eos, T, P)
    Plin, Tlin = P, T
    mat_dir = joinpath(Master_folder, "Figures", CES, "$(CES)_$(subs)")
    ensure_directory_exists(mat_dir)
    plt.figure()
    plt.title("Experimental Isobars")
    plt.xlabel("Temperature [K]")
    plt.ylabel("$(Name)")
    plt.grid()

    for j in 1:500
        plt.plot(Tlin, exp_data[:, j], label="P=$(Plin[j])")
    end

    plt.savefig(joinpath(mat_dir,"experimental_isobars_$(Name)_$(subs).png"), dpi=1000)
    plt.close()

    plt.figure()
    plt.plot(Tsat_eos ./Tc_cool ,sat_liq_eos,label="sat liq")
    plt.plot(Tsat_eos ./Tc_cool ,sat_vap_eos,label="sat vap")
    plt.xlabel("Tr [-]")
    plt.ylabel("Saturation $(Name) ")
    plt.title("$(Name) $(CES) for $(subs)")
    plt.savefig(joinpath(mat_dir, "$(CES) $(subs) sat $(Name).png"))
    plt.close()


    eos_style = ":"
    coo_style = "-"
    eos_color = "k"
    coo_color = "k"
    

    Tmin = minimum(T)
    Pmin = minimum(P)

    handle = CoolProp.AbstractState("HEOS", subs)
    lowest_T(temperature) = sat_prop(handle, temperature, "p", 1) - Pmin
    try
        Tlow = scipy.optimize.fsolve(lowest_T, (Tc_cool + Tmin) / 2)[1]
    catch
        Tlow = Tmin
    end
    
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
        #mat_data_cp = nothing
        ces_files = [
            joinpath(mat_dir_eos, "$(EOS) $compound density.mat"),
            joinpath(mat_dir_eos, "$(EOS) $compound residual entropy.mat"),
            joinpath(mat_dir_eos, "$(EOS) $compound fugacity coefficient.mat"),
            joinpath(mat_dir_eos, "$(EOS) $compound Hv.mat"),
            joinpath(mat_dir_eos, "$(EOS) $compound pv.mat")
        ]
        mat_data_ces = all(isfile, ces_files) ? [matread(f) for f in ces_files] : nothing
        #mat_data_ces = nothing
               

        T, P, Tsat = nothing, nothing, nothing
        Rho_CP, Sres_CP, Hv_CP, pv_CP, Rho_sat_liq_CP, Rho_sat_vap_CP, Sres_sat_liq_CP, Sres_sat_vap_CP, Phi_CP, Phi_sat_liq_CP, Phi_sat_vap_CP, Tsat_cool, Tc_cool, Pc_cool = nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing
        Rho_CES, Sres_CES, Hv_CES, pv_CES, Rho_sat_liq_CES, Rho_sat_vap_CES, Sres_sat_liq_CES, Sres_sat_vap_CES, Phi_CES, Phi_sat_liq_CES, Phi_sat_vap_CES = nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing
        
        
        if mat_data_cp === nothing
            println("Doing coolprop")
            T, P, Rho_CP, Sres_CP, Hv_CP, pv_CP, Rho_sat_liq_CP, Rho_sat_vap_CP, Sres_sat_liq_CP, Sres_sat_vap_CP, Phi_CP, Phi_sat_liq_CP, Phi_sat_vap_CP, Tsat_cool, Tc_cool, Pc_cool, T_shift = density_CP(compound)
            println("Done with coolprop")
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

CESs = ["PatelTeja","iPCSAFT","PR","PCPSAFT","PR78","tcPRC","tcRK","cPR","ADPCSAFT", "BACKSAFT", "Berthelot", "CKSAFT", "Clausius", "CPA", "CPPCSAFT", "PR","DAPT", "EPPR78", "GEPCSAFT" , "GEPCSAFT" , "HeterogcPCPSAFT", "HomogcPCPSAFT", "iPCSAFT", "KU", "LJSAFT","ogSAFT", "PatelTeja", "PCPSAFT", "PCSAFT", "pharmaPCSAFT", "PR78","PSRK", "PTV", "QCPR", "OPCSAFT", "RK", "RKPR","SAFTgammaMie","SAFTVRMie", "SAFTVRMie15", "SAFTVRQMie", "SAFTVRSMie", "SAFTVRSW", "sCKSAFT","sCPA", "softSAFT2016","sPCSAFT", "SRK", "structSAFTgammaMie", "tcPR", "tcPRW" ,"tcRK", "TVTPR", "gcsPCSAFT","TWUSRK", "UMRPR", "vdW", "VTPR"] 



CESs_reversed = ["VTPR","vdW","UMRPR","TWUSRK","gcsPCSAFT","TVTPR","tcRK","tcPRW","tcPR","structSAFTgammaMie","SRK","sPCSAFT","softSAFT2016","sCPA","sCKSAFT",
    "SAFTVRSW","SAFTVRSMie","SAFTVRQMie","SAFTVRMie15","SAFTVRMie","SAFTgammaMie","RKPR","RK","OPCSAFT","QCPR","PTV","PSRK","PR78","pharmaPCSAFT","PCSAFT",
    "PCPSAFT","PatelTeja","ogSAFT","LJSAFT","KU","iPCSAFT","HomogcPCPSAFT","HeterogcPCPSAFT","GEPCSAFT","GEPCSAFT","EPPR78","DAPT","PR","CPPCSAFT","CPA","Clausius","CKSAFT","Berthelot","BACKSAFT","ADPCSAFT","cPR"]

CESs_midfirst = ["PSRK","PR78","pharmaPCSAFT","PCSAFT","PCPSAFT","PatelTeja","ogSAFT","LJSAFT",
    "KU","iPCSAFT","HomogcPCPSAFT","HeterogcPCPSAFT","GEPCSAFT","GEPCSAFT","EPPR78",	
    "DAPT","PR","CPPCSAFT","CPA","Clausius","CKSAFT","Berthelot","BACKSAFT",
    "ADPCSAFT","cPR","VTPR","vdW","UMRPR","TWUSRK","gcsPCSAFT","TVTPR","tcRK",
    "tcPRW","tcPR","structSAFTgammaMie","SRK","sPCSAFT","softSAFT2016","sCPA",
    "sCKSAFT","SAFTVRSW","SAFTVRSMie","SAFTVRQMie","SAFTVRMie15","SAFTVRMie",
    "SAFTgammaMie","RKPR","RK","OPCSAFT","QCPR","PTV"]


for EOS in [ARGS[2]]
    for subs in [ARGS[1]]
        path = joinpath(Master_folder, "NPZ_files", EOS, "$(EOS)_$(subs)")
        #if !ispath(path)
     
        if true
            #model = eval(Meta.parse("$(EOS)([\"$(subs)\"])"))
            println(subs," ",EOS)
            #process_comp(subs,EOS)
            
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

