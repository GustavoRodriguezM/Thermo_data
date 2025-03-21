using Clapeyron, PyCall, Plots, MAT, JSON
using Base.Filesystem
using Statistics

include("iPCSAFT.jl")
include("TVTPR.jl")

optimize = pyimport("scipy.optimize")
CoolProp = pyimport("CoolProp")
scipy = pyimport("scipy")
matplotlib = pyimport("matplotlib")
plt = pyimport("matplotlib.pyplot")
np = pyimport("numpy")

SaftVR = read("Saft_VR_mie.json", String)
SaftVRp = JSON.parse(SaftVR)

CIDD = read("CIDs.json", String)
CID = JSON.parse(CIDD)

function check_and_load_matfiles(compound, CES)
    mat_dir = "/work/vt2/cgr7735/Fugacity_project/NPZ_files/$(CES)/$(CES)_$(compound)"
    files = [
        "$(CES) $compound density.mat",
        "$(CES) $compound residual entropy.mat",
        "$(CES) $compound fugacity coefficient.mat",
        "$(CES) $compound Hv.mat",
        "$(CES) $compound pv.mat"
    ]
    all_exist = all(f -> isfile("$mat_dir/$f"), files)
    data = nothing

    if all_exist
        println("CES MAT files exist for $compound. Loading data...")
        data = (
            matread("$mat_dir/$(CES) $compound density.mat"),
            matread("$mat_dir/$(CES) $compound residual entropy.mat"),
            matread("$mat_dir/$(CES) $compound fugacity coefficient.mat"),
            matread("$mat_dir/$(CES) $compound Hv.mat"),
            matread("$mat_dir/$(CES) $compound pv.mat")
        )
    end

    return data
end

function ensure_directory_exists(dir_path::String)
    if !ispath(dir_path)
        mkpath(dir_path)
    end
end

# Helper function to calculate the average value of the surrounding matrix elements (ignoring NaNs)
function average_surrounding(matrix, i, j)
    sum = 0.0
    count = 0
    for di in -1:1
        for dj in -1:1
            
            if !(di == 0 && dj == 0) && i + di >= 1 && i + di <= size(matrix, 1) && j + dj >= 1 && j + dj <= size(matrix, 2)
                val = matrix[i + di, j + dj]
                if !isnan(val) && val!=0
                    println("val = ", val)
                    sum += val
                    count += 1
                end
            end
        end
    end
    return sum / count 
end



function SAFTVR_get(Name)
    Prop=CID[Name]
    Saft_parameters=SaftVRp[Prop["CID"]]
    return Saft_parameters,parse(Float64,Prop["Mw"]),parse(Float64,Prop["n_H"]),parse(Float64,Prop["n_e"])

end


function Initiator(CES,Name)
    if CES=="SAFTVRMie"
        Parameters,Mw1,H,e=SAFTVR_get(Name)
        if Parameters["epsilonAB"]!="\u2014\u2014"

            a=float(Parameters["epsilonAB"])
            b=float(Parameters["kAB "])
        else
            a=0
            b=0
        end

        model = SAFTVRMie([Name]; userlocations=(;
        Mw = [Mw1],
        segment = [float(Parameters["m"])],
        sigma = [float(Parameters["sigma"])], 
        epsilon = [float(Parameters["epsilon"])], 
        lambda_a = [float(Parameters["i_a"])],
        lambda_r = [float(Parameters["i_r"])],
        n_H=[(H)],
        n_e=[(e)],
        epsilon_assoc = Dict(((Name,"e"),(Name,"H")) =>a),
        bondvol = Dict(((Name,"e"),(Name,"H")) => b*10^-30 )
        ))
    else
        model = eval(Meta.parse(CES*"([\""*Name*"\"])"))
    end
    return model
end

function density_CES(compound, CES; T_shift = 0.0)
    N = 500

    handle = CoolProp.AbstractState("HEOS", compound)

    pc = CoolProp.AbstractState.p_critical(handle)
    Tmax = CoolProp.AbstractState.Tmax(handle)

    handle.update(CoolProp.QT_INPUTS, 0, CoolProp.AbstractState.Tmin(handle))

    pmin = handle.p()
    pmax = CoolProp.AbstractState.pmax(handle)
        
    if pmin == handle.p()
        Tmin = CoolProp.AbstractState.Tmin(handle) + T_shift
    else
        try
            handle.update(CoolProp.PQ_INPUTS, pmin*100, 1)
            Tmin = max(CoolProp.AbstractState.Tmin(handle), handle.T()-50) + T_shift
        catch
            Tmin = CoolProp.AbstractState.Tmin(handle) + T_shift
        end
    end
    print("Tmin = ",Tmin)
    Tc = CoolProp.AbstractState.T_critical(handle)
    
    model = Initiator(CES,compound)

    T = LinRange(Tmin, Tmax, N)
    Tsat = [t for t in T if t < Tc]
    P = collect(LinRange(pmin, pmax, N))


    # Preallocate the density matrix with NaN for handling later
    Rho_CES = fill(NaN, length(T), length(P))
    Phi_CES = zeros(Float64, length(T), length(P))
    Sres_CES = zeros(Float64, length(T), length(P))
    Rho_sat_liq_CES = zeros(length(Tsat))
    Phi_sat_liq_CES = zeros(length(Tsat))
    Sres_sat_liq_CES = zeros(length(Tsat))
    Rho_sat_vap_CES = zeros(length(Tsat))
    Phi_sat_vap_CES = zeros(length(Tsat))
    Sres_sat_vap_CES = zeros(length(Tsat))
    pv_CES = zeros(length(Tsat))
    Hv_CES = zeros(length(Tsat))

    v0 = nothing
    pv = nothing
    
    Den_sat_model(T, A, B, n) = A * B .^ (.-(abs.(1 .- T ./ Tc)).^ n)
    # From CES
    (Tc, pc, vc) = crit_pure(model)

     

    for (t_idx, t) in enumerate(T)
        if t in Tsat
            
            if t == T[1]
                (pv, vl, vv) = saturation_pressure(model, t)
            else
                (pv, vl, vv) = saturation_pressure(model, t)
                if isnan(vl) || isinf(vl)
                    (pv, vl, vv) = saturation_pressure(model, t, IsoFugacitySaturation(p0 = pv, vl = v0[2], vv = v0[1]))
                    if isnan(vl) || isinf(vl)
                        (pv, vl, vv) = saturation_pressure(model, t, v0=(v0[2],v0[1]))
                        if isnan(vl)  || isinf(vl)
                            (pv, vl, vv) = saturation_pressure(model, t, v0=(1.3*v0[2],1.4*v0[1]))
                            """
                            vl_pol = nothing
                            vv_pol = nothing
                            try
                                Tfit = T[t_idx-4:t_idx-1]
                                vl_fit = Rho_sat_liq_CES[t_idx-4:t_idx-1]
                                vv_fit = Rho_sat_vap_CES[t_idx-4:t_idx-1]
                                vl_pol = np.polyfit(Tfit, vl_fit, 2)
                                vv_pol = np.polyfit(Tfit, vv_fit, 2)
                            catch
                                continue
                            end
                            vl0 = np.polyval(vl_pol, t)
                            vv0 = np.polyval(vv_pol, t)

                            (pv, vl, vv) = saturation_pressure(model, t, v0=(vl0,vv0))
                            """
                        end
                    end
                end
            end
           
            if vl > vv
                vl, vv = vv, vl
            end
            
            hl = Clapeyron.VT_enthalpy(model, vl, t, [1.])
            hv = Clapeyron.VT_enthalpy(model, vv, t, [1.])
            v0 = (vl, vv)

            Rho_sat_liq_CES[t_idx] = 1 / vl
            Rho_sat_vap_CES[t_idx] = 1 / vv
            Phi_sat_liq_CES[t_idx] = Clapeyron.VT_fugacity_coefficient(model, vl, t, [1.])[1]
            Phi_sat_vap_CES[t_idx] = Clapeyron.VT_fugacity_coefficient(model, vv, t, [1.])[1]
            Sres_sat_liq_CES[t_idx] = Clapeyron.VT_entropy_res(model, vl, t, [1.])
            Sres_sat_vap_CES[t_idx] = Clapeyron.VT_entropy_res(model, vv, t, [1.])
            Hv_CES[t_idx] = hv - hl
            pv_CES[t_idx] = pv

            handle.update(CoolProp.QT_INPUTS, 0, t)
        end

        for (p_idx, pr) in enumerate(P)
            if pr < pc && t < Tc
                if pr < pv
                    density_value = 1 / volume(model, pr, t; phase = :vapor)
                    if isnan(density_value) || isinf(density_value)
                        density_value = 1 / volume(model, pr, t,vol0 = 1/average_surrounding(Rho_CES, t_idx, p_idx); phase = :vapor)
                    end
                else
                    density_value = 1 / volume(model, pr, t; phase = :liquid)
                    if isnan(density_value) || isinf(density_value)
                        density_value = 1 / volume(model, pr, t,vol0 = 1/average_surrounding(Rho_CES, t_idx, p_idx); phase = :liquid)
                    end
                end
            else
                density_value = 1 / volume(model, pr, t,)
                if isnan(density_value) || isinf(density_value)
                    density_value = 1 / volume(model, pr, t,vol0 = 1/average_surrounding(Rho_CES, t_idx, p_idx))
                end
            end

            

            Rho_CES[t_idx, p_idx] = density_value
            Sres_CES[t_idx, p_idx] = Clapeyron.VT_entropy_res(model, 1 / density_value, t, [1.])
            Phi_CES[t_idx, p_idx] = Clapeyron.VT_fugacity_coefficient(model, 1/density_value, t, [1.])[1]
        end
    end

    for (t_ind, t) in enumerate(Iterators.reverse(Tsat))
        idx = length(Tsat) - t_ind + 1  # Calculate the current index in the original order
    
        if (!isnan(Rho_sat_liq_CES[idx]) && !isnan(Rho_sat_vap_CES[idx])) || (!isinf(Rho_sat_liq_CES[idx]) && !isinf(Rho_sat_vap_CES[idx]))
            continue
        else
            prev_idx = idx + 1
            if prev_idx <= length(Tsat)
                (pv, vl, vv) = saturation_pressure(model, t,IsoFugacitySaturation(p0=pv_CES[prev_idx],vl=1/Rho_sat_liq_CES[prev_idx],vv=1/Rho_sat_vap_CES[prev_idx]))
                if isnan(vl) || isnan(vv)        
                    (pv, vl, vv) = saturation_pressure(model, t, v0=(1/Rho_sat_liq_CES[prev_idx], 1/Rho_sat_vap_CES[prev_idx]))
                end
            else
                # In case we're at the end, no previous point to use
                (pv, vl, vv) = saturation_pressure(model, t)
            end
            """
            # If that failed, fit a parabola with the last 3 valid points
            if isnan(vl) || isnan(vv)
                start_fit = max(1, t_ind - 3)
                end_fit = t_ind -2
    
                Tfit = Tsat[start_fit:end_fit]
                vl_fit = Rho_sat_liq_CES[start_fit:end_fit]
                vv_fit = Rho_sat_vap_CES[start_fit:end_fit]
                print("vl_fit = ",vl_fit," vv_fit = ",vv_fit)
                vl_pol = np.polyfit(Tfit, vl_fit, 2)
                vv_pol = np.polyfit(Tfit, vv_fit, 2)
    
                vl0 = np.polyval(vl_pol, t)
                vv0 = np.polyval(vv_pol, t)
    
                (pv, vl, vv) = saturation_pressure(model, t, v0=(vl0, vv0))
            end
            """
        end
    
        # Store results
        Rho_sat_liq_CES[idx] = 1 / vl
        Rho_sat_vap_CES[idx] = 1 / vv
        Phi_sat_liq_CES[idx] = Clapeyron.VT_fugacity_coefficient(model, vl, t, [1.])[1]
        Phi_sat_vap_CES[idx] = Clapeyron.VT_fugacity_coefficient(model, vv, t, [1.])[1]
        Sres_sat_liq_CES[idx] = Clapeyron.VT_entropy_res(model, vl, t, [1.])
        Sres_sat_vap_CES[idx] = Clapeyron.VT_entropy_res(model, vv, t, [1.])
        hv = Clapeyron.VT_enthalpy(model, vv, t, [1.]) # Assuming you calculate hv somewhere
        hl = Clapeyron.VT_enthalpy(model, vl, t, [1.]) # Assuming you calculate hl somewhere
        Hv_CES[idx] = hv - hl
        pv_CES[idx] = pv
    end


    for (t_idx, t) in enumerate(T)
        for (p_idx, pr) in enumerate(P)
            if isnan(Rho_CES[t_idx, p_idx]) || isinf(Rho_CES[t_idx, p_idx])
                density_value = 1 / volume(model, pr, t,vol0 = 1/average_surrounding(Rho_CES, t_idx, p_idx)) 
                Rho_CES[t_idx, p_idx] = density_value
                Sres_CES[t_idx, p_idx] = Clapeyron.VT_entropy_res(model, 1 / density_value, t, [1.])
                Phi_CES[t_idx, p_idx] = Clapeyron.VT_fugacity_coefficient(model, 1/density_value, t, [1.])[1]
            end
        end
    end

    
    N = 10

    for x_bias in -N:N
        for y_bias in -N:N
            for (t_idx, t) in enumerate(T)
                for (p_idx, pr) in enumerate(P)
                    if (y_bias==0 && x_bias==0) || (!isnan(Rho_CES[t_idx, p_idx]) && !isinf(Rho_CES[t_idx, p_idx])) 
                        continue
                    end
                    if isnan(Rho_CES[t_idx, p_idx]) || !isinf(Rho_CES[t_idx, p_idx])
                        density_value = 1 / volume(model, pr, t,vol0 = 1/Rho_CES[min(500,max(1,t_idx+x_bias)), min(500,max(1,p_idx+y_bias))]) 
                        Rho_CES[t_idx, p_idx] = density_value
                        Sres_CES[t_idx, p_idx] = Clapeyron.VT_entropy_res(model, 1 / density_value, t, [1.])
                        Phi_CES[t_idx, p_idx] = Clapeyron.VT_fugacity_coefficient(model, 1/density_value, t, [1.])[1]
                    end
                end
            end
        end
    end

    for x_bias in -N:N
        for y_bias in -N:N
            for (t_idx, t) in enumerate(T)
                for (p_idx, pr) in enumerate(P)
                    if (y_bias==0 && x_bias==0) || (!isnan(Rho_CES[t_idx, p_idx]) && !isinf(Rho_CES[t_idx, p_idx])) 
                        continue
                    end
                    if isnan(Rho_CES[t_idx, p_idx]) || !isinf(Rho_CES[t_idx, p_idx])
                        density_value = 1 / volume(model, pr, t,vol0 = 1/Rho_CES[min(500,max(1,t_idx+x_bias)), min(500,max(1,p_idx+y_bias))]) 
                        Rho_CES[t_idx, p_idx] = density_value
                        Sres_CES[t_idx, p_idx] = Clapeyron.VT_entropy_res(model, 1 / density_value, t, [1.])
                        Phi_CES[t_idx, p_idx] = Clapeyron.VT_fugacity_coefficient(model, 1/density_value, t, [1.])[1]
                    end
                end
            end
        end
    end


    for (t_idx, t) in enumerate(T)
        for (p_idx, pr) in enumerate(P)
            if isnan(Rho_CES[t_idx, p_idx]) || isinf(Rho_CES[t_idx, p_idx])
                density_value = 1 / volume(model, pr, t,vol0 = 1/average_surrounding(Rho_CES, t_idx, p_idx)) 
                Rho_CES[t_idx, p_idx] = density_value
                Sres_CES[t_idx, p_idx] = Clapeyron.VT_entropy_res(model, 1 / density_value, t, [1.])
                Phi_CES[t_idx, p_idx] = Clapeyron.VT_fugacity_coefficient(model, 1/density_value, t, [1.])[1]
            end
        end
    end

    mat_dir = "/work/vt2/cgr7735/Fugacity_project/NPZ_files/$(CES)/$(CES)_$(compound)"
    ensure_directory_exists(mat_dir) # Create directory if it doesn't exist
    println("Rho_sat_liq_CES = ",Rho_sat_liq_CES)
    println("Rho_sat_vap_CES = ",Rho_sat_vap_CES)
    matwrite("$mat_dir/$CES $compound density.mat", Dict("Rho_CES" => Rho_CES, "Rho_sat_liq" => Rho_sat_liq_CES, "Rho_sat_vap" => Rho_sat_vap_CES))
    matwrite("$mat_dir/$CES $compound residual entropy.mat", Dict("Sres_CES" => Sres_CES, "Sres_sat_liq_CES" => Sres_sat_liq_CES, "Sres_sat_vap_CES" => Sres_sat_vap_CES))
    matwrite("$mat_dir/$CES $compound fugacity coefficient.mat", Dict("Phi_CES" => Phi_CES, "Phi_sat_liq_CES" => Phi_sat_liq_CES, "Phi_sat_vap_CES" => Phi_sat_vap_CES))
    matwrite("$mat_dir/$CES $compound Hv.mat", Dict("Hv_CES" => Hv_CES))
    matwrite("$mat_dir/$CES $compound pv.mat", Dict("pv_CES" => pv_CES))
    
    return T, P, Rho_CES, Sres_CES, Hv_CES, pv_CES, Rho_sat_liq_CES, Rho_sat_vap_CES, Sres_sat_liq_CES, Sres_sat_vap_CES, Phi_CES, Phi_sat_liq_CES, Phi_sat_vap_CES, Tsat
end

function density_CP(compound; T_shift = 0.0)
    N = 500

    handle = CoolProp.AbstractState("HEOS", compound)

    pc = CoolProp.AbstractState.p_critical(handle)
    Tmax = CoolProp.AbstractState.Tmax(handle)

    handle.update(CoolProp.QT_INPUTS, 0, CoolProp.AbstractState.Tmin(handle))

    pmin = handle.p()
    pmax = CoolProp.AbstractState.pmax(handle)

    if pmin == handle.p()
        Tmin = CoolProp.AbstractState.Tmin(handle)+T_shift
    else
        try
            handle.update(CoolProp.PQ_INPUTS, pmin*100, 1)
            Tmin = max(CoolProp.AbstractState.Tmin(handle), handle.T()-50) +T_shift
        catch
            Tmin = CoolProp.AbstractState.Tmin(handle)+T_shift
        end
    end
    
    Tc = CoolProp.AbstractState.T_critical(handle)

    T = collect(LinRange(Tmin, Tmax, N))
    Tsat = [t for t in T if t < Tc]
    P = collect(LinRange(pmin, pmax, N))

    # Preallocate the density matrix
    Rho_CP = zeros(Float64, length(T), length(P))
    Phi_CP = zeros(Float64, length(T), length(P))
    Sres_CP = zeros(Float64, length(T), length(P))
    Rho_sat_liq_CP = zeros(length(Tsat))
    Phi_sat_liq_CP = zeros(length(Tsat))
    Sres_sat_liq_CP = zeros(length(Tsat))
    Rho_sat_vap_CP = zeros(length(Tsat))
    Phi_sat_vap_CP = zeros(length(Tsat))
    Sres_sat_vap_CP = zeros(length(Tsat))
    pv_CP = zeros(length(Tsat))
    Hv_CP = zeros(length(Tsat))

    for t in T
        if t in Tsat
            handle.update(CoolProp.QT_INPUTS, 0, t)
            hl = CoolProp.CoolProp.AbstractState.hmolar(handle)
            pv_CP[t.==Tsat] .= CoolProp.CoolProp.AbstractState.p(handle)
            Rho_sat_liq_CP[t.==Tsat] .= CoolProp.CoolProp.AbstractState.rhomolar(handle)
            Phi_sat_liq_CP[t.==Tsat] .= CoolProp.CoolProp.AbstractState.fugacity_coefficient(handle,0)
            Sres_sat_liq_CP[t.==Tsat] .= CoolProp.CoolProp.AbstractState.smolar_residual(handle)


            handle.update(CoolProp.QT_INPUTS, 1, t)
            hv = CoolProp.CoolProp.AbstractState.hmolar(handle)
            Hv_CP[t.==Tsat] .= hv - hl
            Rho_sat_vap_CP[t.==Tsat] .= CoolProp.CoolProp.AbstractState.rhomolar(handle)
            Phi_sat_vap_CP[t.==Tsat] .= CoolProp.CoolProp.AbstractState.fugacity_coefficient(handle,0)
            Sres_sat_vap_CP[t.==Tsat] .= CoolProp.CoolProp.AbstractState.smolar_residual(handle)
        end

        for pr in P
            handle.update(CoolProp.PT_INPUTS, pr, t)
            Rho_CP[T .== t, P .== pr] .= CoolProp.CoolProp.AbstractState.rhomolar(handle)
            Sres_CP[T .== t, P .== pr] .= CoolProp.CoolProp.AbstractState.smolar_residual(handle)
            Phi_CP[T .== t, P .== pr] .= CoolProp.CoolProp.AbstractState.fugacity_coefficient(handle,0)
        end
    end

    mat_dir = "/work/vt2/cgr7735/Fugacity_project/NPZ_files/CoolProp/CoolProp_$(compound)/"
    ensure_directory_exists(mat_dir) # Create directory if it doesn't exist

    matwrite("$mat_dir/Coolprop $compound density.mat", Dict("Rho_CP" => Rho_CP,"Rho_sat_liq"=>Rho_sat_liq_CP,"Rho_sat_vap"=>Rho_sat_vap_CP,"T"=>T,"P"=>P))
    matwrite("$mat_dir/Coolprop $compound residual entropy.mat", Dict("Sres_CP" => Sres_CP,"Sres_sat_liq_CP"=>Sres_sat_liq_CP,"Sres_sat_vap_CP"=>Sres_sat_vap_CP,"T"=>T,"P"=>P))
    matwrite("$mat_dir/Coolprop $compound fugacity coefficient.mat", Dict("Phi_CP" => Phi_CP,"Phi_sat_liq_CP"=>Phi_sat_liq_CP,"Phi_sat_vap_CP"=>Phi_sat_vap_CP,"T"=>T,"P"=>P))
    matwrite("$mat_dir/Coolprop $compound Hv.mat", Dict("Hv_CP" => Hv_CP,"Tsat"=>Tsat))
    matwrite("$mat_dir/Coolprop $compound pv.mat", Dict("pv_CP" => pv_CP,"Tsat"=>Tsat))
    
    return T, P, Rho_CP, Sres_CP, Hv_CP, pv_CP , Rho_sat_liq_CP , Rho_sat_vap_CP , Sres_sat_liq_CP , Sres_sat_vap_CP , Phi_CP, Phi_sat_liq_CP , Phi_sat_vap_CP, Tsat
end

function graph(CES, Name, subs, eos_data, exp_data, sat_liq_exp, sat_vap_exp, sat_liq_eos, sat_vap_eos, vp, T, P)
    mat_dir = "/work/vt2/cgr7735/Fugacity_project/Figures/$CES/$(CES)_$(subs)/"
    ensure_directory_exists(mat_dir) # Create directory if it doesn't exist

    handle = CoolProp.AbstractState("HEOS", subs)
    pc = handle.p_critical()
    Tc = handle.T_critical()

    Tmin = minimum(T)
    Pmin = minimum(P)

    function lowest_T(temperature)
        return sat_prop(handle, temperature, "p", 1) - Pmin
    end

    try
        Tlow = scipy.optimize.fsolve(lowest_T, (Tc + Tmin) / 2)[1]
    catch
        Tlow = Tmin
    end
    

    Tsat = [T[i] for (i,pv) in enumerate(vp)]
    indexes = round.(Int, range(1, length(Tsat), length=50))

    Tsam = Tsat[indexes]
    Vpsam= vp[indexes]

    # Calculate errors
    Error = (abs.(eos_data .- exp_data) .* 100 ./ abs.(exp_data))
    errors_liq = abs.(sat_liq_exp[indexes] .- sat_liq_eos[indexes]) ./ sat_liq_exp[indexes]
    errors_vap = abs.(sat_vap_exp[indexes] .- sat_vap_eos[indexes]) ./ sat_vap_exp[indexes]
    errors = (errors_liq .+ errors_vap) ./ 2
    P, T = np.meshgrid(P, T)

    # Define levels and colormap
    levels = LinRange(0, 30, 11)
    cmap = matplotlib.cm.get_cmap("RdYlGn_r")
    cmap[:set_over]("red")
    norm = matplotlib.colors.BoundaryNorm(levels, ncolors=cmap[:N], clip=false)

    # Plot large figure
    plt.figure(2^3)
    plt.title("$(CES) $(Name) error for $(subs)")
    plt.yscale("log")
    contour = plt.contourf(T ./ Tc, P ./ pc, Error, levels=levels, cmap=cmap, extend="max")
    plt.colorbar(contour, label="Error (%)")
    plt.grid()
    plt.xlabel("\$T_{r}\$ [-]")
    plt.ylabel("\$P_{r}\$ [-]")
    plt.axvline(x=1, linestyle="--", linewidth=3, color="k")
    plt.axhline(y=1, linestyle="--", linewidth=3, color="k")

    plt.gca()[:set_ylim](bottom=0.01)
    plt.plot(Tsat./ Tc, vp./pc, linestyle="-", linewidth=3, color="k")

    scatter_colors = cmap[:__call__](norm(errors))
    plt.scatter(Tsam./ Tc, Vpsam ./ pc, c=scatter_colors, edgecolor="black", s=50, zorder=2)

    plt.savefig("$(mat_dir)$(CES) $(Name) $(subs) big.png")
    plt.close()

    # Plot small figure
    plt.figure(3^7)
    plt.title("$(CES) $(Name) error for $(subs)")
    plt.yscale("log")
    contour = plt.contourf(T ./ Tc, P ./ pc, Error, levels=levels, cmap=cmap, extend="max")
    plt.colorbar(contour, label="Error (%)")
    plt.grid()
    plt.xlabel("\$T_{r}\$ [-]")
    plt.ylabel("\$P_{r}\$ [-]")

    plt.axvline(x=1, linestyle="--", linewidth=3, color="k")
    plt.axhline(y=1, linestyle="--", linewidth=3, color="k")

    plt.gca()[:set_ylim](bottom=0.01)
    plt.plot([T[i] for (i,pv) in enumerate(vp)]./ Tc, vp./pc, linestyle="-", linewidth=3, color="k")

    plt.scatter(Tsam./ Tc, Vpsam ./ pc, c=scatter_colors, edgecolor="black", s=50, zorder=2)

    plt.xlim(Tmin / Tc, 2 * (Tc - Tmin) / Tc)
    plt.savefig("$(mat_dir)$(CES) $(Name) $(subs) small.png")
    plt.close()
end

CESs = ["SRK", "tcRK", "PSRK", "PR", "PR78", "cPR", "tcPR", "tcPRW", "QCPR", "VTPR", "PatelTeja", "PTV", "PCSAFT", "PCPSAFT", "iPCSAFT", "ADPCSAFT", "SAFTVRMie", "SAFTVRQMie", "DAPT"]
compounds = ["n-Nonane", "MethylLinolenate", "DimethylCarbonate", "R21", "DiethylEther", "trans-2-Butene", "R245fa", "ParaDeuterium", "OrthoDeuterium", "Isohexane", "R365MFC", "n-Dodecane", "R410A", "Deuterium", "D4", "R13", "MD2M", "n-Hexane", "Methane", "Ethane", "CarbonylSulfide", "EthylBenzene", "CarbonMonoxide", "Isopentane", "Xenon", "cis-2-Butene", "R152A", "Oxygen", "EthyleneOxide", "R1234ze(E)", "n-Octane", "R404A", "R236EA", "CycloHexane", "n-Heptane", "R22", "R113", "n-Pentane", "MethylLinoleate", "R11", "SulfurDioxide", "R23", "Helium", "R32", "R227EA", "R407C", "HydrogenSulfide", "Air", "R245ca", "Novec649", "R143a", "D5", "R507A", "R134a", "Dichloroethane", "ParaHydrogen", "R1233zd(E)", "Acetone", "n-Decane", "HeavyWater", "MethylPalmitate", "n-Propane", "R115", "R1234yf", "R236FA", "Ethylene", "R116", "MD4M", "Benzene", "Methanol", "SulfurHexafluoride", "o-Xylene", "R125", "Fluorine", "R1234ze(Z)", "CarbonDioxide", "IsoButane", "n-Butane", "NitrousOxide", "DimethylEther", "RC318", "Toluene", "IsoButene", "MethylStearate", "Ammonia", "Argon", "R218", "R41", "Neon", "Propyne", "CycloPropane", "R12", "Nitrogen", "Water", "MethylOleate", "R161", "D6", "SES36", "HFE143m", "n-Undecane", "R123", "HydrogenChloride", "m-Xylene", "R141b", "R124", "1-Butene", "Propylene", "R14", "p-Xylene", "Cyclopentane", "MDM", "Hydrogen", "Neopentane", "Ethanol", "OrthoHydrogen", "R114", "Krypton", "MD3M", "R1243zf", "MM", "R142b", "R40", "R13I1"]

T, P, Rho_CES, Sres_CES, Hv_CES, pv_CES, Rho_sat_liq_CES, Rho_sat_vap_CES, Sres_sat_liq_CES, Sres_sat_vap_CES, Phi_CES, Phi_sat_liq_CES, Phi_sat_vap_CES, Tsat = density_CES(ARGS[1], ARGS[2]; T_shift = 10)
        

# Main execution
for compound in [ARGS[1]]
  
    ces = ARGS[2]
    T_shift = 0.0
    failed_loading = false
    
    # Check if CoolProp MAT files exist and load
    #mat_data_cp = check_and_load_matfiles(compound, "Coolprop")
    #mat_data_ces = check_and_load_matfiles(compound, ces)
    
    mat_data_cp = nothing
    mat_data_ces = nothing

    # Initialize variables for computed data
    T, P, Tsat = nothing, nothing, nothing
    Rho_CP, Sres_CP, Hv_CP, pv_CP, Rho_sat_liq_CP, Rho_sat_vap_CP, Sres_sat_liq_CP, Sres_sat_vap_CP, Phi_CP, Phi_sat_liq_CP, Phi_sat_vap_CP = nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing
    Rho_CES, Sres_CES, Hv_CES, pv_CES, Rho_sat_liq_CES, Rho_sat_vap_CES, Sres_sat_liq_CES, Sres_sat_vap_CES, Phi_CES, Phi_sat_liq_CES, Phi_sat_vap_CES = nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing

    if mat_data_cp === nothing
        # Compute CoolProp data if MAT files don't exist
        fail = true
        while fail
            try
                T, P, Rho_CP, Sres_CP, Hv_CP, pv_CP, Rho_sat_liq_CP, Rho_sat_vap_CP, Sres_sat_liq_CP, Sres_sat_vap_CP, Phi_CP, Phi_sat_liq_CP, Phi_sat_vap_CP = density_CP(compound; T_shift = T_shift)
                fail = false
            catch
                T_shift += 1
            end
        end
    else
        # If MAT files exist, load CoolProp data
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
        Tsat = mat_data_cp[4]["Tsat"]
    end

    if mat_data_ces === nothing
        # Compute CES data if MAT files don't exist
        try
            T, P, Rho_CES, Sres_CES, Hv_CES, pv_CES, Rho_sat_liq_CES, Rho_sat_vap_CES, Sres_sat_liq_CES, Sres_sat_vap_CES, Phi_CES, Phi_sat_liq_CES, Phi_sat_vap_CES, Tsat = density_CES(compound, ces; T_shift = T_shift)
        catch e
            println("Failed for CES $ces on $compound")
            println(e)
            failed_loading = true
        end
    else
        # If MAT files exist, load CES data
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
    end

    # Proceed to graphing if calculations or loading succeeded
    if !failed_loading
        mat_dir = "/work/vt2/cgr7735/Fugacity_project/Figures/$(ARGS[2])/$(ARGS[2])_$(ARGS[1])/"
        ensure_directory_exists(mat_dir) # Create directory if it doesn't exist
        
        # Graph entropy
        graph(ARGS[2], "Residual molar entropy", compound, Sres_CES, Sres_CP, Sres_sat_liq_CP, Sres_sat_vap_CP, Sres_sat_liq_CES, Sres_sat_vap_CES, pv_CP, T, P)

        # Graph density
        graph(ARGS[2], "Density", compound, Rho_CES, Rho_CP, Rho_sat_liq_CP, Rho_sat_vap_CP, Rho_sat_liq_CES, Rho_sat_vap_CES, pv_CP, T, P)

        # Graph fugacity coefficient
        graph(ARGS[2], "Fugacity Coefficient", compound, Phi_CES, Phi_CP, Phi_sat_liq_CP, Phi_sat_vap_CP, Phi_sat_liq_CES, Phi_sat_vap_CES, pv_CP, T, P)

        # Additional plots for vapor pressure and enthalpy
        plt = plot(Tsat, abs.(pv_CP .- pv_CES) .* 100 ./ pv_CP, xlabel = "Temperature [K]", ylabel = "Pressure Error", title = "Vapor Pressure $compound",xlims=(minimum(Tsat), maximum(Tsat)))
        savefig(plt, "$(mat_dir)Vapor Pressure $(ARGS[2]) $compound.png")
        
        plt = plot(Tsat, abs.(Hv_CP .- Hv_CES) .* 100 ./ Hv_CP, xlabel = "Temperature [K]", ylabel = "Enthalpy Error", title = "Enthalpy $compound",xlims=(minimum(Tsat), maximum(Tsat)))
        savefig(plt, "$(mat_dir)Enthalpy $(ARGS[2]) $compound.png")
    end
end

