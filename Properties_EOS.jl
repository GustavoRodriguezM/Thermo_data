using Clapeyron, NPZ, PyCall

CoolProp_ = pyimport("CoolProp")
include("iPCSAFT.jl")
include("TVTPR.jl")

handle = CoolProp_.AbstractState("HEOS", "R40")
Tmin = CoolProp_.AbstractState.Tmin(handle)
Tc = CoolProp_.AbstractState.T_critical(handle)

# Function to calculate vapor pressure across a temperature range
function calculate_vp(compound, CES)
    N = 200
    vp_values = zeros(N)

    model = eval(Meta.parse(CES * "([" * "\"" * compound * "\"])"))

    Tmax = 0.9999 * Tc
    T = LinRange(Tmin, Tmax, N)

    for i in 1:length(T)
        vp_values[i] = saturation_pressure(model, T[i])[1]
    end

    filename_vp = "vp " * CES * " " * compound * ".npz"
    NPZ.npzwrite(filename_vp, Dict("vp" => vp_values))
end

# Function to calculate enthalpy of vaporization across a temperature range
function calculate_pHv(compound, CES)
    N = 200

    model = eval(Meta.parse(CES * "([" * "\"" * compound * "\"])"))

    Tmax = 0.9999 * Tc

    T = LinRange(Tmin, Tmax, N)
    Hv_values = zeros(N)
    pv_values = zeros(N)
    v0 = nothing
    for i in 1:length(T)
        if i == 1
            (pv_values[i], vl, vv) = saturation_pressure(model, T[i])
        else
            (pv_values[i], vl, vv) = saturation_pressure(model, T[i]; v0=v0)
        end
        hl = Clapeyron.VT_enthalpy(model, vl, T[i], [1.])
        hv = Clapeyron.VT_enthalpy(model, vv, T[i], [1.])
        v0 = (vl, vv)
        Hv_values[i] = hv - hl
    end

    filename_Hv = "Hv " * CES * " " * compound * ".npz"
    NPZ.npzwrite(filename_Hv, Dict("Hv" => Hv_values))

    filename_pv = "pv " * CES * " " * compound * ".npz"
    NPZ.npzwrite(filename_pv, Dict("pv" => pv_values))
    return Hv_values, pv_values
end

# Function to calculate density matrix
function density(compound, CES)
    N = 6000
    
    model = eval(Meta.parse(CES * "([" * "\"" * compound * "\"])"))

    pc = CoolProp_.AbstractState.p_critical(handle)
    Tmax = CoolProp_.AbstractState.Tmax(handle)

    pmin = 0.001 * pc
    pmax = CoolProp_.AbstractState.pmax(handle)

    T = LinRange(Tmin, Tmax, N)
    P = LinRange(pmin, pmax, N)

    # Preallocate the density matrix
    Phi_PR = zeros(Float64, length(T), length(P))
    v0 = nothing
    for t in T
        (pv, vl, vv) = saturation_pressure(model, t; v0=v0)
        v0 = (vl, vv)
            
        for pr in P
            if pr>pv
                ph=liquid
            else
                ph=vapor
            density_value = 1/volume(model, pr, t; phase=:ph)

            if isnan(density_value)
                handle.update(CoolProp_.PT_INPUTS, pr, t)
                density_value = 1/volume(model, pr, t; vol0=1/handle.rhomolar(); phase=:ph)
            end
            Phi_PR[T .== t, P .== pr] .= density_value

        end
    end

    filename =  CES * " " * compound * " density.npz"
    NPZ.npzwrite(filename, Dict("Phi_PR" => Phi_PR))
    
    return Phi_PR
end


# Example usage:
# To calculate and save the vapor pressure
calculate_vp(ARGS[1], ARGS[2])

# To calculate and save the enthalpy of vaporization
calculate_Hv(ARGS[1], ARGS[2])

# To calculate and print the density matrix
density(ARGS[1], ARGS[2])
