using Clapeyron, NPZ, PyCall

CoolProp_ = pyimport("CoolProp")
include("iPCSAFT.jl")
include("TVTPR.jl")

handle = CoolProp_.AbstractState("HEOS", "R134a")
Tmin = CoolProp_.AbstractState.Tmin(handle)
Tc = CoolProp_.AbstractState.T_critical(handle)

# Function to calculate density matrix
function density(compound, CES)
    N = 200
    
    model = eval(Meta.parse(CES * "([" * "\"" * compound * "\"])"))

    pc = CoolProp_.AbstractState.p_critical(handle)
    Tmax = CoolProp_.AbstractState.Tmax(handle)

    pmin = 0.001 * pc
    pmax = CoolProp_.AbstractState.pmax(handle)

    T = LinRange(Tmin, Tmax, N)
    P = exp10.(LinRange(log10(pmin), log10(pmax), N))

    # Preallocate the density matrix
    Phi_PR = zeros(Float64, length(T), length(P))
    pv_values = zeros(N)
    Hv_values = zeros(N)

    v0 = nothing
    pv = nothing

    (Tc, pc, vc) = crit_pure(model)

    for t in T
        println(t)
        if t < Tc
            if t == T[1]
                (pv, vl, vv) = saturation_pressure(model, t)
            else
                (pv, vl, vv) = saturation_pressure(model, t, IsoFugacitySaturation(p0 = pv, vl = v0[2], vv = v0[1]))
            end

            hl = Clapeyron.VT_enthalpy(model, vl, t, [1.])
            hv = Clapeyron.VT_enthalpy(model, vv, t, [1.])
            v0 = (vl, vv)
            Hv_values[t.==T] .= hv - hl
            pv_values[t.==T] .= pv
        end

        for pr in P
            if pr < pc && t < Tc
                if pr < pv
                    density_value = 1/volume(model, pr, t; phase = :vapor)
                else
                    density_value = 1/volume(model, pr, t; phase = :liquid)
                end
            else
                density_value = 1/volume(model, pr, t)
            end

            if isnan(density_value)
                handle.update(CoolProp_.PT_INPUTS, pr, t)
                density_value = 1/volume(model, pr, t, vol0=1/handle.rhomolar())
            end
            Phi_PR[T .== t, P .== pr] .= density_value
        end
    end

    filename =  CES * " " * compound * " density.npz"
    NPZ.npzwrite(filename, Dict("Phi_PR" => Phi_PR))

    filename_Hv = "Hv " * CES * " " * compound * ".npz"
    NPZ.npzwrite(filename_Hv, Dict("Hv" => Hv_values))

    filename_pv = "pv " * CES * " " * compound * ".npz"
    NPZ.npzwrite(filename_pv, Dict("pv" => pv_values))
    
    return Phi_PR, Hv_values, pv_values
end


# Example usage:
# To calculate and save the vapor pressure
# calculate_vp(A[1], A[2])

# To calculate and save the enthalpy of vaporization
# calculate_Hv(A[1], A[2])

# To calculate and print the density matrix
# density(A[1], A[2])
