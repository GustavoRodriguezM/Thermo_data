using Clapeyron, NPZ, PyCall, Plots, MAT
using Base.Filesystem

CoolProp = pyimport("CoolProp")
include("iPCSAFT.jl")
include("TVTPR.jl")
scipy = pyimport("scipy")
matplotlib = pyimport("matplotlib")
np = pyimport("numpy")

function density_CES(compound, CES; T_shift = 0.0)
    N = 200

    handle = CoolProp.AbstractState("HEOS", compound)
    Tmin = CoolProp.AbstractState.Tmin(handle) + T_shift
    Tc = CoolProp.AbstractState.T_critical(handle)
    
    model = eval(Meta.parse(CES * "([" * "\"" * compound * "\"])"))

    pc = CoolProp.AbstractState.p_critical(handle)
    Tmax = CoolProp.AbstractState.Tmax(handle)

    pmin = 0.001 * pc
    pmax = CoolProp.AbstractState.pmax(handle)

    T = LinRange(Tmin, Tmax, N)
    P = exp10.(LinRange(log10(pmin), log10(pmax), N))

    # Preallocate the density matrix
    Rho_CES = zeros(Float64, length(T), length(P))
    Phi_CES = zeros(Float64, length(T), length(P))
    Sres_CES = zeros(Float64, length(T), length(P))
    Rho_sat_liq_CES = zeros(N)
    Phi_sat_liq_CES = zeros(N)
    Sres_sat_liq_CES = zeros(N)
    Rho_sat_vap_CES = zeros(N)
    Phi_sat_vap_CES = zeros(N)
    Sres_sat_vap_CES = zeros(N)
    pv_CES = zeros(N)
    Hv_CES = zeros(N)

    v0 = nothing
    pv = nothing

    # From CES
    (Tc, pc, vc) = crit_pure(model)

    for t in T
        if t < Tc
            if t == T[1]
                (pv, vl, vv) = saturation_pressure(model, t)
            else
                (pv, vl, vv) = saturation_pressure(model, t, IsoFugacitySaturation(p0 = pv, vl = v0[2], vv = v0[1]))
            end
            
            if vl > vv
                vl, vv = vv, vl
            end

            hl = Clapeyron.VT_enthalpy(model, vl, t, [1.])
            hv = Clapeyron.VT_enthalpy(model, vv, t, [1.])
            v0 = (vl, vv)

            Rho_sat_liq_CES[t.==T] .= 1 / volume(model, pv, t; phase = :liquid, vol0 = vl)
            Rho_sat_liq_CES[t.==T] .= 1 / volume(model, pv, t; phase = :vapor, vol0 = vv)
            Phi_sat_liq_CES[t.==T] .= fugacity_coefficient(model, pv, t; phase = :liquid, vol0 = vl)
            Phi_sat_vap_CES[t.==T] .= fugacity_coefficient(model, pv, t; phase = :vapor, vol0 = vv)
            Sres_sat_liq_CES[t.==T] .= Clapeyron.VT_entropy_res(model, vl, t, [1.])
            Sres_sat_vap_CES[t.==T] .= Clapeyron.VT_entropy_res(model, vv, t, [1.])
            Hv_CES[t.==T] .= hv - hl
            pv_CES[t.==T] .= pv
        end

        for pr in P
            if pr < pc && t < Tc
                if pr < pv
                    density_value = 1 / volume(model, pr, t; phase = :vapor)
                    fugacity_value = fugacity_coefficient(model, pr, t; phase = :vapor)
                else
                    density_value = 1 / volume(model, pr, t; phase = :liquid)
                    fugacity_value = fugacity_coefficient(model, pr, t; phase = :liquid)
                end
            else
                density_value = 1 / volume(model, pr, t)
                fugacity_value = fugacity_coefficient(model, pr, t)
            end

            if isnan(density_value)
                handle.update(CoolProp.PT_INPUTS, pr, t)
                density_value = 1 / volume(model, pr, t, vol0 = 1 / handle.rhomolar())
            end

            if isnan(fugacity_value)
                handle.update(CoolProp.PT_INPUTS, pr, t)
                fugacity_value = fugacity_coefficient(model, pr, t, vol0 = 1 / handle.rhomolar())
            end

            Rho_CES[T .== t, P .== pr] .= density_value
            Sres_CES[T .== t, P .== pr] .= Clapeyron.VT_entropy_res(model, 1 / density_value, t, [1.])
            Phi_CES[T .== t, P .== pr] .= fugacity_value
        end
    end

    mat_dir = "/perm/vt2/cgr7735/Fugacity_project/Results/$compound/$CES/"
    mkdir(mat_dir)

    Rho_CES = zeros(Float64, length(T), length(P))
    Phi_CES = zeros(Float64, length(T), length(P))
    Sres_CES = zeros(Float64, length(T), length(P))
    Rho_sat_liq_CES = zeros(N)
    Phi_sat_liq_CES = zeros(N)
    Sres_sat_liq_CES = zeros(N)
    Rho_sat_vap_CES = zeros(N)
    Phi_sat_vap_CES = zeros(N)
    Sres_sat_vap_CES = zeros(N)
    pv_CES = zeros(N)
    Hv_CES = zeros(N)

    matwrite("$mat_dir/$CES $compound density.mat", Dict("Rho_CES" => Rho_CES,"Rho_sat_liq"=>Rho_sat_liq_CES,"Rho_sat_vap"=>Rho_sat_vap_CES))
    matwrite("$mat_dir/$CES $compound residual entropy.mat", Dict("Sres_CES" => Sres_CES,"Sres_sat_liq_CES"=>Sres_sat_liq_CES,"Sres_sat_vap_CES"=>Sres_sat_vap_CES))
    matwrite("$mat_dir/$CES $compound fugacity coefficient.mat", Dict("Phi_CES" => Phi_CES,"Phi_sat_liq_CES"=>Phi_sat_liq_CES,"Phi_sat_vap_CES"=>Phi_sat_vap_CES))
    matwrite("$mat_dir/Hv $CES $compound.mat", Dict("Hv_CES" => Hv_CES))
    matwrite("$mat_dir/pv $CES $compound.mat", Dict("pv_CES" => pv_CES))
    
    return T, P, Rho_CES, Sres_CES, Hv_CES, pv_CES , Rho_sat_liq_CES , Rho_sat_vap_CES , Sres_sat_liq_CES , Sres_sat_vap_CES , Phi_CES, Phi_sat_liq_CES , Phi_sat_vap_CES
end

function density_CP(compound; T_shift = 0.0)
    N = 200

    handle = CoolProp.AbstractState("HEOS", compound)
    Tmin = CoolProp.AbstractState.Tmin(handle) + T_shift
    Tc = CoolProp.AbstractState.T_critical(handle)
    
    pc = CoolProp.AbstractState.p_critical(handle)
    Tmax = CoolProp.AbstractState.Tmax(handle)

    pmin = 0.001 * pc
    pmax = CoolProp.AbstractState.pmax(handle)

    T = LinRange(Tmin, Tmax, N)
    P = exp10.(LinRange(log10(pmin), log10(pmax), N))

    # Preallocate the density matrix
    Rho_CP = zeros(Float64, length(T), length(P))
    Sres_CP = zeros(Float64, length(T), length(P))
    pv_CP = zeros(N)
    Hv_CP = zeros(N)

    for t in T
        if t < Tc
            handle.update(CoolProp.QT_INPUTS, 0, t)
            hl = handle.hmolar()
            pv_CP[t.==T] .= handle.p()

            handle.update(CoolProp.QT_INPUTS, 1, t)
            hv = handle.hmolar()
            Hv_CP[t.==T] .= hv - hl
        end

        for pr in P
            handle.update(CoolProp.PT_INPUTS, pr, t)
            Rho_CP[T .== t, P .== pr] .= handle.rhomolar()
            Sres_CP[T .== t, P .== pr] .= handle.smolar_residual()
        end
    end

    mat_dir = "/perm/vt2/cgr7735/Fugacity_project/Results/$compound/"
    mkdir(mat_dir)


    matwrite("$mat_dir/Coolprop $compound density.mat", Dict("Rho_CP" => Rho_CP,"Rho_sat_liq"=>Rho_sat_liq_CP,"Rho_sat_vap"=>Rho_sat_vap_CP))
    matwrite("$mat_dir/Coolprop $compound residual entropy.mat", Dict("Sres_CES" => Sres_CP,"Sres_sat_liq_CES"=>Sres_sat_liq_CP,"Sres_sat_vap_CES"=>Sres_sat_vap_CP))
    matwrite("$mat_dir/Coolprop $compound fugacity coefficient.mat", Dict("Phi_CES" => Phi_CP,"Phi_sat_liq_CES"=>Phi_sat_liq_CP,"Phi_sat_vap_CES"=>Phi_sat_vap_CP))
    matwrite("$mat_dir/Hv Coolprop $compound.mat", Dict("Hv_CP" => Hv_CP))
    matwrite("$mat_dir/pv Coolprop $compound.mat", Dict("pv_CP" => pv_CP))
    
    return T, P, Rho_CP, Sres_CP, Hv_CP, pv_CP , Rho_sat_liq_CP , Rho_sat_vap_CP , Sres_sat_liq_CP , Sres_sat_vap_CP , Phi_CP, Phi_sat_liq_CP , Phi_sat_vap_CP
end


function graph(CES,Name,subs,eos_data,exp_data,sat_liq_exp,sat_vap_exp,sat_liq_eos,sat_vap_eos,vp,T,P)
    mat_dir = "/perm/vt2/cgr7735/Fugacity_project/Figures/$compound/"
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

    # Calculate errors
    Error = transpose(abs.(eos_data .- exp_data) .* 100 ./ exp_data)
    errors_liq = abs.(sat_liq_exp .- sat_liq_eos) ./ sat_liq_exp
    errors_vap = abs.(sat_vap_exp .- sat_vap_eos) ./ sat_vap_exp
    errors = (errors_liq .+ errors_vap) ./ 2

    P, T = np.meshgrid(P,T)

    # Define levels and colormap
    levels = LinRange(0, 30, 11)
    cmap = matplotlib.cm.get_cmap("RdYlGn_r")
    cmap[:set_over]("red")
    norm = matplotlib.colors.BoundaryNorm(levels, ncolors=cmap[:N], clip=false)

    # Plot large figure
    plt.figure(2^k)
    plt.title("$(CES) $(Name) error for $(subs)")
    plt.yscale("log")
    contour = plt.contourf(T ./ Tc, P ./ pc, Error, levels=levels, cmap=cmap, extend="max")
    plt.colorbar(contour, label="Error (%)")
    plt.grid()
    plt.xlabel("Tr")
    plt.ylabel("Pr")
    plt.axvline(x=1, linestyle="--", linewidth=3, color="k")
    plt.axhline(y=1, linestyle="--", linewidth=3, color="k")

    plt.gca()[:set_ylim](bottom=0.01)
    plt.plot([T[i] for (i,pv) in enumerate(vp)]./ Tc, vp./pc, linestyle="-", linewidth=3, color="k")

    scatter_colors = cmap[:__call__](norm(errors))
    plt.scatter([T[i] for (i,pv) in enumerate(vp)]./ Tc, vp ./ pc, c=scatter_colors, edgecolor="black", s=50, zorder=2)

    plt.savefig("$(mat_dir)$(CES) $(Name) $(subs) big.png")
    plt.close()

    # Plot small figure
    plt.figure(3^k)
    plt.title("$(CES) $(Name) error for $(subs)")
    plt.yscale("log")
    contour = plt.contourf(T ./ Tc, P ./ pc, Error, levels=levels, cmap=cmap, extend="max")
    plt.colorbar(contour, label="Error (%)")
    plt.grid()
    plt.xlabel("Tr")
    plt.ylabel("Pr")
    plt.axvline(x=1, linestyle="--", linewidth=3, color="k")
    plt.axhline(y=1, linestyle="--", linewidth=3, color="k")

    plt.gca()[:set_ylim](bottom=0.01)
    plt.plot([T[i] for (i,pv) in enumerate(vp)]./ Tc, vp./pc, linestyle="-", linewidth=3, color="k")

    plt.scatter([T[i] for (i,pv) in enumerate(vp)]./ Tc, vp ./ pc, c=scatter_colors, edgecolor="black", s=50, zorder=2)

    plt.xlim(Tmin / Tc, 2 * (Tc - Tmin) / Tc)
    plt.savefig("$(mat_dir)$(CES) $(Name) $(subs) small.png")
    plt.close()
end

CESs = ["SRK", "tcRK", "PSRK", "PR", "PR78", "cPR", "tcPR", "tcPRW", "QCPR", "VTPR", "PatelTeja", "PTV", "PCSAFT", "PCPSAFT", "iPCSAFT", "ADPCSAFT", "SAFTVRMie", "SAFTVRQMie", "DAPT"]
compounds = ["n-Nonane", "MethylLinolenate", "DimethylCarbonate", "R21", "DiethylEther", "trans-2-Butene", "R245fa", "ParaDeuterium", "OrthoDeuterium", "Isohexane", "R365MFC", "n-Dodecane", "R410A", "Deuterium", "D4", "R13", "MD2M", "n-Hexane", "Methane", "Ethane", "CarbonylSulfide", "EthylBenzene", "CarbonMonoxide", "Isopentane", "Xenon", "cis-2-Butene", "R152A", "Oxygen", "EthyleneOxide", "R1234ze(E)", "n-Octane", "R404A", "R236EA", "CycloHexane", "n-Heptane", "R22", "R113", "n-Pentane", "MethylLinoleate", "R11", "SulfurDioxide", "R23", "Helium", "R32", "R227EA", "R407C", "HydrogenSulfide", "Air", "R245ca", "Novec649", "R143a", "D5", "R507A", "R134a", "Dichloroethane", "ParaHydrogen", "R1233zd(E)", "Acetone", "n-Decane", "HeavyWater", "MethylPalmitate", "n-Propane", "R115", "R1234yf", "R236FA", "Ethylene", "R116", "MD4M", "Benzene", "Methanol", "SulfurHexafluoride", "o-Xylene", "R125", "Fluorine", "R1234ze(Z)", "CarbonDioxide", "IsoButane", "n-Butane", "NitrousOxide", "DimethylEther", "RC318", "Toluene", "IsoButene", "MethylStearate", "Ammonia", "Argon", "R218", "R41", "Neon", "Propyne", "CycloPropane", "R12", "Nitrogen", "Water", "MethylOleate", "R161", "D6", "SES36", "HFE143m", "n-Undecane", "R123", "HydrogenChloride", "m-Xylene", "R141b", "R124", "1-Butene", "Propylene", "R14", "p-Xylene", "Cyclopentane", "MDM", "Hydrogen", "Neopentane", "Ethanol", "OrthoHydrogen", "R114", "Krypton", "MD3M", "R1243zf", "MM", "R142b", "R40", "R13I1"]

for compound in [ARGS[1]]
    try
        fail = true
        T_shift = 0.0

        while fail
            try
                T, P, Rho_CP, Sres_CP, Hv_CP, pv_CP , Rho_sat_liq_CP , Rho_sat_vap_CP , Sres_sat_liq_CP , Sres_sat_vap_CP , Phi_CP, Phi_sat_liq_CP , Phi_sat_vap_CP = density_CP(compound; T_shift = T_shift)
                fail = false
            catch
                T_shift += 1
            end
        end

        T, P, Rho_CP, Sres_CP, Hv_CP, pv_CP , Rho_sat_liq_CP , Rho_sat_vap_CP , Sres_sat_liq_CP , Sres_sat_vap_CP , Phi_CP, Phi_sat_liq_CP , Phi_sat_vap_CP = density_CP(compound; T_shift = T_shift)

        for ces in [ARGS[2]]
            try
                T, P, Rho_CES, Sres_CES, Hv_CES, pv_CES , Rho_sat_liq_CES , Rho_sat_vap_CES , Sres_sat_liq_CES , Sres_sat_vap_CES , Phi_CES, Phi_sat_liq_CES , Phi_sat_vap_CES = density_CES(compound, ces; T_shift = T_shift)

                mat_dir = "/perm/vt2/cgr7735/Fugacity_project/Figures/$compound/"

                # Graph density
                graph(ces, "Density", compound, Rho_CES, Rho_CP, Rho_sat_liq_CP, Rho_sat_vap_CP, Rho_sat_liq_CES, Rho_sat_vap_CES, pv_CP, T, P)

                # Graph fugacity coefficient
                graph(ces, "Fugacity Coefficient", compound, Phi_CES, Phi_CP, Phi_sat_liq_CP, Phi_sat_vap_CP, Phi_sat_liq_CES, Phi_sat_vap_CES, pv_CP, T, P)

                # Graph entropy
                graph(ces, "Residual Entropy", compound, Sres_CES, Sres_CP, Sres_sat_liq_CP, Sres_sat_vap_CP, Sres_sat_liq_CES, Sres_sat_vap_CES, pv_CP, T, P)

                # Additional plots for vapor pressure and enthalpy
                plt = plot(T, abs.(pv_CP .- pv_CES) ./ pv_CP, xlabel = "Temperature [K]", ylabel = "Pressure Error", title = "Vapor Pressure $compound")
                savefig(plt, "$mat_dir/Vapor_Pressure_$(ces)_$compound.png")

                plt = plot(T, abs.(Hv_CP .- Hv_CES) ./ Hv_CP, xlabel = "Temperature [K]", ylabel = "Enthalpy Error", title = "Enthalpy $compound")
                savefig(plt, "$mat_dir/Enthalpy_$(ces)_$compound.png")
            catch
                println("Failed for CES $ces on $compound")
            end
        end
    catch
        println("Failed for $compound")
    end
end
