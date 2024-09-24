using Pkg

Pkg.activate(".")

using MiCRM_stability
using LinearAlgebra, KrylovKit
using StatsBase, Distances, Distributions
using CairoMakie
using DelimitedFiles
using Random
using Graphs, GraphMakie, NetworkLayout, SimpleWeightedGraphs
using LsqFit

println(pwd())
println(Threads.nthreads())

include("./functions.jl")

function get_param_e_mean(p::MiCRM_stability.Parameters)
    fe = fieldnames(MiCRM_stability.exponential_params)
    ue = mean.(getfield.(Ref(p.e), fe))
    vcat(ue...) 
end

function get_param_s_mean(p::MiCRM_stability.Parameters)
    fs = fieldnames(MiCRM_stability.structural_params)
    us = mean.(getfield.(Ref(p.s), fs))
    vcat(us...) 
end


function get_exponential_parameters(N,M)
    σ = 0.1
    gx = 0.95 .* ones(N) .+ rand(Uniform(-σ,σ), N)
    gs = ones(N) .+ rand(Uniform(-0.25,0.25), N)
    gw = zeros(N) 
    mx = 1.1ones(N) .+ rand(Uniform(-σ,σ), N)
    
    fy = ones(N,M)
    hy = -ones(N,M)
    
    λy = zeros(N,M)
    ωy = zeros(N,M)

    iy = zeros(M)
    oy = ones(M) .+ rand(Uniform(-σ,σ), M)

    return MiCRM_stability.exponential_params(gx,gs,gw,mx,fy,hy,λy,ωy,iy,oy)
end

function shuffle_community!(com)
    com.U[:] .= sample(com.U, com.N*com.M, replace = false)
    com.L[:] .= sample(com.L, com.N*com.M, replace = false)
end

function fmodu_rand(N,M,C) 
    #resource classes
    Mclass = 5
    #group sizes
    Min = M / Mclass
    Mout = M - Min
    Co = 0.5C + (0.5C * rand())
    Ci = (M*C - Mout*Co) / Min
    
    return(MiCRM_stability.modular_community(N,M,Mclass, 0.0, 0.0, Co, Ci, Co, Co, Ci), Co, Ci)
end

N,M = 50,100
C = 0.3
Ncom = 10000
Nrep = 100

k = [0]

λ_mat = zeros(Ncom, Nrep)
c_mat = zeros(Ncom, 2)
# p_mat = Array{Float64,4}(undef, Ncom, 4, Nrep, length(p_names))

#modularity
Threads.@threads for i = 1:Ncom
    k[1] += 1
    if k[1] % 100 == 0
        print("\r", k)
    end

    #allocate jacobian
    J = zeros(N+M,N+M)    
    com, ci, co = fmodu_rand(N,M,C) 
    c_mat[i,:] .= [ci,co]
    sp = MiCRM_stability.get_structural_params(com.U,com.L, com.N, com.M)
    
    for r = 1:Nrep
        J .= 0
        ep = get_exponential_parameters(com.N, com.M)
        p = MiCRM_stability.Parameters(com.N,com.M,sp,ep)      
            
        MiCRM_stability.jacobian!(p, J)
        λ_mat[i,r] = get_real(eigsolve(J, 1, (:LR))[1][1])    
    end
end

save("./Results/data/modularity_stabiltiy.jld2", Dict("l" => λ_mat, "c" => c_mat))