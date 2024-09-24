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
    gx = 0.9ones(N) .+ rand(Uniform(-σ,σ), N)
    gs = ones(N) .+ rand(Uniform(-σ,σ), N)
    gw = -ones(N) .+ rand(Uniform(-σ,σ), N)
    mx = 1.1ones(N) .+ rand(Uniform(-σ,σ), N)
    
    fy = rand(Uniform(0,2)) .+ rand(Uniform(-σ,σ),N,M)
    hy = rand(Uniform(-2,0)) .+ rand(Uniform(-σ,σ),N,M)
    
    λy = rand(Uniform(-2,2)) .+ rand(Uniform(-σ,σ),N,M)
    L = rand()
    ωy = - λy .* (L / (1 - L))

    iy = zeros(M) 
    oy = ones(M) .+ rand(Uniform(-σ,σ), M)

    return MiCRM_stability.exponential_params(gx,gs,gw,mx,fy,hy,λy,ωy,iy,oy)
end

N,M = 50,100
C = 0.2
Ncom = 1e3 |> Int
Nrep = 1

c = 0
k = [0]

λ_mat = zeros(Ncom, 4, Nrep)
p_mat = Array{Float64,4}(undef, Ncom, 4, Nrep,20)

Threads.@threads for i = 1:Ncom
    k[1] += 1
    if k[1] % 100 == 0
        print("\r", k)
    end
    J = zeros(N+M,N+M)

    for m = 1:4
        
        f = true
        c = 0
        while f
            c = models[m](N,M)
            f = !check_connected(c)
        end
        
        for j = 1:Nrep
            sp = MiCRM_stability.get_structural_params(c.U, c.L, c.N, c.M)
            ep = get_exponential_parameters(c.N, c.M)
            p = MiCRM_stability.Parameters(N,M,sp,ep)
            
            MiCRM_stability.jacobian!(p, J)
            λ_mat[i,m,j] = get_real(eigsolve(J, 1, (:LR))[1][1])
    
            p_mat[i,m,j,:] .= vcat(get_param_e_mean(p),get_param_s_mean(p))
        end
    end
end

println("\n", sum([l < 1e-5 for l = λ_mat[:]]) )

save("./Results/data/switching_stabiltiy.jld2", Dict("p" => p_mat, "l" => λ_mat))