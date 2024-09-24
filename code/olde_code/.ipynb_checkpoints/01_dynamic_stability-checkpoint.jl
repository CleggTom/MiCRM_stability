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
    gx = rand(Uniform(0.5, 1.0)) .+ rand(Uniform(-σ,σ), N)
    gs = rand(Uniform(0,  3)) .+ rand(Uniform(-σ,σ), N)
    gw = rand(Uniform(-3, 0)) .+ rand(Uniform(-σ,σ), N)
    mx = rand(Uniform(0.5, 1.0)) .+ rand(Uniform(-σ,σ), N)
    
    fy = ones(N,M)
    hy = -ones(N,M)
    
    λy = zeros(N,M) 
    ωy = zeros(N,M)


    iy = rand(Uniform(0.0,1.0)) .+ rand(Uniform(-σ,σ), M)
    oy = rand(Uniform(0.5,1.5)) .+ rand(Uniform(-σ,σ), M)

    return MiCRM_stability.exponential_params(gx,gs,gw,mx,fy,hy,λy,ωy,iy,oy)
end

N,M = 50,100
C = 0.2
Ncom = 1e2 |> Int
Nrep = 1

k = [0]

p_names = vcat(fieldnames(MiCRM_stability.exponential_params)..., fieldnames(MiCRM_stability.structural_params)...)

λ_mat = zeros(Ncom, 4, Nrep)
p_mat = Array{Float64,4}(undef, Ncom, 4, Nrep, length(p_names))
G_mat = zeros(Ncom, 4, Nrep)


Threads.@threads for i = 1:Ncom
    k[1] += 1
    # if k[1] % 100 == 0
        print("\r", k)
    # end

    for m = 1:1
        #generate inital community
        f = true
        c = 0
        
        while f
            c = models[m](N,M)      
            f = !check_connected(c)
        end
    
        #allocate jacobian
        J = zeros(c.N+c.M,c.N+c.M) 
        G = similar(J)
        #loop over reps
        for j = 1:Nrep
            
            #generate parameters
            sp = MiCRM_stability.get_structural_params(c.U, c.L, c.N, c.M)
            ep = get_exponential_parameters(c.N, c.M)
            p = MiCRM_stability.Parameters(c.N,c.M,sp,ep)
    
            #calculate stabiltiy
            MiCRM_stability.jacobian!(p, J)
            λ_mat[i,m,j] = get_real(eigsolve(J, 1, (:LR))[1][1]) 
    
            #save parameters used
            p_mat[i,m,j,:] .= vcat(get_param_e_mean(p), get_param_s_mean(p))

            MiCRM_stability.gc!(G,J)
            G_mat[i,m,j] = mean(G[1:N,1:N])
            G .= 0
        end
    end
end

save("./Results/data/dynamic_stabiltiy.jld2", Dict("p" => p_mat, "l" => λ_mat, "G" => G_mat))