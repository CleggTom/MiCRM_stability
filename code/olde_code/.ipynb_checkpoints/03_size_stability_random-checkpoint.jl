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

function get_exponential_parameters(N,M)
    σ = 0.1
    gx = 0.95 .* ones(N) .+ rand(Uniform(-σ,σ), N)
    gs = ones(N) .+ rand(Uniform(-0.5,0.5), N)
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



#size and connectance
Nrep = 250
NC = 1
n = 40
m = 40

minS = log(25.1)
maxS = log(250.1)

N_vec = Int.(floor.(exp.(range(minS, maxS, length = n))))
M_vec = Int.(floor.(exp.(range(minS, maxS, length = m))));
C_vec = [0.2] #range(0.2,0.9, length = NC)
C = 0.2
λ_mat = zeros(n,m,1,NC,Nrep)
N_mat = similar(λ_mat)
M_mat = similar(λ_mat)
c_mat = similar(λ_mat)

frand(N,M) = MiCRM_stability.rand_community(N,M, C, Int(floor(M * C)), Int(floor(M * C))) 

models = [frand]

println("starting")
nrep = [0]


Threads.@threads for t = 1:(n*m)
    nrep[1] += 1
    i = ((t - 1) .÷ n) + 1
    j = ((t - 1) .% m) + 1

    N = N_vec[i]
    M = M_vec[j]
    
    print("\r", N, " ", M, "  ", nrep[1], "   ")

    J = zeros(N_vec[i] + M_vec[j], N_vec[i] + M_vec[j])
    for m = 1:1
        for c = 1:NC
            for k = 1:Nrep    

                com = models[m](N_vec[i], M_vec[j])
    
                sp = MiCRM_stability.get_structural_params(com.U,com.L, com.N, com.M)
                ep = get_exponential_parameters(com.N, com.M)
                p = MiCRM_stability.Parameters(com.N,com.M,sp,ep)      
                    
                MiCRM_stability.jacobian!(p, J)
                λ_mat[i,j,m,c,k] = get_real(eigsolve(J, 1, (:LR))[1][1])
                N_mat[i,j,m,c,k] = N
                M_mat[i,j,m,c,k] = M
                #     c_mat[i,j,m,c,k] = check_connected(com)
            end
        end
    end
end

save("./Results/data/size_stabiltiy_random2.jld2", Dict("N" => N_mat, "M" => M_mat, "l" => λ_mat, "c" => c_mat))