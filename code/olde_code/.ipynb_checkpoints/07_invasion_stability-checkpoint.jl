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

function get_U_overlap(com)
    U = com.U
    v = 0
    k = 0
    for i = 1:N
        for j = (i+1):N
            t = sum(U[i,:] .* U[j,:])
            b = sqrt(sum(U[i,:] .^ 2) * sum(U[j,:] .^ 2))
            if b > 0
                v += (t/b)
                k += 1
            end 
        end
    end
    
    return v / k
end

function add_consumer(com, I)
    #calculate exploitation
    κU, κL = -I * 10.0, I * 10.0
    exp_vec = sum(com.U,dims=1)[:]
    exp_vec = exp_vec ./ maximum(exp_vec)
    pU = [p > 0 ? p^κU : 0 for p = exp_vec]
    
    U_new = fill(0.01, N+1, M)
    U_new[1:N,:] .= com.U
    
    k = rand(Binomial(M, 0.1))
    U_new[end, sample(collect(1:M), Weights(pU), k, replace = false)] .+= 0.5
    
    #calculate leakage
    # leak_vec = sum(com.L, dims = 1)[:]
    # leak_vec = leak_vec ./ maximum(leak_vec)
    pL = [p > 0 ? p^κL : 0 for p = exp_vec]
    
    L_new = fill(0.01, N+1, M)
    L_new[1:N,:] .= com.L
    
    k = rand(Binomial(M, 0.1))
    L_new[end, sample(collect(1:M), Weights(pL), k , replace = false)] .+= 0.5
    
    new_com = MiCRM_stability.Community(N+1, M, U_new, L_new)
end

#size and connectance
N = 50
M = 100

Ncom = 1000
Nrep = 100

λ_mat = zeros(Ncom, Nrep,2)
I_vec = zeros(Ncom)

frand(N,M,C) = MiCRM_stability.rand_community(N,M, rand(), Int(floor(M * C)), Int(floor(M * C))) 
fspec(N,M,C) = MiCRM_stability.specialist_community(N, M, 0.1, 0.1rand(), Int(floor(M * C)), Int(floor(M * C)))

println("starting")
nrep = [0]
k = [0]

Threads.@threads for t = 1:Ncom
    k[1] += 1
    if k[1] % 100 == 0
        print("\r", k)
    end
    
    J1 = zeros(N+M,N+M)
    com = fspec(N,M, 0.2)

    sp = MiCRM_stability.get_structural_params(com.U,com.L, com.N, com.M)
    
    for r = 1:Nrep
        J1 .= 0
        ep = get_exponential_parameters(com.N, com.M)
        p = MiCRM_stability.Parameters(com.N,com.M,sp,ep)      
            
        MiCRM_stability.jacobian!(p, J1)
        λ_mat[t,r, 1] = get_real(eigsolve(J1, 1, (:LR))[1][1])
    end

    J2 = zeros(N+M+1,N+M+1)
    I_vec[t] = rand(Uniform(-1,1))
    com2 = add_consumer(com,  I_vec[t])
    
    sp = MiCRM_stability.get_structural_params(com2.U,com2.L, com2.N, com2.M)
    
    for r = 1:Nrep
        J2 .= 0
        ep = get_exponential_parameters(com2.N, com2.M)
        p = MiCRM_stability.Parameters(com2.N,com2.M,sp,ep)      
            
        MiCRM_stability.jacobian!(p, J2)
        λ_mat[t,r, 2] = get_real(eigsolve(J2, 1, (:LR))[1][1])
    end
    
end

save("./Results/data/invasion_stabiltiy.jld2", Dict("l" => λ_mat, "I" => I_vec))