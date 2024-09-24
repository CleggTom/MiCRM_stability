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
    for i = 1:com.N
        for j = (i+1):com.N
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


function get_L_overlap(com)
    U = com.U
    L = com.L .- com.U
    v = 0
    for i = 1:com.N
        for j = 1:com.N
            t = sum(U[i,:] .* L[j,:])
            b = sqrt(sum(U[i,:] .^ 2) * sum(L[j,:] .^ 2))
            if b > 0
                v += (t/b)
            end 
        end
    end
    
    return v / (com.N*com.N)
end

#size and connectance
N = 50
M = 100

Ncom = 10000
Nrep = 100

λ_mat = zeros(Ncom, Nrep)
overlap_mat = zeros(Ncom,2)

fspec(N,M,C) = MiCRM_stability.specialist_community(N, M, 0.1, 0.1rand(), rand(Uniform(-10,10)), Int(floor(M * C)), Int(floor(M * C)))

println("starting")
nrep = [0]
k = [0]
Threads.@threads for t = 1:Ncom
    k[1] += 1
    if k[1] % 100 == 0
        print("\r", k)
    end

    
    J = zeros(N+M,N+M)
    com = fspec(N,M, 0.2)
    sp = MiCRM_stability.get_structural_params(com.U,com.L, com.N, com.M)
    overlap_mat[t,1] = get_U_overlap(com)
    overlap_mat[t,2] = get_L_overlap(com)
    
    for r = 1:Nrep
        J .= 0
        ep = get_exponential_parameters(com.N, com.M)
        p = MiCRM_stability.Parameters(com.N,com.M,sp,ep)      
            
        MiCRM_stability.jacobian!(p, J)
        λ_mat[t,r] = get_real(eigsolve(J, 1, (:LR))[1][1])
        

    end
end

save("./Results/data/interaction_stabiltiy.jld2", Dict("l" => λ_mat, "int" => overlap_mat))