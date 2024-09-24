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


function get_exponential_parameters(N,M,L)
    σ = 0.1
    gx = 0.9ones(N) .+ rand(Uniform(-σ,σ), N)
    gs = ones(N) .+ rand(Uniform(-σ,σ), N)
    gw = -ones(N) .+ rand(Uniform(-σ,σ), N)
    mx = 1.1ones(N) .+ rand(Uniform(-σ,σ), N)
    
    fy = ones(N,M) #rand(Uniform(0,2)) .+ rand(Uniform(-σ,σ),N,M)
    hy = -ones(N,M) #rand(Uniform(-2,0)) .+ rand(Uniform(-σ,σ),N,M)
    
    λy = zeros(N,M) #rand(Uniform(-1,1)) .+ rand(Uniform(-σ,σ),N,M)
    ωy = - λy .* (L / (1 - L))

    iy = zeros(M) 
    oy = ones(M) .+ rand(Uniform(-σ,σ), M)

    return MiCRM_stability.exponential_params(gx,gs,gw,mx,fy,hy,λy,ωy,iy,oy)
end


N,M = 50,100
C = 0.3
Ncom = 100
Nrep = 100

k = [0]

λ_mat = zeros(Ncom, Nrep)
p_mat = Matrix{Any}(undef, Ncom, Nrep)
G_mat = zeros(Ncom, Nrep)


Threads.@threads for i = 1:Ncom      
    k[1] += 1
   #generate inital community
    p = 0.0
    L = 0.0
    Λ = 0.0
    T = 10 ^ rand(Uniform(-1,2.5))

    c = MiCRM_stability.attachment_community(N, M, C, T)

    J = zeros(c.N+c.M,c.N+c.M) 
    G = similar(J)
    x0 = zeros(c.N + c.M)
        
    #allocate jacobian
    
    d = [0]
    #loop over reps
        for j = 1:Nrep
        d[1] += 1

        #generate parameters
        sp = MiCRM_stability.get_structural_params(c.U, c.L, c.N, c.M, Λ = 0)
        ep = get_exponential_parameters(c.N, c.M, rand())
        p = MiCRM_stability.Parameters(c.N,c.M,sp,ep)

        #calculate stabiltiy
        MiCRM_stability.jacobian!(p, J)
        λ_mat[i,j] = get_real(eigsolve(J, 1, (:LR))[1][1]) 

        
        MiCRM_stability.gc!(G,J, x0)
        G_mat[i,j] = mean(G[1:N,1:N])
        G .= 0
        x0 .= 0

        A = MiCRM_stability.get_A(c)
        p_mat[i,j] = (MiCRM_stability.get_TC(A), L, T)

         println(k," ",d)
    end
end

save("./Results/data/crossfeeding_stabiltiy.jld2", Dict("l" => λ_mat, "G" => G_mat, "p" => p_mat))