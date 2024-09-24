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
    gx = 0.95 .* ones(N) .+ rand(Uniform(-σ,σ), N)
    gs = ones(N) .+ rand(Uniform(-σ,σ), N)
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

#calculate stabiltiy
function calc_stab(c, Nrep)
    λ = zeros(Complex, Nrep)
    sp = MiCRM_stability.get_structural_params(c.U, c.L, c.N, c.M) 
    J = zeros(c.N + c.M, c.N + c.M)
    for k = 1:Nrep
        #get params and stabiltiy
        ep = get_exponential_parameters(c.N, c.M)
        p = MiCRM_stability.Parameters(c.N,c.M,sp, ep) 
        MiCRM_stability.jacobian!(p, J)
        λ[k] = eigsolve(J, 1, (:LR))[1][1]
    end

    return(λ)
end

N = 25
M = 100
C = 0.2
Ncom = 1000
Nrep = 500
Nshuffle = 2

J = zeros(N+M,N+M)

λ_mat = zeros(Ncom, Nshuffle, Nrep)
c_mat = Array{MiCRM_stability.Community, 2}(undef, Ncom, Nshuffle)

k = [0]

Threads.@threads for i = 1:Ncom
    k[1] += 1
    print("\r", k)
    
    #generate communites - assert it is connected
    begin
        f = true
        c = rand_com(N,M,C)
        
        while f
            c = rand_com(N,M,C)
            f = !check_connected(c)
        end
        
        # c.U .= c.U .* rand(N,M)
        # c.L .= c.L .* rand(N,M)
    end

    for j = 1:Nshuffle
        # #get struct
        c_mat[i,j] = deepcopy(c)
        
        #shuffle
        c_vec = Vector{MiCRM_stability.Community}(undef, 1001)
        c_vec[1] = deepcopy(c)
        for s = 2:1000
            dis_con = true
            while dis_con
                shuffle!(c)
                if check_connected(c)
                        dis_con = false
                else
                     c = deepcopy(c_vec[s-1])
                end
            end
            c_vec[s] = deepcopy(c)
        end
    end
end

#calculate stabiltiy
λ = zeros(Ncom, Nshuffle)
λ2 = zeros(Ncom, Nshuffle)

k = [0]
Threads.@threads for i = 1:Ncom
    k[1] += 1
    for j = 1:Nshuffle
        print(k, "\r")
        λ[i,j] = mean(get_real.(calc_stab(c_mat[i,j], Nrep)).< 0)
    end
end

save("./Results/data/niche_stabiltiy.jld2", Dict("l" => λ))