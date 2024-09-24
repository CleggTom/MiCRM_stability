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


function get_real(x::T) where T <: AbstractFloat
    x
end

get_real(x::Complex) = x.re


frand(N,M) = MiCRM_stability.rand_community(N,M,C, 10, 10) 
fnich(N,M) = MiCRM_stability.niche_community(N,M,C) 

function fmodu(N,M) 
    #resource classes
    Mclass = 5
    #group sizes
    Min = M / Mclass
    Mout = M - Min
    Co = 0.1
    Ci = (M*C - Mout*Co) / Min
    
    return MiCRM_stability.modular_community(N,M,Mclass, 0.0, 0.0, Co, Ci, Co, Co, Ci) 
end

fnest(N,M) = MiCRM_stability.nested_community(N,M,-rand(), 2rand(), 10, 10) 

models = [frand,fnich,fmodu,fnest]

function shuffle!(com::MiCRM_stability.Community)
    U,L = com.U, com.L
    N,M = com.N, com.M

    #U
    if rand() < 0.5
        ind = findall(com.U .!= 0)
        #sample links
        f = rand(ind)
        #dont assign upatke to last node
        t = rand(CartesianIndices((1:N, 1:(M-1))))

        tmp = U[f]
        U[f] = U[t]
        U[t] = tmp
    else
        ind = findall(com.L .!= 0)
        #sample links
        f = rand(ind)
        #dont assign upatke to last node
        t = rand(CartesianIndices((1:N, 2:M)))

        tmp = L[f]
        L[f] = L[t]
        L[t] = tmp
    end
        
   
end

#network structure functions
function get_A(com)
    N,M = com.N, com.M
    A = zeros(N+M, N+M)
    
    for i = 1:N
        for j = 1:M
            A[N + j, i] = com.U[i,j]
            A[i, N + j] = com.L[i,j]
        end
    end
    
    return(A)
end

function get_A_undir(com::MiCRM_stability.Community)
     N,M = com.N, com.M
    A = zeros(N+M, N+M)
    
    for i = 1:N
        for j = 1:M
            A[N + j, i] = com.U[i,j] + com.L[i,j]
            A[i, N + j] = com.L[i,j] + com.U[i,j]
        end
    end
    
    return(A)

end
    
function check_connected(com)
    A = get_A(com)
    g = SimpleWeightedDiGraph(A)
    return(is_connected(g))

    # return(sum(abs.(eigen(B).values) .< 1e-10) == 1 & all(sum(com.U,dims=2) .> 0))
end

#get trophic levels
function get_TL(A)
    v = sum(A, dims = 2)[:]
    v[v .== 0] .= 1
    Λ = diagm(v) - A
    
    if rank(Λ) >= length(A[1,:])
        s = inv(Λ) * v
        return s
    else
        return 0
    end
end

function get_TC(A)
    s = get_TL(A)

    if s != 0  
        S = length(s)
        x = [s[i] - s[j] for i = 1:S, j = 1:S]
        return std(x[x .!= 0])
    else
        return 0
    end
end

#plotting layout
function mylayout(g, N, M, A)
    #get avg for consumers
    con_indx = [findall(A[(N+1):end,i] .!= 0) for i = 1:N]
    con_wt = [filter(x -> x != 0, A[(N+1):end, i]) for i = 1:N]
    con_wt = [con_wt[i] ./ sum(con_wt[i]) for i = 1:N]
    con_mean = [sum(con_indx[i] .* con_wt[i]) for i = 1:N]
    
    sN = sum(A, dims = 2)[1:N]

    
    xs = vcat(con_mean, 1:M)
    ys = vcat(ones(N), fill(0.0, M))
   

    return Point.(zip(xs, ys))
end


function plot_com!(com, ax; ns = 10, as = 0.1)
    A = get_A(com)
    c = vcat(fill(:red,com.N),fill(:blue,com.M))
    c[com.N+1] = :green
    c[end] = :green

    ns = ns .* ones(com.N + com.M)
    as = as .* sum(A .> 0)
    
    g = SimpleWeightedDiGraph(A)
    ec = [(:black, max(i.weight,0.1)) for i = collect(edges(g))]
    layout = Spring()#mylayout(g, N, M, A)
    graphplot!(ax, g, layout = layout, node_color = c, node_size = ns, edge_color = ec, arrow_size = as)
end

# Grainyness
function get_n(α,S)
    dG = Pareto(α,1)
    G = rand(dG, S)
    n = [sum(G[1:i]) / sum(G) for i = 1:S]
    return(n)
end