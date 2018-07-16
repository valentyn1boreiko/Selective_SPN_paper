using Base,Distributions
MI(x,y,base) = (res=0; N=length(x) ; x_un = unique(x); y_un = unique(y); x_count = [countnz(x.==xx) for xx in x_un]; y_count = [countnz(y.==yy) for yy in y_un]; str_ar = [string(x[i],y[i]) for i in 1:N]; str_un = unique(str_ar); str_count = [countnz(str_ar.==x) for x in str_un]; for i in str_un n_x = x_count[findin(x_un,parse(Int64,i[1]))]; n_y = y_count[findin(y_un,parse(Int64,i[2]))]; count = str_count[findin(str_un,[i])][1]; println("Current pair and count",i," ",count); println("x_n and y_n",n_x," ",n_y) ; println(x_count[findin(parse(Int64,i[1]),x_un)]); println("computation: sum = sum +",count,"*(", (count/N),"*log(",base,",",(count/N),"/(",n_x[1],"*",n_y[1],")/",N^2 ); res=res+count*( (count/N) * log(base,(count/N)/((n_x[1]*n_y[1])/(N^2)) )) end; return res;)
MI([0,1,0],[1,0,1],e)
show(2)
2^2
length([1,2])
a = [1,2,4,4,4,2,7,5,1]
sort_a = sort(a)
num =  unique(a)
[2][1]
num_count = [countnz(a.==x) for x in num]
i="45"
parse(Int64,"2")
num_count[findin(int(i[1]),num)]
show(num_count)
log(10,2)
str_c = [string(a[i],b[i]) for i in 1:length(a)]
findin(str_c,["12"])
un_str = unique(str_c)
str_count = [countnz(str_c.==x) for x in un_str]
b = a+1

findin(a,1)
c = hcat(a,b)
C = [(x,y) for x in a, y in b]
c = diag(C)
num_c = unique(diag(C))


string(2,3)

[(i,j) for (i,j) in c for (x,y) in num_c if i==x && j==y]
[countnz(string(c).==string(x,y)) for (x,y) in num_c]
(diag(C),(1,2))
[(x,y) for x,y in c]
[countnz(a.==x) for x in c]
