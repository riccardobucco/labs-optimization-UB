import numpy as np
import time

def Newton_step(lamb0,dlamb,s0,ds):
    alp=1;
    idx_lamb0=np.array(np.where(dlamb<0))
    if idx_lamb0.size>0:
        alp = min(alp,np.min(-lamb0[idx_lamb0]/dlamb[idx_lamb0]))
        
    idx_s0=np.array(np.where(ds<0))
    if idx_s0.size>0:
        alp = min(alp,np.min(-s0[idx_s0]/ds[idx_s0]))
    return alp

def F(n,G,g,C,d,x,lam,s):
    m=2*n   
    a=np.array(np.dot(G,x)+g-np.dot(C,lam))
    b=np.array(s+d-np.dot(C.T,x))
    c=lam*s
    return np.concatenate((a,b,c))

def testproblem(n,maxiter=100,tol=1e-16):
    "Solves the test problem by solving the hole M_KKT by linag.solve"
    m=n+n
    G=np.eye(n); g=np.random.normal(0,1,size=n)
    C=np.block([[G,-G]]); 
    d=np.empty(m);d.fill(-10)
    x=np.zeros(n) #x0
    s=np.empty(m);s.fill(1) #s0
    lam=np.empty(m);lam.fill(1) #lambda0
    Mkkt=np.block([[G, -C, np.zeros((n,m))],
                   [-C.T,np.zeros((m,m)),np.eye(m,m)],
                   [np.zeros((m,n)),np.diag(s),np.diag(lam)]]) 
    k=0
    while k<maxiter:
        Fz=-F(n,G,g,C,d,x,lam,s)
        if np.linalg.norm(Fz[:n],2)<tol or  np.linalg.norm(Fz[n:n+m],2)<tol: 
            return x+g  
        #step 1 - solve dz:  Mkkt(z) dz= -F(z)
        dz=np.linalg.solve(Mkkt,Fz)   
        #step2 -sizecorrection substep    
        dlam=dz[n:n+m]; ds=dz[n+m:]
        alpha=Newton_step(lam,dlam,s,ds)
        #step 3
        mu=np.dot(s,lam)/np.float(m); muhat=np.dot((s+alpha*ds),(lam+alpha*dlam))/m; sigma= (muhat/mu)**3
        if np.abs(mu<tol):
            return x+g
        #step 4           
        Fz[n+m:n+m+m]=Fz[n+m:n+m+m]-ds*dlam+sigma*mu*np.ones(m)  
        dz=np.linalg.solve(Mkkt,Fz)
        #step5
        dlam=dz[n:n+m]; ds=dz[n+m:]
        alpha=Newton_step(lam,dlam,s,ds) 
        #step 6
        x+=0.95*alpha*dz[:n]; lam+=0.95*alpha*dz[n:n+m]; s+=0.95*alpha*dz[n+m:] 
        #Actualizing Mkkt      
        Mkkt[n+m:n+m+m,n:n+m]=np.diag(s); Mkkt[n+m:n+m+m,n+m:n+m+m]=np.diag(lam)
        k+=1
    print('warning: max mum iterats')
    return x+g

def testproblem_c3(n,maxiter=100,tol=1e-16):
    "Solves the test problem by solving the hole M_KKT by linag.solve"
    m=n+n
    G=np.eye(n); g=np.random.normal(0,1,size=n)
    C=np.block([[G,-G]]); 
    d=np.empty(m);d.fill(-10)
    x=np.zeros(n) #x0
    s=np.empty(m);s.fill(1) #s0
    lam=np.empty(m);lam.fill(1) #lambda0
    Mkkt=np.block([[G, -C, np.zeros((n,m))],
                   [-C.T,np.zeros((m,m)),np.eye(m,m)],
                   [np.zeros((m,n)),np.diag(s),np.diag(lam)]]) 
    k=0
    t0= timeit.default_timer()
    while k<maxiter:
        Fz=-F(n,G,g,C,d,x,lam,s)
        if np.linalg.norm(Fz[:n],2)<tol or  np.linalg.norm(Fz[n:n+m],2)<tol: 
            return np.linalg.norm(x+g), k, time.clock()-t0
        #step 1 - solve dz:  Mkkt(z) dz= -F(z)
        dz=np.linalg.solve(Mkkt,Fz)   
        #step2 -sizecorrection substep    
        dlam=dz[n:n+m]; ds=dz[n+m:]
        alpha=Newton_step(lam,dlam,s,ds)
        #step 3
        mu=np.dot(s,lam)/np.float(m); muhat=np.dot((s+alpha*ds),(lam+alpha*dlam))/m; sigma= (muhat/mu)**3
        if np.abs(mu<tol):
            return np.linalg.norm(x+g), k, time.clock()-t0
        #step 4           
        Fz[n+m:n+m+m]=Fz[n+m:n+m+m]-ds*dlam+sigma*mu*np.ones(m)  
        dz=np.linalg.solve(Mkkt,Fz)
        #step5
        dlam=dz[n:n+m]; ds=dz[n+m:]
        alpha=Newton_step(lam,dlam,s,ds) 
        #step 6
        x+=0.95*alpha*dz[:n]; lam+=0.95*alpha*dz[n:n+m]; s+=0.95*alpha*dz[n+m:] 
        #Actualizing Mkkt      
        Mkkt[n+m:n+m+m,n:n+m]=np.diag(s); Mkkt[n+m:n+m+m,n+m:n+m+m]=np.diag(lam)
        k+=1
    print('warning: max mum iterats')
    return np.linalg.norm(x+g), k, time.clock()-t0

def LDLT_fact(A):
    "Finds L downtriangular D diagonal st LDL.T=A. A must be symmetric"
    n= A.shape[0]
    L= np.eye(n)
    D= np.zeros(n)
    for i in range(n):
        D[i] = A[i, i] - np.dot(L[i, 0:i] ** 2, D[0:i])
        for j in range(i + 1, n):
            L[j, i] = (A[j, i] - np.dot(L[j, 0:i] * L[i, 0:i], D[0:i])) / D[i]
    return(L,D)
def solve_LDLT(l,d,b): 
    "solves Ax=b given L,D such that LDL.T=A, d is given as a vector"
    linalg.solve_triangular(l,b,lower=True,overwrite_b=True,unit_diagonal=True) #solves Lz=b
    b=b/d # Dy=z  
    linalg.solve_triangular(l.T,b,overwrite_b=True,unit_diagonal=True) #L.T x=y
    return b  

def testproblem_str1(n,maxiter=100,tol=1e-16):
    "Solves min xGx/2 +gx with constrains Cx<=d by itirior point. x,lam,s are seeds to start the algorithm"
    m=2*n
    k=0
    t0=time.clock()
    G=np.eye(n); g=np.random.normal(0,1,size=n)
    C=np.block([[G,-G]]); 
    d=np.empty(m);d.fill(-10)
    x=np.zeros(n) #x0
    s=np.empty(m);s.fill(1) #s0
    lam=np.empty(m);lam.fill(1) #lambda0
    A=np.block([[G, -C,], [-C.T,np.diag((-s/lam))]]) 

    while k<maxiter:
        Fz=-F(n,G,g,C,d,x,lam,s)
        if np.linalg.norm(Fz[:n],2)<tol or  np.linalg.norm(Fz[n:n+m],2)<tol: 
            return np.linalg.norm(x+g),k,time.clock()-t0   
        #strategy1 solve A dz=b by LDLT factorization
        r2hat=Fz[n:n+m]-(1/lam)*Fz[n+m:] #modifing r_2 according system A dz= b   
        L, D = LDLT_fact(A)
        #solve LDL.T dz=b 
        dz=solve_LDLT(L,D,np.concatenate((Fz[:n],r2hat)))
        dlam=dz[n:n+m];
        ds=(Fz[n+m:]-s*dlam)/lam 
        alpha=Newton_step(lam,dlam,s,ds)
        #step 3
        mu=np.dot(s,lam)/m; muhat=np.dot((s+alpha*ds),(lam+alpha*dlam))/m; sigma= (muhat/mu)**3
        if np.abs(mu<tol):
            return np.linalg.norm(x+g),k,time.clock()-t0
        #step 4     
        Fz[n+m:]+=-ds*dlam+sigma*mu*np.ones(m)   
        Fz[n:n+m]+=-(1/lam)*Fz[n+m:]
        dz=solve_LDLT(L,D,Fz[:n+m])
        #step5
        dlam=dz[n:n+m];ds= (Fz[n+m:]-s*dlam)/lam
        alpha=Newton_step(lam,dlam,s,ds) 
        #step 6
        x+=0.95*alpha*dz[:n]; lam+=0.95*alpha*dlam; s+=0.95*alpha*ds     
        A[n:n+m,n:n+m]=np.diag(-s/lam)
        k+=1
    print('warning: max mum iterats')
    return np.linalg.norm(x+g),k,time.clock()-t0

def G_hat(G,C,lam,s):
    n,m=np.shape(C)
    B=np.empty((n,m))
    for i in range(n): #using prop: M*D (D diagonal) = multiplying each row of M by diag(D)
        B[i,:]=C[i,:]*(lam/s)
    return(G+B.dot(C.T))

def solve_LLT(l,b): 
    "solves Ax=b given L such that L*L.T=A" 
    linalg.solve_triangular(l,b,lower=True,overwrite_b=True,unit_diagonal=False) #solves Lz=b
    linalg.solve_triangular(l.T,b,overwrite_b=True,unit_diagonal=False) #L.T x=y
    return b 

def testproblem_str2(n,maxiter=100,tol=1e-16):
    "Solves min xGx/2 +gx with constrains Cx<=d by itirior point. x,lam,s are seeds to start the algorithm"
    m=2*n
    k=0
    G=np.eye(n); g=np.random.normal(0,1,size=n)
    C=np.block([[G,-G]]); 
    d=np.empty(m);d.fill(-10)
    x=np.zeros(n) #x0
    s=np.empty(m);s.fill(1) #s0
    lam=np.empty(m);lam.fill(1) #lambda0
    #inicializa Ghat
    t0=time.clock()
    Ghat=G_hat(G,C,lam,s)
    while k<maxiter:
        Fz=-F(n,G,g,C,d,x,lam,s)
        if np.linalg.norm(Fz[:n],2)<tol or  np.linalg.norm(Fz[n:n+m],2)<tol: 
            return np.linalg.norm(x+g),k,time.clock()-t0  
        #step 1 - solve dz:  Ghat(z) dx= rhat, then find dlam,ds
        L=np.linalg.cholesky(Ghat)
        aux=(Fz[n+m:]-lam*Fz[n:n+m])/s
        rhat=Fz[:n]+C.dot(aux)
        dx=solve_LLT(L,rhat); aux2=np.dot(C.T,dx)
        ds=Fz[n:n+m]+aux2
        dlam=aux-(lam/s)*aux2  
        #step2 -sizecorrection substep
        alpha=Newton_step(lam,dlam,s,ds)
        #step 3
        mu=np.dot(s,lam)/np.float(m); muhat=np.dot((s+alpha*ds),(lam+alpha*dlam))/m; sigma= (muhat/mu)**3
        if np.abs(mu<tol):
            return np.linalg.norm(x+g),k,time.clock()-t0 
        #step 4
        Fz[n+m:]=Fz[n+m:]-ds*dlam+sigma*mu*np.ones(m)        
        aux=(Fz[n+m:]-lam*Fz[n:n+m])/s
        rhat=Fz[:n]+C.dot(aux)
        dx=solve_LLT(L,rhat); aux2=np.dot(C.T,dx)
        ds=Fz[n:n+m]+aux2
        dlam=aux-(lam/s)*aux2
        #step5
        alpha=Newton_step(lam,dlam,s,ds) 
        #step 6
        x+=0.95*alpha*dx; lam+=0.95*alpha*dlam; s+=0.95*alpha*ds
        #Actualizing Ghat
        Ghat=G_hat(G,C,lam,s)
        k+=1
    print('warning: max mum iterats')
    return np.linalg.norm(x+g),k,time.clock()-t0 

def F_gen(G,g,C,d,A,b,x,gam,lam,s):
    n=np.size(g)
    m=2*n   
    r1=np.dot(G,x)+g-np.dot(A,gam)-np.dot(C,lam) 
    r2=b-np.dot(A.T,x)
    r3=s+d-np.dot(C.T,x)
    r4=lam*s    
    return np.concatenate((r1,r2,r3,r4))

def general_case_lu(G,g,C,d,A,b,x,gam,lam,s,maxiter=100,tol=1e-16):
    "Solves min xGx/2 +gx with constrains Cx<=d by itirior point. x,lam,s are seeds to start the algorithm"
    n=np.size(g); p=np.size(b)
    m=2*n
    k=0
    t0= time.clock()
    #inicializa Mkkt
    Mkkt=np.block([[G, -A,-C, np.zeros((n,m))], [-A.T,np.zeros((p,p)),np.zeros((p,m)),np.zeros((p,m))],
                  [-C.T,np.zeros((m,p)),np.zeros((m,m)),np.eye(m,m)],[np.zeros((m,n)),np.zeros((m,p)),np.diag(s),np.diag(lam)]]) 
    while k<maxiter:
        Fz=-F_gen(G,g,C,d,A,b,x,gam,lam,s)
        if np.linalg.norm(Fz[:n],2)<tol or  np.linalg.norm(Fz[n+p:n+p+m],2)<tol: 
            return x,k,time.clock()-t0   
        #step 1 - solve dz:  Mkkt(z) dz= -F(z)
        dz=np.linalg.solve(Mkkt,Fz)   
        #step2 -sizecorrection substep    
        dlam=dz[n+p:n+p+m]; ds=dz[n+p+m:]
        alpha=Newton_step(lam,dlam,s,ds)
        #step 3
        mu=np.dot(s,lam)/m; muhat=np.dot((s+alpha*ds),(lam+alpha*dlam))/m; sigma= (muhat/mu)**3
        if np.abs(mu<tol):
            return x,k,time.clock()-t0
        #step 4 
        Fz[n+p+m:]+=-ds*dlam+sigma*mu*np.ones(m)  
        dz=np.linalg.solve(Mkkt,Fz)
        #step5
        dlam=dz[n+p:n+p+m]; ds=dz[n+p+m:]
        alpha=Newton_step(lam,dlam,s,ds) 
        #step 6
        x+=0.95*alpha*dz[:n];gam+=0.95*alpha*dz[n:n+p];lam+=0.95*alpha*dz[n+p:n+p+m];s+=0.95*alpha*dz[n+p+m:] 
        #Actualizing Mkkt
        Mkkt[n+p+m:, n+p:n+p+m]=np.diag(s); Mkkt[n+p+m:,n+p+m:]=np.diag(lam)
        k+=1
    print('warning: max mum iterats')
    return x,k,time.clock()-t0

def general_case_ldlt(G,g,C,d,A,b,x,gam,lam,s,maxiter=100,tol=1e-16):
    "Solves min xGx/2 +gx with constrains Cx<=d by itirior point. x,lam,s are seeds to start the algorithm"
    n=np.size(g); p=np.size(b)
    m=2*n
    k=0
    #inicializa M
    eps=np.empty(p); eps.fill(1e-10)
    M=np.block([[G, -A,-C], [-A.T,np.diag(eps),np.zeros((p,m))],[-C.T,np.zeros((m,p)),np.diag(-s/lam)]]) 
    while k<maxiter:
        print(k)
        Fz=-F_gen(G,g,C,d,A,b,x,gam,lam,s)
        if np.linalg.norm(Fz[:n],2)<tol or  np.linalg.norm(Fz[n+p:n+p+m],2)<tol: 
            return x,k   
        #step 1 - solve dz:  M dz= -F(z) by means of LDLT
        r3hat=Fz[n+p:n+p+m]-(Fz[n+p+m:]/lam) #modifing r_3 according system M dz= b  
        print(n+p+m,np.linalg.matrix_rank(M))
        L,D=LDLT_fact(M)
        dz=solve_LDLT(L,D,np.concatenate((Fz[:n+p],r3hat))) 
        #step2 -sizecorrection substep    
        dlam=dz[n+p:]; ds=Fz[n+p+m:]/lam-s*dlam
        alpha=Newton_step(lam,dlam,s,ds)
        #step 3
        mu=np.dot(s,lam)/np.float(m); muhat=np.dot((s+alpha*ds),(lam+alpha*dlam))/m; sigma= (muhat/mu)**3
        if np.abs(mu<tol):
            return x,k
        #step 4           
        Fz[n+p+m:]+=-ds*dlam+sigma*mu*np.ones(m)  
        Fz[n+p:n+p+m]+= -Fz[n+p+m:]/lam   
        dz=solve_LDLT(L,D,Fz[:n+p+m]) 
        #step5
        dlam=dz[n+p:]; ds=Fz[n+p+m:]/lam -s*dlam
        alpha=Newton_step(lam,dlam,s,ds) 
        #step 6
        x+=0.95*alpha*dz[:n]; gam+=0.95*alpha*dz[n:n+p]; lam+=0.95*alpha*dlam; s+=0.95*alpha*ds
        #Actualizing M
        M[n+p:,n+p:]=np.diag(-s/lam)
        k+=1
    print('warning: max mum iterats')
    return x,k

#read data
def read_file_M(file_path,n,m):
    "read and returns the matrix of dimension (n,m) contained in the  file"
    file=open(file_path,'r')
    F=file.readlines()
    M=np.zeros((n,m))
    for line in F:
        i,j,x=line.split()
        M[int(i)-1,int(j)-1]=float(x)  
    file.close()
    return (M)
def read_file_V(file_path,n):
    "read and returns the vector of lenght n contained in the  file"
    file=open(file_path,'r')
    F=file.readlines()
    M=np.zeros(n)
    for line in F:
        i,x=line.split()
        M[int(i)-1]=float(x)  
    file.close()
    return (M)