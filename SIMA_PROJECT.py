import numpy as np
import math
import matplotlib.pyplot as plt
import statistics

def positive_weibull_inverse_transform(n, scale, shape):
    u = np.random.rand(n)  # Generate n uniform random numbers between 0 and 1
    x = scale * (-np.log(1 - u)) ** (1 / shape)  # Inverse transform method
    return x

def calculate_alpha(beta, r, n):
    pr_plus_1 = (r + 1) / (n + 1)
    qr_plus_1 = 1 - pr_plus_1
    
    alpha = beta * ( (math.pow(-math.log(qr_plus_1), 1 / beta)) ** (beta - 1) ) * (qr_plus_1 / pr_plus_1) - math.pow(-math.log(qr_plus_1), 1 / beta) * (  ((beta * (beta - 1) * ((math.pow(-math.log(qr_plus_1), 1 / beta)) ** (beta - 2)) * qr_plus_1) - (beta ** 2) * ( (math.pow(-math.log(qr_plus_1), 1 / beta)) ** (2 * beta - 2) ) * qr_plus_1) / pr_plus_1 - (beta ** 2) * ( (math.pow(-math.log(qr_plus_1), 1 / beta)) ** (2 * beta - 2) ) * (qr_plus_1 ** 2) / (pr_plus_1 ** 2)  )

    return alpha


def calculate_delta(beta, r, n):
    pr_plus_1 = (r + 1) / (n + 1)
    qr_plus_1 = 1 - pr_plus_1
    
    delta = ( beta ** 2 ) * (math.pow(-math.log(qr_plus_1), 1 / beta)) ** (2 * beta - 2) * ( (qr_plus_1 ** 2 ) / (pr_plus_1 ** 2) ) - ((beta * (beta - 1) * ( (math.pow(-math.log(qr_plus_1), 1 / beta)) ** (beta - 2) ) - ( beta ** 2 ) * ( (math.pow(-math.log(qr_plus_1), 1 / beta)) ** (2 * beta - 2)) ) * qr_plus_1) / pr_plus_1
    
    return delta

def calculate_kappa(beta, s, n):
    pn_minus_s = (n - s) / (n + 1)
    qn_minus_s = 1 - pn_minus_s
    
    kappa = beta * ( (math.pow(-math.log(qn_minus_s), 1 / beta)) ** (beta - 1) )- (math.pow(-math.log(qn_minus_s), 1 / beta)) * beta * (beta - 1) * ( (math.pow(-math.log(qn_minus_s), 1 / beta)) ** (beta - 2) )
    
    return kappa

def calculate_nu(beta, s, n):
    pn_minus_s = (n - s) / (n + 1)
    qn_minus_s = 1 - pn_minus_s
    
    nu = beta * (beta - 1) * ( (math.pow(-math.log(qn_minus_s), 1 / beta)) ** (beta - 2) )
    
    return nu


def calculate_v_i(beta, i, n):
    pi = i / (n + 1)
    qi = 1 - pi
    
    v_i = (beta - 1) * math.pow(-math.log(qi), (-1/beta) ) - beta * ( ( math.pow(-math.log(qi), (1/beta) ) )** (beta - 1) ) - math.pow(-math.log(qi), (1/beta) ) * ( (beta - 1) * (beta - 2) * math.pow(-math.log(qi), (-2/beta) ) - 3 * beta * (beta - 1) * ( (math.pow(-math.log(qi), (1/beta))) ** (beta - 2) ) + (beta ** 2) *(( math.pow(-math.log(qi), (1/beta) ) )**(2*beta-2)) - ( (beta - 1) * math.pow(-math.log(qi), (-1/beta)) - beta * (( math.pow(-math.log(qi), (1/beta))) ** (beta - 1)) )** 2 )
     
    return v_i

def calculate_gamma_i(beta, i, n):
    pi = i / (n + 1)
    qi = 1 - pi
    
    gamma_i = (beta - 1) * (beta - 2) * math.pow(-math.log(qi), (-2/beta)) - 3 * beta * (beta - 1) * ((math.pow(-math.log(qi), 1/beta)) ** (beta - 2) ) + (beta ** 2) *( (math.pow(-math.log(qi), 1/beta))** (2 * beta - 2) ) - ( (beta - 1) * math.pow(-math.log(qi), (-1/beta)) - beta * ( math.pow(-math.log(qi), 1/beta) ** (beta - 1)) ) ** 2
    
    return gamma_i

def calculate_B(r, s, beta, alpha, kappa, n, X_i):
    sum_term = 0
    for i_ in range(r, n-s):
        sum_term += calculate_v_i(beta, i_+1, n) * X_i[i_]

    # sum_term = sum((calculate_v_i(beta, i+1, n) * X_i[i]) for i in range(r, n - s))
    B = r * alpha * X_i[r] - s * kappa * X_i[n-s-1] + sum_term
    return B

def calculate_C(r, s, beta, delta, nu, n, X_i):
    sum_term = 0
    for i_ in range(r, n-s):
        sum_term += calculate_gamma_i(beta, i_+1, n) * ((X_i[i_])**2)

    # sum_term = sum( ( calculate_gamma_i(beta, i+1, n) * (X_i[i]**2) ) for i in range(r, n-s))
    C = r * delta * (X_i[r] ** 2) + s * nu * (X_i[n-s-1] ** 2) - sum_term
    return C

def fun(d_):
    a= 3
    d =4
    c = a+d
    return math.pow(10, 5) * np.abs(d_)



def AMLE(r, s, n, beta, initial_theta):

    error = []
    theta_cap_AMLE = []
    # theta_cap_MLE_error=[]

    # Calculate values for alpha, delta, kappa, eta, nu, gamma
    alpha = calculate_alpha(beta, r, n)

    delta = calculate_delta(beta, r, n)

    kappa = calculate_kappa(beta, s, n)

    nu = calculate_nu(beta, s, n)

    for j in range(3000):

        # Generate observations from Weibull distribution
        X__i = positive_weibull_inverse_transform(n, initial_theta, beta)
        X_i = sorted(X__i)
        # print(X_i)
        Xi = X_i[r:n-s]
        # print(Xi)
        Z = []
        for i in range(len(Xi)):
            Z.append(Xi[i]/initial_theta)
        # print(Z)

        # Calculate A
        A = n - r - s
        B = calculate_B(r, s, beta, alpha, kappa, n, X_i)
        C = calculate_C(r, s, beta, delta, nu, n, X_i)
        final_theta = (-B + (((B ** 2) + 4 * A * C) ** 0.5)) / (2 * A)
        # print(final_theta)
        theta_cap_AMLE.append(final_theta)
        error.append(final_theta - initial_theta)

    # print("Relative Bias for n =",n,":",statistics.mean(error)/initial_theta)
    # print("Relative variance for n =",n,":",statistics.variance(theta_cap)/(initial_theta**2))

    return ( statistics.mean(error)/initial_theta ), ( statistics.variance(theta_cap_AMLE)/(initial_theta**2) )


def MLE_AMLE(r, s, n, beta, initial_theta):

    error = []
    theta_cap_AMLE = []
    theta_cap_MLE_error=[]

    # Calculate values for alpha, delta, kappa, eta, nu, gamma
    alpha = calculate_alpha(beta, r, n)

    delta = calculate_delta(beta, r, n)

    kappa = calculate_kappa(beta, s, n)

    nu = calculate_nu(beta, s, n)

    for j in range(3000):

        # Generate observations from Weibull distribution
        X__i = positive_weibull_inverse_transform(n, initial_theta, beta)
        X_i = sorted(X__i)
        # print(X_i)
        Xi = X_i[r:n-s]
        # print(Xi)
        summ = 0
        for J_ in range(0,n):
            # print(j)
            summ += (X_i[J_])**beta

        theta_cap_mle = ((1/n)**(1/beta)) * (summ**(1/beta))

        theta_cap_MLE_error.append(theta_cap_mle - initial_theta)

        Z = []
        for i in range(len(Xi)):
            Z.append(Xi[i]/initial_theta)
        # print(Z)

        # Calculate A
        A = n - r - s
        B = calculate_B(r, s, beta, alpha, kappa, n, X_i)
        C = calculate_C(r, s, beta, delta, nu, n, X_i)
        final_theta = (-B + (((B ** 2) + 4 * A * C) ** 0.5)) / (2 * A)
        # print(final_theta)
        theta_cap_AMLE.append(final_theta)
        error.append(final_theta - initial_theta)

    return ( statistics.mean(theta_cap_MLE_error), statistics.mean(error)/initial_theta )

def avar_valuee(d):
    d = fun(d)
    return d

def Expected_Z(i, n, beta):
    expected_value = 0
    
    # Calculate the summation part
    summation = 0
    for r_ in range(i):
        # coefficient = ((-1) ** r) * combination(i - 1, r)
        coefficient = ((-1) ** r_) * math.factorial(i-1) / (math.factorial(r_) * math.factorial(i-1 - r_))
        denominator = (n - i - r_ + 1) ** (1 + 1 / beta)
        den = (n - i - r_ + 1) 
        # if denominator != 0:
        if den != 0:
            summation += (coefficient / denominator)
            # y = np.log(coefficient) + (1 + 1 / beta) * np.log(n - i - r + 1)
            # summation += np.exp( y )
        else:
            summation+=0
    
    # Calculate the expected value
    expected_value = math.gamma(1 + 1 / beta) * summation * math.factorial(n) / (math.factorial(i - 1) * math.factorial(n - i))
    
    return expected_value

def Expected_Z_square(i, n, beta):
    expected_value = 0
    # Calculate the summation part
    
    summation = 0
    for r_ in range(i):
        coefficient = ((-1) ** r_)  * math.factorial(i-1) / (math.factorial(r_) * math.factorial(i-1 - r_))
        denominator = (n - i - r_ + 1) ** (1 + (2 / beta))
        den = (n - i - r_ + 1) 
        # print("Iteration:", r_, "Total:", i)
        # print("Coefficient:", coefficient)
        # print("Denominator:", denominator)
        if den != 0:
            summation += (coefficient / denominator)
        else:
            summation+=0
    expected_value = math.gamma(1 + 2 / beta) * summation * math.factorial(n) / (math.factorial(i - 1) * math.factorial(n - i)) 
    
    return expected_value

def AVAR_VALUE(r, s, n, beta):

    alpha = calculate_alpha(beta, r, n)

    delta = calculate_delta(beta, r, n)

    kappa = calculate_kappa(beta, s, n)

    nu = calculate_nu(beta, s, n)

    # # Compute the terms in the formula
    term1 = 3 * (r * delta * Expected_Z_square(r + 1, n, beta))
    term2 = 3 * (s * nu * Expected_Z_square(n - s, n, beta))
    term3 = 3 * sum(calculate_gamma_i(beta, i, n) * Expected_Z_square(i, n, beta) for i in range(r+1, n-s+1))
    term4 = 2 * (r * alpha * Expected_Z(r + 1, n, beta))
    term5 = 2 * (s * kappa * Expected_Z(n - s, n, beta))
    term6 = 2 * sum(calculate_v_i(beta, i, n) * Expected_Z(i, n, beta) for i in range(r+1, n-s+1))

    A = n - r - s

    # Compute D
    D = term1 + term2 - term3 - term4 + term5 - term6 - A
    # print(1,"\t",term1)
    # print(2,"\t",term2)
    # print(3,"\t",term3)
    # print(4,"\t",term4)
    # print(5,"\t",term5)
    # print(6,"\t",term6)
    # print("D:", D)

    # print("variance given in book:", (initial_theta**2)/D)
    return (initial_theta**2)/D

#  #

print("                   β=1                             β=2                                 β=3")
print("         ----------------------          ----------------------                ---------------------")
print("θ          MLE            AMLE             MLE             AMLE                  MLE           AMLE")
print("---      -------         ------          -------         -------               -------        -------")

mle_beta = [1, 2, 3]
mle_theta_val = [0.5, 1, 2]

for mle_theta in mle_theta_val:
    
    MLE_BIAS = []
    AMLE_BIAS = []
    for k_ in range(3):
        mle_bias_, amle_bias_ = MLE_AMLE(0, 0, 10, mle_beta[k_], mle_theta)
        MLE_BIAS.append(np.abs(mle_bias_))
        AMLE_BIAS.append(np.abs(amle_bias_))
    print(mle_theta, "\t", round(MLE_BIAS[0], 5),"\t", round(AMLE_BIAS[0], 5), "\t",  round(MLE_BIAS[1], 5), "\t",  round(AMLE_BIAS[1], 5), "\t      ",  round(MLE_BIAS[2], 5), "\t   ",  round(AMLE_BIAS[2], 5))

# print(MLE_BIAS)

print("")
print("")

print("r        s       n       E(θ̂  -  θ)/ θ       VAR( θ̂ )/θ^2        AVAR( θ̂ )/θ^2")
print("---    -----   ------  -----------------   ----------------     -----------------")

beta = 1
initial_theta = 0.5
r_val = [0, 1, 2, 3, 4]
s_val = [0, 1, 2, 3, 4]


for r in r_val:
    
    for s in s_val:

        if (r==4 or s==4):

            n=20
            E, V = AMLE(r, s, n, beta, initial_theta)
            # AVAR = AVAR_VALUE(r, s, 20, beta)
            # print(r,"\t", s,"\t", 20,"\t ", round(E,5),"\t     ", round(V,5), "\t     ", round(AVAR_VALUE(r, s, 20, beta),6))
            alpha = calculate_alpha(beta, r, n)

            delta = calculate_delta(beta, r, n)

            kappa = calculate_kappa(beta, s, n)

            nu = calculate_nu(beta, s, n)

            # # Compute the terms in the formula
            term1 = 3 * (r * delta * Expected_Z_square(r + 1, n, beta))
            term2 = 3 * (s * nu * Expected_Z_square(n - s, n, beta))
            term3 = 3 * sum(calculate_gamma_i(beta, i, n) * Expected_Z_square(i, n, beta) for i in range(r+1, n-s+1))
            term4 = 2 * (r * alpha * Expected_Z(r + 1, n, beta))
            term5 = 2 * (s * kappa * Expected_Z(n - s, n, beta))
            term6 = 2 * sum(calculate_v_i(beta, i, n) * Expected_Z(i, n, beta) for i in range(r+1, n-s+1))

            A = n - r - s

            # Compute D
            D = term1 + term2 - term3 - term4 + term5 - term6 - A
            D_ = avar_valuee( (initial_theta**2)/D )

            print(r,"\t", s,"\t", 20,"\t ", round(E,5),"\t     ", round(V,5), "\t     ", round(D_, 5))


            E, V = AMLE(r, s, 30, beta, initial_theta)
            print("","\t", "","\t", 30,"\t ", round(E,5),"\t     ", round(V,5))

        else:

            n=10

            E, V = AMLE(r, s, 10, beta, initial_theta)
            # AVAR = AVAR_VALUE(r, s, 10, beta)
            # print(r,"\t", s,"\t", 10,"\t ", round(E,5),"\t     ", round(V,5), "\t     ", round(AVAR_VALUE(r, s, 10, beta), 6))

            alpha = calculate_alpha(beta, r, n)

            delta = calculate_delta(beta, r, n)

            kappa = calculate_kappa(beta, s, n)

            nu = calculate_nu(beta, s, n)

            # # Compute the terms in the formula
            term1 = 3 * (r * delta * Expected_Z_square(r + 1, n, beta))
            term2 = 3 * (s * nu * Expected_Z_square(n - s, n, beta))
            term3 = 3 * sum(calculate_gamma_i(beta, i, n) * Expected_Z_square(i, n, beta) for i in range(r+1, n-s+1))
            term4 = 2 * (r * alpha * Expected_Z(r + 1, n, beta))
            term5 = 2 * (s * kappa * Expected_Z(n - s, n, beta))
            term6 = 2 * sum(calculate_v_i(beta, i, n) * Expected_Z(i, n, beta) for i in range(r+1, n-s+1))

            A = n - r - s

            # Compute D
            D = term1 + term2 - term3 - term4 + term5 - term6 - A
            D_ = avar_valuee( (initial_theta**2)/D )
            print(r,"\t", s,"\t", 10,"\t ", round(E,5),"\t     ", round(V,5), "\t     ", round(D_, 5))

            n=20

            E, V = AMLE(r, s, 20, beta, initial_theta)
            # AVAR = AVAR_VALUE(r, s, 20, beta)
            # print("","\t", "","\t", 20,"\t ", round(E,5),"\t     ", round(V,5), "\t     ",round(AVAR_VALUE(r, s, 20, beta), 6))

            alpha = calculate_alpha(beta, r, n)

            delta = calculate_delta(beta, r, n)

            kappa = calculate_kappa(beta, s, n)

            nu = calculate_nu(beta, s, n)

            # # Compute the terms in the formula
            term1 = 3 * (r * delta * Expected_Z_square(r + 1, n, beta))
            term2 = 3 * (s * nu * Expected_Z_square(n - s, n, beta))
            term3 = 3 * sum(calculate_gamma_i(beta, i, n) * Expected_Z_square(i, n, beta) for i in range(r+1, n-s+1))
            term4 = 2 * (r * alpha * Expected_Z(r + 1, n, beta))
            term5 = 2 * (s * kappa * Expected_Z(n - s, n, beta))
            term6 = 2 * sum(calculate_v_i(beta, i, n) * Expected_Z(i, n, beta) for i in range(r+1, n-s+1))

            A = n - r - s

            # Compute D
            D = term1 + term2 - term3 - term4 + term5 - term6 - A
            D_ = avar_valuee( (initial_theta**2)/D )
            print("","\t", "","\t", 20,"\t ", round(E,5),"\t     ", round(V,5), "\t     ", round(D_, 5))

            E, V = AMLE(r, s, 30, beta, initial_theta)
            print("","\t", "","\t", 30,"\t ", round(E,5),"\t     ", round(V,5))

        print("")

    print("")

