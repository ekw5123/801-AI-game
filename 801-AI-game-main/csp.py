def DSSCSP(A, M, N):

    # Static list to store solutions
    S = []

    # Check if A is a Zero-Matrix
    if is_zero_matrix(A):
        # Add M to the solution set
        S.append(M)
        return S
    
    def argMax(vector):
        index, value = 0, vector[0]
        for i,v in enumerate(vector):
            if v > value:
                index, value = i,v
        return index

    # Find the column index i
    i = argMax(M) # i = argmaxi∈M(sum((j,Mi),j)

    # Create copies of A, M, and N for modifications
    A_prime, M_prime, N_prime = A.copy(), M.copy(), N.copy()
    M_prime[i] = 0

    # Check if A_prime * M_prime = N_prime is possible
    if is_possible(A_prime, M_prime, N_prime):
        # Reduce the matrices
        Reduce(A_prime, N_prime, i, M_prime[i])
        # Further reductions using DSS
        DSS(A_prime, M_prime, N_prime)
        # Recursive call
        DSSCSP(A_prime, M_prime, N_prime)

    # Create second set of copies of A, M, and N
    A_double_prime, M_double_prime, N_double_prime = A.copy(), M.copy(), N.copy()
    M_double_prime[i] = 1

    # Check if A_double_prime * M_double_prime = N_double_prime is possible
    if is_possible(A_double_prime, M_double_prime, N_double_prime):
        # Reduce the matrices
        Reduce(A_double_prime, N_double_prime, i, M_double_prime[i])
        # Further reductions using DSS
        DSS(A_double_prime, M_double_prime, N_double_prime)
        # Recursive call
        DSSCSP(A_double_prime, M_double_prime, N_double_prime)

    # Return the list of solutions
    return S

def Reduce(A, N, j, value):
    # Check if value is 0
    if value == 0:
        # Set column j of A to 0
        A[:, j] = 0
    elif value == 1:
        # Reduce the value of N[j] by 1
        N[j] -= 1
        # Set column j of A to 0
        A[:, j] = 0

def DSS(A, M, N):
    # Initialize the list to store determined variables
    Determined = []
    
    # Set a large number of loops for the iterations
    loops = 10  
    
    # Outer loop for a fixed number of iterations
    for _ in range(loops):
        # Iterate through each row in A
        for i in range(len(A)):
            # Count the number of variables in the current row
            Variables = sum(A[i])
            
            # If there are no variables in the row
            if Variables == 0:
                # Delete row i from A
                A.pop(i)
            
            # If the value of N[i] is 0
            elif N[i] == 0:
                # Iterate through each variable in the row
                for j in range(len(A[i])):
                    if A[i][j] == 1:
                        # Reduce with value 0
                        Reduce(A, N, j, 0)
                        # Add the variable to Determined with value 0
                        Determined.append((j, 0))
            
            # If the number of variables equals the value of N[i]
            elif Variables == N[i]:
                # Iterate through each variable in the row
                for j in range(len(A[i])):
                    if A[i][j] == 1:
                        # Reduce with value 1
                        Reduce(A, N, j, 1)
                        # Add the variable to Determined with value 1
                        Determined.append((j, 1))
    
    # Return the list of determined variables
    return Determined

def is_zero_matrix(A):
    # Check if the sum of all elements in A is 0
    return sum(sum(row) for row in A) == 0


def multiply_matrices(A, M):
    if len(A[0]) != len(M):
        raise ValueError("Incompatible dimensions for matrix multiplication")
    
    result = [0] * len(A)  # Initialize result vector with zeros
    for i in range(len(A)):
        result[i] = sum(A[i][j] * M[j] for j in range(len(M)))
    
    return result

def is_possible(A, M, N):
    # Iterate through each row in A
    try:
        # Multiply A'' and M'' to get the result
        result = multiply_matrices(A, M)
        # Compare the result with N''
        return result == N
    except ValueError:
        return False
