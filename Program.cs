using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

// Created by Massimo Di Pierro in Python - Adapted to C# by Christin Mirabelli and Mike Schmitt for CSC431.

namespace CSC431_Final_Project
{
    class Program
    {
        /* This will be for running tests. */
        static void Main(string[] args)
        {
            //Numeric n = new Numeric();
            
            Matrix m1 = new Matrix(4, 3, 2);
            Matrix m2 = new Matrix();
            Matrix m3 = new Matrix(3, 3, 4.5);
            Matrix m4 = new Matrix(6, 2, 7);
            Matrix m5 = new Identity(3, 4, 7); 
            double[] v = new double[] {4, 3, 2, 1};
            Matrix m6 = new Diagonal(v);
            List <double[]> m7 = new List<double[]>();

            //Console.WriteLine("Definition is Matrix(4, 3): " + m1);
            //Console.WriteLine("\nDefinition is Matrix(): " + m2);
            //Console.WriteLine("\nDefinition is Matrix(3, 3, 4.5): " + m3);
            //Console.WriteLine("\nDefinition is Matrix(6, 2, 7): " + m4);
            //m4.setElement(1, 1, 2222);
            //Console.WriteLine(m1*m3);
            //Console.WriteLine(m5);

            //Console.WriteLine(m1*m3);
            //Console.WriteLine(m1*m2);
            //Console.WriteLine(m6);
            m7 = m6.as_list(m6);
            Console.WriteLine(m7[0][0]);
            Console.WriteLine(m7[0][1]);
            Console.WriteLine(m7[0][2]);
            Console.WriteLine(m7[0][3]);
            Console.WriteLine(m7[1][0]);
            Console.WriteLine(m7[1][1]);
            Console.WriteLine(m7[1][2]);
            Console.WriteLine(m7[1][3]);
            Console.WriteLine(m7[2][0]);
            Console.WriteLine(m7[2][1]);
            Console.WriteLine(m7[2][2]);
            Console.WriteLine(m7[2][3]);
            Console.WriteLine(m7[3][0]);
            Console.WriteLine(m7[3][1]);
            Console.WriteLine(m7[3][2]);
            Console.WriteLine(m7[3][3]);
            Console.ReadLine();
        }
    }

    class Function
    {
        public String full_function { get; set; }

        public Function(String full_function)
        {
            this.full_function = full_function;
        }
    }



    class Identity : Matrix
    {
        /* Constructor: Identity 
         * Purpose: Constructs an n x n Matrix with the diagonal elements equal to the same value.  All non-diagonal elements are equal as well.
         *          Inherits from the Matrix class.
         * Parameters:  rows - the integer number of rows and columns.
         *              one - the value of all diagonal elements
         *              fill - the value to be used to fill the non-diagonal elements of the identity matrix
         */
        public Identity (int rows = 1, double one = 1.0, double fill = 0.0)
        {
            this.rows = rows;
            this.cols = rows;
            data = new double[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    if (i == j)
                        data[i, j] = one;
                    else
                        data[i, j] = fill;
                }
            }
        }
    }

    class Diagonal : Matrix
    {
        /* Constructor: Diagonal 
         * Purpose: Constructs an n x n Matrix with the i-th diagonal element equal to the i-th value of the passed array.
         *          Inherits from the Matrix class.
         * Parameters:  d - array of elements to populate in diagonal elements of Matrix. M[i, i] = d[i]
         */
        public Diagonal(double[] d)
        {
            this.rows = d.Length;
            this.cols = d.Length;
            data = new double[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    if (i == j)
                        data[i, j] = d[i];
                    else
                        data[i, j] = 0.0;
                }
            }
        }
    }

    class Matrix
    {
        /* ??? should these be public or private ??? */
        public int rows, cols;
        public double[,] data; /* ??? Does this need to be type Object ??? */
        //private double fill;

        /* Constructor: Matrix
         * Purpose: 
         * Parameters:  rows - the integer number of rows
         *              cols - the integer number of columns
         *              fill - the value or callable to be used to fill the matrix
         */
        public Matrix(int rows = 1, int cols = 1, double fill = 0.0, bool optimize = false /* ??? What does optimize do ??? */)
        {
            this.rows = rows;
            this.cols = cols;
            data = new double[rows, cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    data[i, j] = fill;
        }

        public void setElement(int i, int j, double value) { data[i, j] = value; }
        public double getElement(int i, int j) { return data[i, j]; }

        /* Override: ToString
         * Purpose: Prints out the full string of a Matrix in a readable row by row format.
         * Parameters: None
         * Returns: String output.
         */
        public override string ToString()
        {
            String mFormat = "[";

            for (int i = 0; i < rows; i++)
            {
                if (i != 0) mFormat += ", ";
                mFormat += "[";
                for (int j = 0; j < cols; j++)
                {
                    if (j != 0) mFormat += ", ";
                    mFormat += data[i, j].ToString();
                }
                mFormat += "]";
            }
            mFormat += "]";
            return mFormat;
        }

     /* def __init__(self,rows=1,cols=1,fill=0.0,optimize=False):
        """
        Constructor a zero matrix
        Parameters
        - rows: the integer number of rows
        - cols: the integer number of columns
        - fill: the value or callable to be used to fill the matrix
        """
        self.rows = rows
        self.cols = cols
        if callable(fill):
            self.data = [fill(r,c) for r in xrange(rows) for c in xrange(cols)]
        else:
            self.data = [fill for r in xrange(rows) for c in xrange(cols)]
        if optimize:
            import array
            self.data = array.array('d',self.data)
        
        

    def __getitem__(self,(i,j)):
        return self.data[i*self.cols+j]

    def __setitem__(self,(i,j),value):
        self.data[i*self.cols+j] = value

    def row(A,i):
        return Matrix(self.cols,1,fill=lambda r,c: A[i,c])

    def col(self,i):
        return Matrix(self.rows,1,fill=lambda r,c: A[r,i])
        */

        /* Function: as_list
         * Purpose: 
         * Parameters: 
         * Returns: 
         */
        /* ??? Can this instead be a function defined within Matrix as as_list() ??? */
        public List<double[]> as_list(Matrix A)
        {
            List <double[]> toList = new List<double[]>();
            double[] values;
            
            for (int r = 0; r < A.rows; r++)
            {
                values = new double[A.cols];

                for (int c = 0; c < cols; c++)
                    values[c] = A.getElement(r, c);
                    
                toList.Add(values);
            }
            return toList;
        }
        

        /*
    def __str__(self):
        return str(self.as_list())

 
    @staticmethod
    def diagonal(d):
        M = Matrix(len(d),len(d))
        for i,e in enumerate(d): M[i,i] = e
        return M

    @staticmethod
    def from_list(v):
        "builds a matrix from a list of lists"
        return Matrix(len(v),len(v[0]),fill=lambda r,c: v[r][c])

    def __add__(A,B):
        """
        Adds A and B element by element, A and B must have the same size
        Example
        >>> A = Matrix.from_list([[4,3.0], [2,1.0]])
        >>> B = Matrix.from_list([[1,2.0], [3,4.0]])
        >>> C = A + B
        >>> print C
        [[5, 5.0], [5, 5.0]]
        """
        n, m = A.rows, A.cols
        if not isinstance(B,Matrix):
            if n==m:
                B = Matrix.identity(n,B)
            elif n==1 or m==1:
                B = Matrix(n,m,fill=B)
        if B.rows!=n or B.cols!=m:
            raise ArithmeticError, "Incompatible dimensions"
        C = Matrix(n,m)
        for r in xrange(n):
            for c in xrange(m):
                C[r,c] = A[r,c]+B[r,c]
        return C

    def __sub__(A,B):
        """
        Adds A and B element by element, A and B must have the same size
        Example
        >>> A = Matrix.from_list([[4.0,3.0], [2.0,1.0]])
        >>> B = Matrix.from_list([[1.0,2.0], [3.0,4.0]])
        >>> C = A - B
        >>> print C
        [[3.0, 1.0], [-1.0, -3.0]]
        """
        n, m = A.rows, A.cols
        if not isinstance(B,Matrix):
            if n==m:
                B = Matrix.identity(n,B)
            elif n==1 or m==1:
                B = Matrix(n,m,fill=B)
        if B.rows!=n or B.cols!=m:
            raise ArithmeticError, "Incompatible dimensions"
        C = Matrix(n,m)
        for r in xrange(n):
            for c in xrange(m):
                C[r,c] = A[r,c]-B[r,c]
        return C
    def __radd__(A,B): #B+A
        return A+B
    def __rsub__(A,B): #B-A
        return (-A)+B
    def __neg__(A):
        return Matrix(A.rows,A.cols,fill=lambda r,c:-A[r,c])

    def __rmul__(A,x):
        "multiplies a number of matrix A by a scalar number x"
        import copy
        M = copy.deepcopy(A)
        for r in xrange(M.rows):
            for c in xrange(M.cols):
                 M[r,c] *= x
        return M
      */

        /* Override: Multiplication (*)
         * Purpose: Multiply two matrices A and B together to return matrix M. M[i,j] = A_i1*B_1j + A_i2*B_2j + ... + A_in*B_nj.
         *          Exception will thrown if number of columns of matrix A does not equal the number of rows in matrix B.
         * Parameters: A - First matrix operand.
         *             B - Second matrix operand.
         * Returns: Resulting matrix of the matrix multiplication.
         */
        public static Matrix operator *(Matrix A, Matrix B)
        {
            try
            {
                var ex = new InvalidOperationException("Incompatible Dimensions: Matrix A.cols != Matrix B.rows"); 
                
                if (A.cols != B.rows) { throw ex; }
            }
            catch(Exception ex)
            {
                Console.WriteLine(ex);
            }

            Matrix M = new Matrix(A.rows, B.cols);
        
            for (int r = 0; r < A.rows; r++)
                for (int c = 0; c < B.cols; c++)
                    for (int k = 0; k < A.cols; k++)
                        M.setElement(r, c, M.getElement(r, c) + (A.getElement(r, k) * B.getElement(k, c)));

            return M;
        }
        
/*
    def __rdiv__(A,x):
        """Computes x/A using Gauss-Jordan elimination where x is a scalar"""
        import copy
        n = A.cols
        if A.rows != n:
           raise ArithmeticError, "matrix not squared"
        indexes = range(n)
        A = copy.deepcopy(A)
        B = Matrix.identity(n,x)
        for c in indexes:
            for r in xrange(c+1,n):
                if abs(A[r,c])>abs(A[c,c]):
                    A.swap_rows(r,c)
                    B.swap_rows(r,c)
            p = 0.0 + A[c,c] # trick to make sure it is not integer
            for k in indexes:
                A[c,k] = A[c,k]/p
                B[c,k] = B[c,k]/p
            for r in range(0,c)+range(c+1,n):
                p = 0.0 + A[r,c] # trick to make sure it is not integer
                for k in indexes:
                    A[r,k] -= A[c,k]*p
                    B[r,k] -= B[c,k]*p
            # if DEBUG: print A, B
        return B

    def __div__(A,B):
        if isinstance(B,Matrix):
            return A*(1.0/B) # matrix/marix
        else:
            return (1.0/B)*A # matrix/scalar

    def swap_rows(A,i,j):
        for c in xrange(A.cols):
            A[i,c],A[j,c] = A[j,c],A[i,c]

    @property
    def t(A):
        """Transposed of A"""
        return Matrix(A.cols,A.rows, fill=lambda r,c: A[c,r])

def is_almost_symmetric(A, double ap=1e-6, double rp=1e-4):
    if A.rows != A.cols: return False
    for r in xrange(A.rows-1):
        for c in xrange(r):
            delta = abs(A[r,c]-A[c,r])
            if delta>ap and delta>max(abs(A[r,c]),abs(A[c,r]))*rp:
                return False
    return True

def is_almost_zero(A, double ap=1e-6, double rp=1e-4):
    for r in xrange(A.rows):
        for c in xrange(A.cols):
            delta = abs(A[r,c]-A[c,r])
            if delta>ap and delta>max(abs(A[r,c]),abs(A[c,r]))*rp:
                return False
    return True

def norm(A,p=1):
    if isinstance(A,(list,tuple)):
        return sum(x**p for x in A)**(1.0/p)
    elif isinstance(A,Matrix):
        if A.rows==1 or A.cols==1:
             return sum(norm(A[r,c])**p \
                for r in xrange(A.rows) \
                for c in xrange(A.cols))**(1.0/p)
        elif p==1:
             return max([sum(norm(A[r,c]) \
                for r in xrange(A.rows)) \
                for c in xrange(A.cols)])
        else:
             raise NotImplementedError
    else:
        return abs(A)

def condition_number(f,x=None,h=1e-6):
    if callable(f) and not x is None:
        return D(f,h)(x)*x/f(x)
    elif isinstance(f,Matrix): # if is the Matrix J
        return norm(f)*norm(1/f)
    else:
        raise NotImplementedError

def exp(x,double ap=1e-6,double rp=1e-4,int ns=40):
    if isinstance(x,Matrix):
       t = s = Matrix.identity(x.cols)
       for k in range(1,ns):
           t = t*x/k   # next term
           s = s + t   # add next term
           if norm(t)<max(ap,norm(s)*rp): return s
       raise ArithmeticError, 'no convergence'
    elif type(x)==type(1j):
       return cmath.exp(x)
    else:
       return math.exp(x)

def Cholesky(A):
    import copy, math
    if not is_almost_symmetric(A):
        raise ArithmeticError, 'not symmetric'
    L = copy.deepcopy(A)
    for k in xrange(L.cols):
        if L[k,k]<=0:
            raise ArithmeticError, 'not positive definitive'
        p = L[k,k] = math.sqrt(L[k,k])
        for i in xrange(k+1,L.rows):
            L[i,k] /= p
        for j in xrange(k+1,L.rows):
            p=float(L[j,k])
            for i in xrange(k+1,L.rows):
                L[i,j] -= p*L[i,k]
    for  i in xrange(L.rows):
        for j in range(i+1,L.cols):
            L[i,j]=0
    return L

def is_positive_definite(A):
    if not is_symmetric(A):
        return False
    try:
        Cholesky(A)
        return True
    except RuntimeError:
        return False

def Markovitz(mu, A, r_free):
    """Assess Markovitz risk/return.
    Example:
    >>> cov = Matrix.from_list([[0.04, 0.006,0.02],
    ...                        [0.006,0.09, 0.06],
    ...                        [0.02, 0.06, 0.16]])
    >>> mu = Matrix.from_list([[0.10],[0.12],[0.15]])
    >>> r_free = 0.05
    >>> x, ret, risk = Markovitz(mu, cov, r_free)
    >>> print x
    [0.556634..., 0.275080..., 0.1682847...]
    >>> print ret, risk
    0.113915... 0.186747...
    """
    x = Matrix(A.rows, 1)
    x = (1/A)*(mu - r_free)
    x = x/sum(x[r,0] for r in range(x.rows))
    portfolio = [x[r,0] for r in range(x.rows)]
    portfolio_return = mu*x
    portfolio_risk = sqrt(x*(A*x))
    return portfolio, portfolio_return, portfolio_risk

def fit_least_squares(points, f):
    """
    Computes c_j for best linear fit of y[i] \pm dy[i] = fitting_f(x[i])
    where fitting_f(x[i]) is \sum_j c_j f[j](x[i])

    parameters:
    - a list of fitting functions
    - a list with points (x,y,dy)

    returns:
    - column vector with fitting coefficients
    - the chi2 for the fit
    - the fitting function as a lambda x: ....
    """
    def eval_fitting_function(f,c,x):
        if len(f)==1: return c*f[0](x)
        else: return sum(func(x)*c[i,0] for i,func in enumerate(f))
    A = Matrix(len(points),len(f))
    b = Matrix(len(points))
    for i in range(A.rows):
        weight = 1.0/points[i][2] if len(points[i])>2 else 1.0
        b[i,0] = weight*float(points[i][1])
        for j in range(A.cols):
            A[i,j] = weight*f[j](float(points[i][0]))
    c = (1.0/(A.t*A))*(A.t*b)
    chi = A*c-b
    chi2 = norm(chi,2)**2
    fitting_f = lambda x, c=c, f=f, q=eval_fitting_function: q(f,c,x)
    return c.data, chi2, fitting_f

     */
    }
       

    class Numeric
    {
        /* Function: sqrt
         * Purpose: 
         * Parameters: 
         * Returns: 
         */
        public double sqrt(double x)
        {
/*    try:
        return math.sqrt(x)
    except ValueError:
        return cmath.sqrt(x)
 */
            double a;

            try
            {
                a = Math.Sqrt(x);
            }
            catch (Exception e)
            {
                return 0.0;
            }
            return a;
        }


        /* Function: solve_fixed_point
         * Purpose: 
         * Parameters: 
         * Returns: 
         */
        public void solve_fixed_point(Function f, double x, double ap = 1e-6, double rp = 1e-4, int ns = 100)
        {
            /* 
             def g(x): return f(x)+x # f(x)=0 <=> g(x)=x
    Dg = D(g)
    for k in xrange(ns):
        if abs(Dg(x)) >= 1:
            raise ArithmeticError, 'error D(g)(x)>=1'
        (x_old, x) = (x, g(x))
        if k>2 and norm(x_old-x)<max(ap,norm(x)*rp):
            return x
    raise ArithmeticError, 'no convergence'

             */
        }

        /* Function: solve_bisection
         * Purpose: 
         * Parameters: 
         * Returns: 
         */
        public void solve_bisection(Function f, double a, double b, double ap=1e-6, double rp=1e-4, int ns=100)
        {
            /* 
             fa, fb = f(a), f(b)
    if fa == 0: return a
    if fb == 0: return b
    if fa*fb > 0:
        raise ArithmeticError, 'f(a) and f(b) must have opposite sign'
    for k in xrange(ns):
        x = (a+b)/2
        fx = f(x)
        if fx==0 or norm(b-a)<max(ap,norm(x)*rp): return x
        elif fx * fa < 0: (b,fb) = (x, fx)
        else: (a,fa) = (x, fx)
    raise ArithmeticError, 'no convergence'

             */
        }

        /* Function: solve_newton
         * Purpose: 
         * Parameters: 
         * Returns: 
         */
        public void solve_newton(Function f, double x, double ap=1e-6, double rp=1e-4, int ns=20)
        {
            /* 
                 x = float(x) # make sure it is not int
    for k in xrange(ns):
        (fx, Dfx) = (f(x), D(f)(x))
        if norm(Dfx) < ap:
            raise ArithmeticError, 'unstable solution'
        (x_old, x) = (x, x-fx/Dfx)
        if k>2 and norm(x-x_old)<max(ap,norm(x)*rp): return x
    raise ArithmeticError, 'no convergence'
             */
        }

        /* Function: solve_secant
         * Purpose: 
         * Parameters: 
         * Returns: 
         */
        public void solve_secant(Function f, double x, double ap=1e-6, double rp=1e-4, int ns=20)
        {
            /* 
             x = float(x) # make sure it is not int
    (fx, Dfx) = (f(x), D(f)(x))
    for k in xrange(ns):
        if norm(Dfx) < ap:
            raise ArithmeticError, 'unstable solution'
        (x_old, fx_old,x) = (x, fx, x-fx/Dfx)
        if k>2 and norm(x-x_old)<max(ap,norm(x)*rp): return x
        fx = f(x)
        Dfx = (fx-fx_old)/(x-x_old)
    raise ArithmeticError, 'no convergence'

             */
        }

        /* Function: solve_newton_stabilized
         * Purpose: 
         * Parameters: 
         * Returns: 
         */
        public void solve_newton_stabilized(Function f, double a, double b, double ap=1e-6, double rp=1e-4, int ns=20)
        {
            /* 
    fa, fb = f(a), f(b)
    if fa == 0: return a
    if fb == 0: return b
    if fa*fb > 0:
        raise ArithmeticError, 'f(a) and f(b) must have opposite sign'
    x = (a+b)/2
    (fx, Dfx) = (f(x), D(f)(x))
    for k in xrange(ns):
        x_old, fx_old = x, fx
        if norm(Dfx)>ap: x = x - fx/Dfx
        if x==x_old or x<a or x>b: x = (a+b)/2
        fx = f(x)
        if fx==0 or norm(x-x_old)<max(ap,norm(x)*rp): return x
        Dfx = (fx-fx_old)/(x-x_old)
        if fx * fa < 0: (b,fb) = (x, fx)
        else: (a,fa) = (x, fx)
    raise ArithmeticError, 'no convergence'          
             */
        }

        /* Function: optimize_bisection
         * Purpose: 
         * Parameters: 
         * Returns: 
         */
        public void optimize_bisection(Function f, double a, double b, double ap=1e-6, double rp=1e-4, int ns=100)
        {
            /* 
             Dfa, Dfb = D(f)(a), D(f)(b)
    if Dfa == 0: return a
    if Dfb == 0: return b
    if Dfa*Dfb > 0:
        raise ArithmeticError, 'D(f)(a) and D(f)(b) must have opposite sign'
    for k in xrange(ns):
        x = (a+b)/2
        Dfx = D(f)(x)
        if Dfx==0 or norm(b-a)<max(ap,norm(x)*rp): return x
        elif Dfx * Dfa < 0: (b,Dfb) = (x, Dfx)
        else: (a,Dfa) = (x, Dfx)
    raise ArithmeticError, 'no convergence'

             */
        }

        /* Function: optimize_newton
         * Purpose: 
         * Parameters: 
         * Returns: 
         */
        public void optimize_newton(Function f, float x, double ap=1e-6, double rp=1e-4, int ns=20)
        {
            /* 
             x = float(x) # make sure it is not int
    for k in xrange(ns):
        (Dfx, DDfx) = (D(f)(x), DD(f)(x))
        if Dfx==0: return x
        if norm(DDfx) < ap:
            raise ArithmeticError, 'unstable solution'
        (x_old, x) = (x, x-Dfx/DDfx)
        if norm(x-x_old)<max(ap,norm(x)*rp): return x
    raise ArithmeticError, 'no convergence'
             */
        }

        /* Function: optimize_secant
         * Purpose: 
         * Parameters: 
         * Returns: 
         */
        public void optimize_secant(Function f, float x, double ap=1e-6, double rp=1e-4, int ns=100)
        {
            /* 
             x = float(x) # make sure it is not int
    (fx, Dfx, DDfx) = (f(x), D(f)(x), DD(f)(x))
    for k in xrange(ns):
        if Dfx==0: return x
        if norm(DDfx) < ap:
            raise ArithmeticError, 'unstable solution'
        (x_old, Dfx_old, x) = (x, Dfx, x-Dfx/DDfx)
        if norm(x-x_old)<max(ap,norm(x)*rp): return x
        fx = f(x)
        Dfx = D(f)(x)
        DDfx = (Dfx - Dfx_old)/(x-x_old)
    raise ArithmeticError, 'no convergence'

             */
        }

        /* Function: optimize_newton_stabilized
         * Purpose: 
         * Parameters: 
         * Returns: 
         */
        public void optimize_newton_stabilized(Function f, double a, double b, double ap=1e-6, double rp=1e-4, int ns=20)
        {
            /* 
             Dfa, Dfb = D(f)(a), D(f)(b)
    if Dfa == 0: return a
    if Dfb == 0: return b
    if Dfa*Dfb > 0:
        raise ArithmeticError, 'D(f)(a) and D(f)(b) must have opposite sign'
    x = (a+b)/2
    (fx, Dfx, DDfx) = (f(x), D(f)(x), DD(f)(x))
    for k in xrange(ns):
        if Dfx==0: return x
        x_old, fx_old, Dfx_old = x, fx, Dfx
        if norm(DDfx)>ap: x = x - Dfx/DDfx
        if x==x_old or x<a or x>b: x = (a+b)/2
        if norm(x-x_old)<max(ap,norm(x)*rp): return x
        fx = f(x)
        Dfx = (fx-fx_old)/(x-x_old)
        DDfx = (Dfx-Dfx_old)/(x-x_old)
        if Dfx * Dfa < 0: (b,Dfb) = (x, Dfx)
        else: (a,Dfa) = (x, Dfx)
    raise ArithmeticError, 'no convergence'

             */
        }

        /* Function: optimize_golden_search
         * Purpose: 
         * Parameters: 
         * Returns: 
         */
        public void optimize_golden_search(Function f, double a, double b, double ap=1e-6, double rp=1e-4, int ns=100)
        {
            /* 
             a,b=float(a),float(b)
    tau = (sqrt(5.0)-1.0)/2.0
    x1, x2 = a+(1.0-tau)*(b-a), a+tau*(b-a)
    fa, f1, f2, fb = f(a), f(x1), f(x2), f(b)
    for k in xrange(ns):
        if f1 > f2:
            a, fa, x1, f1 = x1, f1, x2, f2
            x2 = a+tau*(b-a)
            f2 = f(x2)
        else:
            b, fb, x2, f2 = x2, f2, x1, f1
            x1 = a+(1.0-tau)*(b-a)
            f1 = f(x1)
        if k>2 and norm(b-a)<max(ap,norm(b)*rp): return b
    raise ArithmeticError, 'no convergence'

             */
        }

/*
 * ###
##### OPTIONAL
    ###
 */
        /* Function: 
         * Purpose: 
         * Parameters: 
         * Returns: 
         */
        public void partial(Function f, int i, double h=1e-4)
        {
            /* 
                 def df(x,f=f,i=i,h=h):
        u = f([e+(h if i==j else 0) for j,e in enumerate(x)])
        w = f([e-(h if i==j else 0) for j,e in enumerate(x)])
        try:
            return (u-w)/2/h
        except TypeError:
            return [(u[i]-w[i])/2/h for i in range(len(u))]
    return df
             */
        }

        /* Function: gradient
         * Purpose: 
         * Parameters: 
         * Returns: 
         */
        public void gradient(Function f, double x, double h=1e-4)
        {
            /* 
             return Matrix(len(x),fill=lambda r,c: partial(f,r,h)(x))
             */
        }

        /* Function: hessian
         * Purpose: 
         * Parameters: 
         * Returns: 
         */
        public void hessian(Function f, double x, double h=1e-4)
        {
            /* 
             return Matrix(len(x),len(x),fill=lambda r,c: partial(partial(f,r,h),c,h)(x))
*/
        }

        /* Function: jacobian
         * Purpose: 
         * Parameters: 
         * Returns: 
         */
        public void jacobian(Function f, double x, double h=1e-4)
        {
            /* 
             partials = [partial(f,c,h)(x) for c in xrange(len(x))]
    return Matrix(len(partials[0]),len(x),fill=lambda r,c: partials[c][r])

             */
        }

        /* Function: solve_newton_multi
         * Purpose: 
         * Parameters: 
         * Returns: 
         */
        public void solve_newton_multi(Function f, double x, double ap=1e-6, double rp=1e-4, int ns=20)
        {
            /* 
             """
    Computes the root of a multidimensional function f near point x.

    Parameters
    f is a function that takes a list and returns a scalar
    x is a list

    Returns x, solution of f(x)=0, as a list
    """
    x = Matrix.from_list([x]).t
    fx = Matrix.from_list([f(x.data)]).t
    for k in xrange(ns):
        (fx.data, J) = (f(x.data), jacobian(f,x.data))
        if norm(J) < ap:
            raise ArithmeticError, 'unstable solution'
        (x_old, x) = (x, x-(1.0/J)*fx)
        if k>2 and norm(x-x_old)<max(ap,norm(x)*rp): return x.data
    raise ArithmeticError, 'no convergence'

             */
        }

        /* Function: optimize_newton_multi
         * Purpose: 
         * Parameters: 
         * Returns: 
         */
        public void optimize_newton_multi(Function f, double x, double ap=1e-6, double rp=1e-4, int ns=20)
        {
            /* 
             """
    Finds the extreme of multidimensional function f near point x.

    Parameters
    f is a function that takes a list and returns a scalar
    x is a list

    Returns x, which maximizes of minimizes f(x)=0, as a list
    """
    x = Matrix.from_list([x]).t
    for k in xrange(ns):
        (grad,H) = (gradient(f,x.data), hessian(f,x.data))
        if norm(H) < ap:
            raise ArithmeticError, 'unstable solution'
        (x_old, x) = (x, x-(1.0/H)*grad)
        if k>2 and norm(x-x_old)<max(ap,norm(x)*rp): return x.data
    raise ArithmeticError, 'no convergence'

             */
        }
    }
}
