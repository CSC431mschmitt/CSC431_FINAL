using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MatrixLib
{
    class Function
    {
        public String full_function { get; set; }

        public Function(String full_function)
        {
            this.full_function = full_function;
        }
    }

    public class Matrix
    {
        /* ??? should these be public or private ??? */
        /* make private since do not need to be accessed outside class */
        private int _rows;
        private int _cols;
        private double[,] _data; /* ??? Does this need to be type Object ??? */ /*I think double type is sufficient*/
        
                
        public Matrix(int rows = 1, int cols = 1, double fill = 0.0, bool optimize = false /* ??? What does optimize do ??? */)
        /* Constructor: Matrix
         * Purpose:     Constructs zero matrix 
         * Parameters:  rows - the integer number of rows
         *              cols - the integer number of columns
         *              fill - the value or callable to be used to fill the matrix
         */
        {
            _rows = rows;
            _cols = cols;
            _data = new double[rows, cols];

            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    _data[i, j] = fill;
        }

        public int rows
        {
            get { return _rows; }
            set { _rows = value; }
        }

        public int cols
        {
            get { return _cols; }
            set { _cols = value; }
        }

        public double this[int row, int column]
        {
            get { return _data[row, column]; }
            set { _data[row, column] = value; }
        }

        public Matrix Transpose()
        {
            Matrix T = new Matrix(_cols, _rows);
                            
            for (int i = 0; i < _rows; i++)
            {
                for (int j = 0; j < _cols; j++)
                {
                    T[j, i] = this[i, j];
                }
            }

            return T;
        }

        public Matrix Clone()
        {
            Matrix T = new Matrix(_rows, _cols);

            for (int i = 0; i < _rows; i++)
            {
                for (int j = 0; j < _cols; j++)
                {
                    T[i, j] = this[i, j];
                }
            }

            return T;
        }

        public List<double[]> as_list()
        /* 
         * Purpose:     Convert Matrix to List of double arrays, where each list item represents row in Matrix.
         * Parameters:  MxN Matrix
         * Returns:     List representation of Matrix.
         */
        {
            List<double[]> toList = new List<double[]>();
            double[] values;

            for (int r = 0; r < this._rows; r++)
            {
                values = new double[this._cols];

                for (int c = 0; c < _cols; c++)
                    values[c] = this[r, c];
                    
                toList.Add(values);
            }
            
            return toList;
        }

        public static Matrix from_list(List<double[]> vlist)
        /*
         * Purpose:     Builds a matrix from a list of arrays
         * Parameter:   List of arrays
         * Output:      Matrix with rows = number of lists, cols = number of items in first array
         */
        {
            int rows = vlist.Count();
            int cols = (vlist[0]).Length;

            Matrix M = new Matrix(rows, cols);

            for (int i = 0; i < vlist.Count; i++)
            {
                for (int j = 0; j < vlist[i].Length; j++)
                {
                    M[i,j] = vlist[i][j];
                }
            }
            
            return M;
        }

        public override string ToString()
        /* Override:    ToString
        * Purpose:      Prints out the full string of a Matrix in a readable row by row format.
        * Parameters:   None
        * Returns:      String representation of matrix
        */
        {
            String mFormat = "[";

            for (int i = 0; i < _rows; i++)
            {
                if (i != 0) mFormat += ", ";
                mFormat += "[";
                for (int j = 0; j < _cols; j++)
                {
                    if (j != 0) mFormat += ", ";
                    mFormat += _data[i, j].ToString();
                }
                mFormat += "]";
            }
            mFormat += "]";
            return mFormat;
        }

        public Matrix row(int row_num)
        /*
         * Parameters:  Row number of a Matrix.
         * Output:      Vector of the items in the indicated row_num of Matrix.
         */
        {
            Matrix M = new Matrix(1, this._cols);

            for (int c = 0; c < this._cols; c++)
            {
                M[0, c] = this[row_num, c];
            }
            
            return M;
        }
        
        public Matrix col(int col_num)
        /*
         * Parameters:  Column number of a Matrix.
         * Output:      Vector of the items in the indicated col_num of Matrix.
         */
        {
            Matrix M = new Matrix(this._rows, 1);

            for (int r = 0; r < this._rows; r++)
            {
                M[r, 0] = this[r, col_num];
            }

            return M;
        }

        public static Matrix identity(int rows, double diagnol, double fill = 0.0)
        /* Constructor: Identity 
         * Purpose:     Constructs an n x n Matrix with the diagonal elements equal to the same value.  All non-diagonal elements are equal as well.
         *              Inherits from the Matrix class.
         * Parameters:  rows - the integer number of rows and columns.
         *              one - the value of all diagonal elements
         *              fill - the value to be used to fill the non-diagonal elements of the identity matrix
         * Output:      rowsXrows square matrix with ones on the main diagonal and zeros elsewhere.
         */
        {
            Matrix M = new Matrix(rows, rows, fill);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < rows; j++)
                {
                    if (i == j)
                        M[i, j] = diagnol;
                    else
                        M[i, j] = fill;
                }
            }

            return M;
        }

        public static Matrix diagonal(double[] d)
        {
        /* Constructor: Diagonal 
         * Purpose:     Constructs an n x n Matrix with the i-th diagonal element equal to the i-th value of the passed array.
         *              Inherits from the Matrix class.M[i, i] = d[i]
         * Parameters:  d - array of elements to populate in diagonal elements of Matrix. 
         * Output:      Matrix where main diagnol entries M[i, i] = d[i], all other entries are 0.
         */
            
            int rows = d.Length;
            int cols = d.Length;
            Matrix M = new Matrix(rows, cols);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    if (i == j)
                        M[i, j] = d[i];
                    else
                        M[i, j] = 0.0;
                }
            }

            return M;
        }

        public static Matrix operator *(Matrix A, Matrix B)
        /* Override:    Multiplication (*)
         * Purpose:     Multiply two matrices A and B together to return matrix M. M[i,j] = A_i1*B_1j + A_i2*B_2j + ... + A_in*B_nj.
         *              Exception will thrown if number of columns of matrix A does not equal the number of rows in matrix B.
         * Parameters:  A - First matrix operand.
         *              B - Second matrix operand.
         * Returns:     Resulting matrix of the matrix multiplication.
         */
        {
            Matrix M = null;
            
            try
            {
                if ( (A._cols == 1 && B._cols == 1) && (A._rows == B._rows) )
                {
                    M = new Matrix(1, 1);
                    for (int r = 0; r < A._rows; r++)
                    {
                        M[0,0] += A[r, 0] * B[r, 0];
                    }
                }
                else if (A._cols != B._rows)
                {
                    throw new InvalidOperationException("Incompatible Dimensions: Matrix A.cols != Matrix B.rows");
                }
                else
                {
                    M = new Matrix(A._rows, B._cols);

                    for (int r = 0; r < A._rows; r++)
                    {
                        for (int c = 0; c < B._cols; c++)
                        {
                            for (int k = 0; k < A._cols; k++)
                                M[r, c] = M[r, c] + (A[r, k] * B[k, c]);
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }

            return M;
        }

        public static Matrix operator *(Matrix A, double x)
        /* Override:    Multiplication (*)
         * Purpose:     Multiply Matrix A and scalar B together to return matrix M. M[i,j] = A_ij * x
         * Parameters:  A - Matrix
         *              B - Scalar
         * Returns:     Resulting matrix of the matrix by scalar multiplication.
         */
        {
            
            Matrix M = new Matrix(A._rows, A._cols);

            for (int r = 0; r < A._rows; r++)
            {
                for (int c = 0; c < A._cols; c++)
                    M[r, c] = A[r, c] * x;
            }

            return M;
        }

        public static Matrix operator /(Matrix A, double x)
        /* Override:    Division (/)
         * Purpose:     Computes x/A using Gauss-Jordan elimination.
         *              Exception thrown if Matrix A is singular.
         * Parameters:  A - Matrix.
         *              B - Scalar.
         * Returns:     Resulting matrix of the matrix division.
         */
        {
            Matrix B = null;

            int cols = A._cols;

            try
            {
                if (A._rows != cols)
                {
                    throw new ArithmeticException("Matrix A not squared.");
                }
                else
                {
                    int upper_index = cols - 1;
                    
                    A = A.Clone();
                    B = Matrix.identity(cols, x);

                    for (int c = 0; c < cols; c++)
                    {
                        for (int r = c+1; r < cols; r++)
                        {
                            if (Math.Abs(A[r, c]) > Math.Abs(A[c, c]))
                            {
                                A.swap_rows(r, c);
                                B.swap_rows(r, c);
                            }
                        }
                        double p = 0.0 + A[c, c]; //trick to implicitly cast as double

                        for (int k = 0; k < cols; k++)
                        {
                            A[c, k] = A[c, k] / p;
                            B[c, k] = B[c, k] / p;
                        }

                        for(int r=0; r < cols; r++) if(r!=c) 
                        {
                            p = 0.0 + A[r, c]; //trick to implicitly cast as double
                            for (int k = 0; k < cols; k++)
                            {
                                A[r, k] -= A[c, k] * p;
                                B[r, k] -= B[c, k] * p;
                            }
                        }
                    }
                }
            }
            catch (InvalidOperationException ex)
            {
                Console.WriteLine(ex.ToString());
            }

            return B;
        }

        /* This needs to be implemented to complete division operator */
        //def __div__(A,B):
        //    if isinstance(B,Matrix):
        //        return A*(1.0/B) # matrix/marix
        //    else:
        //        return (1.0/B)*A # matrix/scalar

        public void swap_rows(int i, int j)
        {
            try
            {
                //var ex = new InvalidOperationException("Incompatible Dimensions");

                if ((this._data).Rank != 2) 
                {
                    throw new InvalidOperationException("Incompatible Dimensions");
                }

                else if ((i >= this._rows) || (j >= this._rows))
                {
                    throw new InvalidOperationException("Index out of bounds");
                }

                else
                {
                    for (int c = 0; c < this._cols; c++)
                    {
                        double x = this[i, c];
                        this[i, c] = this[j, c];
                        this[j, c] = x;
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }
        }

        public static Matrix operator +(Matrix A, Matrix B)
        /* Override:    Addition (+)
         * Purpose:     Add two matrices A and B together to return matrix M. M[i,j] = A[i,j] + B[i,j]
         *              Exception thrown if matrix A is not equal in size to matrix B.
         * Parameters:  A - First matrix operand.
         *              B - Second matrix operand.
         * Returns:     Resulting matrix of the matrix addition.
         */
        {
            int rows = A._rows;
            int cols = A._cols;
            Matrix M = null;

            try
            {
                if ((B._rows != rows) || (B._cols != cols))
                {
                    throw new InvalidOperationException("Incompatible Dimensions: Matrix A size != Matrix B size");
                }
                else
                {
                    M = new Matrix(rows, cols);

                    for (int i = 0; i < A._rows; i++)
                    {
                        for (int j = 0; j < A._cols; j++)
                        {
                            M[i, j] = A[i, j] + B[i, j];
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }
                        
            return M;
        }

        public static Matrix operator -(Matrix A, Matrix B)
        /* Override:    Addition (-)
         * Purpose:     Subtract two matrices A and B to return matrix M. M[i,j] = A[i,j] - B[i,j]
         *              Exception thrown if matrix A is not equal in size to matrix B.
         * Parameters:  A - First matrix operand.
         *              B - Second matrix operand.
         * Returns:     Resulting matrix of the matrix subtraction.
         */
        {
            int rows = A._rows;
            int cols = A._cols;
            Matrix M = null;

            try
            {
                if ((B._rows != rows) || (B._cols != cols))
                {
                    throw new InvalidOperationException("Incompatible Dimensions: Matrix A size != Matrix B size");
                }
                else
                {
                    M = new Matrix(rows, cols);

                    for (int i = 0; i < A._rows; i++)
                    {
                        for (int j = 0; j < A._cols; j++)
                        {
                            M[i, j] = A[i, j] - B[i, j];
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }
                        
            return M;
        }

        
        /*
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
        public void solve_bisection(Function f, double a, double b, double ap = 1e-6, double rp = 1e-4, int ns = 100)
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
        public void solve_newton(Function f, double x, double ap = 1e-6, double rp = 1e-4, int ns = 20)
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
        public void solve_secant(Function f, double x, double ap = 1e-6, double rp = 1e-4, int ns = 20)
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
        public void solve_newton_stabilized(Function f, double a, double b, double ap = 1e-6, double rp = 1e-4, int ns = 20)
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
        public void optimize_bisection(Function f, double a, double b, double ap = 1e-6, double rp = 1e-4, int ns = 100)
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
        public void optimize_newton(Function f, float x, double ap = 1e-6, double rp = 1e-4, int ns = 20)
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
        public void optimize_secant(Function f, float x, double ap = 1e-6, double rp = 1e-4, int ns = 100)
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
        public void optimize_newton_stabilized(Function f, double a, double b, double ap = 1e-6, double rp = 1e-4, int ns = 20)
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
        public void optimize_golden_search(Function f, double a, double b, double ap = 1e-6, double rp = 1e-4, int ns = 100)
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
        public void partial(Function f, int i, double h = 1e-4)
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
        public void gradient(Function f, double x, double h = 1e-4)
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
        public void hessian(Function f, double x, double h = 1e-4)
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
        public void jacobian(Function f, double x, double h = 1e-4)
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
        public void solve_newton_multi(Function f, double x, double ap = 1e-6, double rp = 1e-4, int ns = 20)
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
        public void optimize_newton_multi(Function f, double x, double ap = 1e-6, double rp = 1e-4, int ns = 20)
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
