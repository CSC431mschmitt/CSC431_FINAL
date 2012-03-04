using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;


/*
 * This code is distributed under the BSD license, and it is a rewrite of
 * code shared in CSC431 class at DePaul University by Massimo Di Pierro.
 */
namespace MatrixLib
{
    // Basic complex number object that allows you to create and print a complex number.  No operation overloads included.
    public class ComplexNumber
    {
        public double real, imaginary;

        // Constructor
        public ComplexNumber(double real, double imaginary)
        {
            this.real = real;
            this.imaginary = imaginary;
        }

        // Print.
        public override string ToString() 
        {
            if (this.imaginary == 0)
                return string.Format("{0}", real);

            if (this.real == 0)
                return string.Format("{0}j", imaginary);
            
            return string.Format("{0} + {1}j", real, imaginary);
        }
    }

    public class Function
    {
        public Function() { }

        public virtual double f(double x) { return 0 * x; }

        public double Df(double x, double h = 1e-5) { return (f(x + h) - f(x - h)) / (2.0 * h); }

        public double DDf(double x, double h = 1e-5) { return (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h); }

        public double solve_newton(double x_guess, double ap = 1e-5, double rp = 1e-4, int ns = 100) {
            
            double x_old, x = x_guess;
            
            try
            {
                for (int k = 0; k < ns; k++) {
                    x_old = x;
                    x = x - f(x) / Df(x);
                    if (Math.Abs(x - x_old) < Math.Max(ap, rp * Math.Abs(x))) return x;
                }

                throw new InvalidOperationException("No Convergence");
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }

            return 0.0;
        }  

        public double optimize_newton(double x_guess, double ap = 1e-5, double rp = 1e-4, int ns = 20)
        {
            double x_old, x = x_guess;

            try
            {
                for (int k = 0; k < ns; k++) {
                    x_old = x;
                    x = x - Df(x) / DDf(x);
                    if (Math.Abs(x - x_old) < Math.Max(ap, rp * Math.Abs(x))) return x;
                }

                throw new InvalidOperationException("No Convergence");
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }

            return 0.0;
        }

        public void solve_fixed_point(Function f, double x, double ap = 0.000001, double rp = 0.0001, int ns = 100)
        /* Function: solve_fixed_point
         * Purpose: 
         * Parameters: 
         * Returns: 
         */
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

        public double solve_bisection(double a, double b, double ap = 0.000001, double rp = 0.0001, int ns = 100)
        /* Function:    solve_bisection
         * Purpose:     Solve f(x) = 0 when function is continuous and is known to change
         *              sign between x=a and x=b.
         * Parameters: ???Are a and b matrix or double parameters???
         * Returns:     Approximation for exact solution
         */
        {
            double fa = f(a), fb = f(b), x, fx;

            if (fa == 0) return a;
            if (fb == 0) return b;

            try
            {
                if (fa * fb > 0) 
                    throw new InvalidOperationException("f(a) and f(b) must have opposite sign.");

                for (int k = 0; k < ns; k++)
                {
                    x = (a + b) / 2;
                    fx = f(x);

                    if (fx == 0 || Math.Abs(b - a) < Math.Max(ap, rp * Math.Abs(x)))
                    {
                        return x;
                    }
                    else if (fx * fa < 0)
                    {
                        b = x;
                        fb = fx;
                    }
                    else
                    {
                        a = x;
                        fa = fx;
                    }
                }

                throw new InvalidOperationException("No Convergence");
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }

            return 0.0;
        }

        public double solve_secant(double x, double ap = 0.000001, double rp = 0.0001, int ns = 20)
        /* Function:    solve_secant
         * Purpose:     Solves f(x) = 0 when function is differentiable by replacing f'(x) 
         *              with difference quotient, using the current point and previous 
         *              point visited in algorithm
         * Parameters:  x - Initial guess for solution x
         *              ap - absolute precision
         *              rp - relative precision
         *              ns - max number of iterations
         * Returns:     Approximation for exact solution
         */
        {
            double fx = f(x);
            double Dfx = Df(x);

            double x_old = 0.0;
            double fx_old = 0.0;

            Numeric n = new Numeric();

            try
            {
                for (int k = 0; k < ns; k++)
                {
                    if (Math.Abs(Dfx) < ap)
                    {
                        throw new ArithmeticException("Unstable Solution");
                    }

                    x_old = x;
                    fx_old = fx;
                    x = x - fx / Dfx;

                    if ((k > 2) && (Math.Abs(x - x_old) < Math.Max(ap, Math.Abs(x) * rp)))
                    {
                        return x;
                    }

                    fx = f(x);
                    Dfx = (fx - fx_old) / (x - x_old);
                }

                throw new ArithmeticException("No convergence.");
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }

            return Double.NaN;
        }

        public double solve_newton_stabilized(double a, double b, double ap = 0.000001, double rp = 0.0001, int ns = 20)
        /* Function:    solve_newton_stabilized
         * Purpose: 
         * Parameters: 
         * Returns:     Approximation for exact solution
         */
        {
            double fa = f(a), fb = f(b), x, fx, Dfx, x_old, fx_old;

            if (fa == 0) return a;
            if (fb == 0) return b;

            try
            {
                if (fa * fb > 0)
                    throw new InvalidOperationException("f(a) and f(b) must have opposite sign.");

                x = (a + b) / 2;
                fx = f(x);
                Dfx = Df(x);

                for (int k = 0; k < ns; k++)
                {
                    x_old = x;
                    fx_old = fx;

                    if (Math.Abs(Dfx) > ap)
                        x -= fx/Dfx;

                    if (x == x_old || x < a || x > b) 
                        x = (a + b) / 2;

                    fx = f(x);

                    if (fx == 0 || Math.Abs(x - x_old) < Math.Max(ap, Math.Abs(x) * rp))
                        return x;

                    Dfx = (fx - fx_old) / (x - x_old);

                    if (fx * fa < 0)
                    {
                        b = x;
                        fb = fx;
                    }
                    else
                    {
                        a = x;
                        fa = fx;
                    }
                }

                throw new InvalidOperationException("No Convergence");
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }

            return 0.0;
        }

        public double optimize_bisection(double a, double b, double ap = 0.000001, double rp = 0.0001, int ns = 100)
        /* Function:    optimize_bisection
         * Purpose:     Solve f'(x) = 0 by optimizing bisection solver
         * Parameters:  a, b - domain for solution x where Df(a) and Df(b) have opposite signs
         *              ap - absolute precision
         *              rp - relative precision
         *              ns - max number of iterations
         * Returns:     Approximation for exact solution
         */
        {
            double Dfa = Df(a), Dfb = Df(b), x, Dfx;

            if (Dfa == 0) return a;
            if (Dfb == 0) return b;

            try
            {
                if (Dfa * Dfb > 0)
                    throw new InvalidOperationException("Df(a) and Df(b) must have opposite sign.");

                for (int k = 0; k < ns; k++)
                {
                    x = (a + b) / 2;
                    Dfx = Df(x);

                    if (Dfx == 0 || Math.Abs(b - a) < Math.Max(ap, rp * Math.Abs(x)))
                    {
                        return x;
                    }
                    else if (Dfx * Dfa < 0)
                    {
                        b = x;
                        Dfb = Dfx;
                    }
                    else
                    {
                        a = x;
                        Dfa = Dfx;
                    }
                }

                throw new InvalidOperationException("No Convergence");
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }

            return 0.0;
        }

        public double optimize_secant(double x, double ap = 0.000001, double rp = 0.0001, int ns = 100)
        /* Function:    optimize_secant
         * Purpose:     Solve f'(x) = 0 by optimizing secant solver
         * Parameters:  x - Initial guess for solution
         *              ap - absolute precision
         *              rp - relative precision
         *              ns - max number of iterations
         * Returns:     Approximation for exact solution
         */
        {
            double fx = f(x);
            double Dfx = Df(x);
            double DDfx = DDf(x);

            double x_old = 0.0;
            double Dfx_old = 0.0;

            Numeric n = new Numeric();

            try
            {
                for (int k = 0; k < ns; k++)
                {
                    if (Dfx == 0)
                        return x;

                    if (Math.Abs(DDfx) < ap)
                    {
                        throw new ArithmeticException("Unstable Solution");
                    }

                    x_old = x;
                    Dfx_old = Dfx;
                    x = x - Dfx / DDfx;

                    if ( (Math.Abs(x - x_old) < Math.Max(ap, Math.Abs(x) * rp)))
                        return x;

                    fx = f(x);
                    Dfx = Df(x);
                    DDfx = (Dfx - Dfx_old) / (x - x_old);
                }

                throw new ArithmeticException("No convergence.");
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }

            return double.NaN;
        }

        public double optimize_newton_stabilized(double a, double b, double ap = 0.000001, double rp = 0.0001, int ns = 20)
        /* Function:    optimize_newton_stabilized
         * Purpose: 
         * Parameters: 
         * Returns:     Approximation for exact solution
         */
        {
            double Dfa = Df(a), Dfb = Df(b), x, fx, Dfx, DDfx, x_old, fx_old, Dfx_old;

            if (Dfa == 0) return a;
            if (Dfb == 0) return b;

            try
            {
                if (Dfa * Dfb > 0)
                    throw new InvalidOperationException("Df(a) and Df(b) must have opposite sign.");

                x = (a + b) / 2;
                fx = f(x);
                Dfx = Df(x);
                DDfx = DDf(x);

                for (int k = 0; k < ns; k++)
                {
                    if (Dfx == 0)
                        return x;

                    x_old = x;
                    fx_old = fx;
                    Dfx_old = Dfx;
 
                    if (Math.Abs(DDfx) > ap)
                        x = x - Dfx / DDfx;

                    if (x == x_old || x < a || x > b) 
                        x = (a + b) / 2;

                    if (Math.Abs(x - x_old) < Math.Max(ap, Math.Abs(x) * rp))
                        return x;
    
                    fx = f(x);
                    Dfx = (fx - fx_old) / (x - x_old);
                    DDfx = (Dfx - Dfx_old)  /(x - x_old);

                    if (Dfx * Dfa < 0)
                    {
                        b = x;
                        Dfb = Dfx;
                    }
                    else
                    {
                        a = x;
                        Dfa = Dfx;
                    }
                }

                throw new InvalidOperationException("No Convergence");
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }

            return 0.0;
        }

        public double optimize_golden_search(double a, double b, double ap = 0.000001, double rp = 0.0001, int ns = 100)
        /* Function:    optimize_golden_search
         * Purpose:     Solve f'(x) = 0 where f(x) is continuous but not differentiable
         * Parameters:  a, b - Domain in which function is concave or convex
         *              ap - absolute precision
         *              rp - relative precision
         *              ns - max number of iterations
         * Returns:     Approximation for exact solution
         */
        {
            double tau = (Math.Sqrt(5.0) - 1.0)/2.0;
            double x1 = a + (1.0 - tau) * (b - a);
            double x2 = a + tau * (b - a);

            double fa = f(a);
            double f1 = f(x1);
            double f2 = f(x2);
            double fb = f(b);

            try
            {
                for (int k = 0; k < ns; k++)
                {
                    if (f1 > f2)
                    {
                        a = x1;
                        fa = f1;
                        x1 = x2;
                        f1 = f2;
                        x2 = a + tau * (b - a);
                        f2 = f(x2);
                    }
                    else
                    {
                        b = x2;
                        fb = f2;
                        x2 = x1;
                        f2 = f1;
                        x1 = a + (1.0 - tau) * (b - a);
                        f1 = f(x1);
                    }

                    if ( (k > 2) && (Math.Abs(b - a) < Math.Max(ap, Math.Abs(b) * rp)))
                        return b;
                }

                throw new ArithmeticException("No Convergence");
            }
            
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }

            return double.NaN;
        }

        /*
         *###
         *##### OPTIONAL
         *###
         */
        public void partial(Function f, int i, double h = 0.0001)
        /* Function: 
         * Purpose: 
         * Parameters: 
         * Returns: 
         */
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

        public void gradient(Function f, double x, double h = 0.0001)
        /* Function: gradient
         * Purpose: 
         * Parameters: 
         * Returns: 
         */
        {
            /* 
             return Matrix(len(x),fill=lambda r,c: partial(f,r,h)(x))
             */
        }

        public void hessian(Function f, double x, double h = 0.0001)
        /* Function: hessian
         * Purpose: 
         * Parameters: 
         * Returns: 
         */
        {
            /* 
             return Matrix(len(x),len(x),fill=lambda r,c: partial(partial(f,r,h),c,h)(x))
*/
        }


        public void jacobian(Function f, double x, double h = 0.0001)
        /* Function: jacobian
         * Purpose: 
         * Parameters: 
         * Returns: 
         */
        {
            /* 
             partials = [partial(f,c,h)(x) for c in xrange(len(x))]
            return Matrix(len(partials[0]),len(x),fill=lambda r,c: partials[c][r])

             */
        }

        public void solve_newton_multi(Function f, double x, double ap = 0.000001, double rp = 0.0001, int ns = 20)
        /* Function: solve_newton_multi
         * Purpose: 
         * Parameters: 
         * Returns: 
         */
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

        public void optimize_newton_multi(Function f, double x, double ap = 0.000001, double rp = 0.0001, int ns = 20)
        /* Function: solve_secant
         * Purpose: 
         * Parameters: 
         * Returns: 
         */
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
    
    public class Numeric
    {
        public bool is_almost_symmetric(Matrix A, double ap = 0.000001, double rp = 0.0001)
        /* 
         * Purpose:     Determine if Matrix A almost symmetric A[i,j] almost equal to A[j, i], within certain precision
         * Parameters:  A - Matrix to test for symmetry
         *              ap - absolute precision
         *              rp - relative precision
         * Returns:     True if matrix is almost symmetric, else false
         */
        {
            if (A.rows != A.cols)
                return false;

            else
            {
                for (int r = 0; r < A.rows; r++)
                {
                    for (int c = 0; c < r; c++)
                    {
                        //Console.WriteLine("r, c: " + r + "," + c);
                        //Console.WriteLine("A[r, c]: " + A[r, c]);
                        //Console.WriteLine("A[c, r]: " + A[c, r]);
                        double delta = Math.Abs(A[r, c] - A[c, r]);
                        //Console.WriteLine("delta: " + delta);
                        double abs_arc = Math.Abs(A[r, c]);
                        double abs_acr = Math.Abs(A[c, r]);
                        //Console.WriteLine("Math.Max(abs_arc, abs_acr)*rp: " + Math.Max(abs_arc, abs_acr) * rp);
                        if ((delta > ap) && (delta > Math.Max(abs_arc, abs_acr) * rp))
                            return false;
                    }
                }
            }

            return true;
        }

        public bool is_almost_zero(Matrix A, double ap = 0.000001, double rp = 0.0001)
        /* 
         * Purpose:     Determine if Matrix A is zero within a certain precision
         * Parameters:  A - Matrix to test 
         *              ap - absolute precision
         *              rp - relative precision
         * Returns:     True if matrix is almost zero, else false
         */
        {
            var result = true;

            for (int r = 0; r < A.rows; r++)
            {
                for (int c = 0; c < A.rows; c++)
                {
                    double delta = Math.Abs(A[r, c] - A[c, r]);
                    double abs_arc = Math.Abs(A[r, c]);
                    double abs_acr = Math.Abs(A[c, r]);
                    if ((delta > ap) && (delta > Math.Max(abs_arc, abs_acr) * rp))
                        result = false;
                }
            }

            return result;
        }

        public double norm(List<double> x, int p=1)
        /* 
         * Purpose:     Compute p-th root of sum of list items each raised to power of p,
         *              which represents magnitude of vector
         * Parameters:  x - List of double values 
         *              p - The norm value to compute 
         *              
         * Returns:     p-norm of vector x
         */
        {
            double sum = 0.0;
            for (int i = 0; i < x.Count; i++)
            {
                sum += Math.Pow(Math.Abs(x[i]), p);
            }
                
            return Math.Pow(sum, 1.0/p);
        }

        public double norm(Matrix A, int p=1)
        /* 
         * Purpose:     Compute p-norm of Matrix A, which represents magnitude of Matrix
         * Parameters:  A - Matrix
         *              p - The norm value to compute
         * Returns:     p-norm for m x 1 and 1 x n Matrix,
         *              or 1-norm for m x n Matrix, else
         *              raises Not Implemented error
         */
        {   
            if ( p == 1 )
            {
                List<double> sums = new List<double>();

                for (int c = 0; c < A.cols; c++)
                {
                    double sum = 0.0;

                    for (int r = 0; r < A.rows; r++)    
                    {
                        sum += Math.Abs(A[r, c]);
                    }
                    
                    sums.Add(sum);
                }

                return sums.Max();
            }
            else if ((A.rows == 1) || (A.cols == 1))
            {
                double sum = 0.0;

                for (int c = 0; c < A.cols; c++)
                {
                    for (int r = 0; r < A.rows; r++)
                    {
                        sum += Math.Pow(Math.Abs(A[r, c]), p);
                    }
                }

                return Math.Pow(sum, 1.0 / p);
            }
            else
                throw new InvalidOperationException("NotImplementedError");
        }
        
        public double condition_number(Function func, double x, double h=0.000001)
        /* 
         * Purpose:     Compute condition number for a function, f
         * Parameters:  f - Specified function
         *              x - Specific point at which to evaluate f
         *              h - Small value to indicate change in x 
         * Returns:     Value estimating how sensitive function f
         *              is to small changes in x
         */
        {
            try 
            {
                return func.Df(x, h) * x / func.f(x);
            }
            catch 
            {
                throw new InvalidOperationException("NotImplementedError");
            }
        }
        
        public double condition_number(Matrix A)
        /* 
         * Purpose:     Compute condition number for a Matrix
         * Parameters:  A - Matrix
         * Returns:     Value representing stability of Matrix A
         */
        {
            try
            {
                return norm(A) * norm(1.0/A);
            }
            catch
            {
                throw new InvalidOperationException("NotImplementedError");
            }
        }


        public Matrix exp(Matrix A, double ap=0.000001, double rp=0.0001, int ns=40)
        /* 
         * Purpose:     Compute exponential of Matrix A using series expansion
         * Parameters:  A - Matrix to test for symmetry.
         *              ap - absolute precision
         *              rp - relative precision
         *              ns - max number of iterations
         * Returns:     Matrix exponential for square matrix, else
         *              raises error
         */
        {
            Matrix T = Matrix.identity(A.cols);
            Matrix S = Matrix.identity(A.cols);

            for (int k = 1; k < ns; k++)
            {
                T = T*A/k; //next term
                S = S + T; //add next term

                if (norm(T) < Math.Max(ap, norm(S) * rp))
                    return S;
            }
            
            throw new ArithmeticException("No Convergence.");
        }

        public bool is_positive_definite(Matrix A)
        /* 
         * Purpose:     Check if symmetric Matrix is positive definite 
         *              based on pass/fail of Cholesky algorithm
         * Parameters:  A - Matrix to test if positive definite
         * Returns:     True if Matrix is symmetric and Cholesky 
         *              algorithm passes, else false
         *              
         */
        {
            if (!is_almost_symmetric(A))
                return false;

            try
            {
                Cholesky(A);
                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
                return false;
            }
        }

        public Matrix Cholesky(Matrix A)
        /* 
         * Purpose:     Compute triangular matrix L that satisfies
         *              A = L*L.t
         * Parameters:  A - Matrix 
         * Returns:     True if Matrix is symmetric and Cholesky 
         *              algorithm passes, else false
         *              
         */
        {
            try
            {
                if (!is_almost_symmetric(A))
                    throw new ArithmeticException("Matrix is not Symmetric");

                double p = 0.0;
                Matrix L = A.Clone();
                for (int k = 0; k < L.cols; k++)
                {
                    if (L[k, k] <= 0)
                        throw new ArithmeticException("Matrix is not Positive Definite");

                    p = L[k, k];
                    L[k, k] = Math.Sqrt(L[k, k]);
                    for (int i = k + 1; i < L.rows; i++)
                    {
                        L[i, k] /= p;
                    }
                    for (int j = k + 1; j < L.rows; j++)
                    {
                        p = (double)L[j, k];
                        for (int i = k + 1; i < L.rows; i++)
                            L[i, j] -= p * L[i, k];
                    }
                }
                
                for (int i = 0; i < L.rows; i++)
                {
                    for (int j = i + 1; j < L.cols; j++)
                        L[i, j] = 0;
                }

                return L;
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
                return null;
            }
        }

        /*
        def Markovitz(mu, A, r_free):
            """Assess Markovitz risk/return.
            Example:
            >>> cov = Matrix.from_list([[0.04, 0.006,0.02], [0.006,0.09, 0.06], [0.02, 0.06, 0.16]])
            >>> mu = Matrix.from_list([[0.10],[0.12],[0.15]])
            >>> r_free = 0.05
            >>> x, ret, risk = Markovitz(mu, cov, r_free)
            >>> print x
            [0.556634..., 0.275080..., 0.1682847...]
            >>> print ret, risk
            0.113915... 0.186747...
            """
         */
        public static void Markovitz(Matrix mu, Matrix A, double r_free, out double[] portfolio, out double portfolio_return, out double portfolio_risk)
        /* 
         * Purpose:     To maximize portfolio expected return for a given amount of portfolio risk and expected return.
         * Parameters:  mu - Expected return.
         *              A - Matrix holding weights of each asset.
         *              r_free - Risk free rate in decimal form.
         * Returns:     portfolio - The proportion of the total portfolio to devote to each asset.
         *              portfolio_return - Percent return of the portfolio.
         *              portfolio_risk - Portfolio return volatility.
         */
        {
            double sum_rows = 0;

            Matrix mu_copy = mu.Clone();
            Matrix x = (1 / A) * (mu_copy - r_free);

            portfolio = new double[x.rows];

            for (int i = 0; i < x.rows; i++)
            {
                sum_rows += x[i, 0];
                portfolio[i] = x[i, 0];
            }

            for (int i = 0; i < x.rows; i++)
                portfolio[i] /= sum_rows;

            x /= sum_rows;

            Matrix port_ret = mu * x;
            Matrix port_risk = x * (A * x);

            portfolio_return = port_ret[0, 0];
            portfolio_risk = Math.Sqrt(port_risk[0, 0]);
        }
        /*

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


        public static ComplexNumber sqrt(double x)
        /* Function: sqrt
         * Purpose: Take the square root of a number.  If the sent value is negative, then an imaginary number is returned.
         * Parameters: x - The number to take the square root of.
         * Returns: A complex number object.
         */
        {
            if (x < 0)
                return new ComplexNumber(0, Math.Sqrt(Math.Abs(x)));
            else
                return new ComplexNumber(Math.Sqrt(x), 0);
        }
    }
}
