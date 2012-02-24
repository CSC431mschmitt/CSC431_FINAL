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

        class MyFunction : Function
        {
            public override double f(double x) { return (x - 2) * (x + 8); }
        };

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
        /* Function: solve_bisection
         * Purpose: 
         * Parameters: ???Are a and b matrix or double parameters???
         * Returns: 
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

        public void solve_secant(Function f, double x, double ap = 0.000001, double rp = 0.0001, int ns = 20)
        /* Function: solve_secant
         * Purpose: 
         * Parameters: 
         * Returns: 
         */
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

        public double solve_newton_stabilized(double a, double b, double ap = 0.000001, double rp = 0.0001, int ns = 20)
        /* Function: solve_newton_stabilized
        * Purpose: 
        * Parameters: 
        * Returns: 
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

        public void optimize_bisection(Function f, double a, double b, double ap = 0.000001, double rp = 0.0001, int ns = 100)
        /* Function: optimize_bisection
         * Purpose: 
         * Parameters: 
         * Returns: 
         */
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

        public void optimize_secant(Function f, float x, double ap = 0.000001, double rp = 0.0001, int ns = 100)
        /* Function: optimize_secant
         * Purpose: 
         * Parameters: 
         * Returns: 
         */
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

        public void optimize_newton_stabilized(Function f, double a, double b, double ap = 0.000001, double rp = 0.0001, int ns = 20)
        /* Function: optimize_newton_stabilized
         * Purpose: 
         * Parameters: 
         * Returns: 
         */
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

        public void optimize_golden_search(Function f, double a, double b, double ap = 0.000001, double rp = 0.0001, int ns = 100)
        /* Function: optimize_golden_search
         * Purpose: 
         * Parameters: 
         * Returns: 
         */
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
         * Purpose:     Determine if Matrix A almost symmetric A[i,j] almost equal to A[j, i], within certain precision.
         * Parameters:  A - Matrix to test for symmetry.
         *              ap = 
         * Returns:     Resulting matrix of the matrix subtraction.
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
                        Console.WriteLine("r, c: " + r + "," + c);
                        Console.WriteLine("A[r, c]: " + A[r, c]);
                        Console.WriteLine("A[c, r]: " + A[c, r]);
                        double delta = Math.Abs(A[r, c] - A[c, r]);
                        Console.WriteLine("delta: " + delta);
                        double abs_arc = Math.Abs(A[r, c]);
                        double abs_acr = Math.Abs(A[c, r]);
                        Console.WriteLine("Math.Max(abs_arc, abs_acr)*rp: " + Math.Max(abs_arc, abs_acr) * rp);
                        if ((delta > ap) && (delta > Math.Max(abs_arc, abs_acr) * rp))
                            return false;
                    }
                }
            }

            return true;
        }

        /* do not know purpose of this and it is not called anywhere in numeric_final.py */
        public bool is_almost_zero(Matrix A, double ap = 0.000001, double rp = 0.0001)
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

        /*
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

        def condition_number(f,x=None,h=0.000001):
            if callable(f) and not x is None:
                return D(f,h)(x)*x/f(x)
            elif isinstance(f,Matrix): # if is the Matrix J
                return norm(f)*norm(1/f)
            else:
                raise NotImplementedError

        def exp(x,double ap=0.000001,double rp=0.0001,int ns=40):
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
    
        
        public double sqrt(double x)
        /* Function: sqrt
         * Purpose: 
         * Parameters: 
         * Returns: 
         */
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
                Console.WriteLine(e.ToString());
                a = 0.0;
            }
            
            return a;
        }
    }
}
