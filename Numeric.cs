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

        // Print
        public override string ToString() 
        {
            if (this.imaginary == 0)
                return string.Format("{0}", real);

            if (this.real == 0)
                return string.Format("{0}j", imaginary);
            
            return string.Format("{0} + {1}j", real, imaginary);
        }
    }

    public class MultivariateFunction
    /*
     *###
     *##### OPTIONAL
     *###
     */
    {
        public MultivariateFunction() { }

        public virtual double f(double[] x) { return 0; }

        public double partial(double[] x, int i, double h = 0.01) 
        /* Function: jacobian
         * Purpose: 
         * Parameters: 
         * Returns: 
         */
        {
            double[] x_plus = (double[])x.Clone();
            double[] x_minus = (double[])x.Clone();

            x_plus[i] += h;
            x_minus[i] -= h;

            return (f(x_plus) - f(x_minus)) / (2.0 * h);
        }

        public Matrix jacobian(List<MultivariateFunction> fs, double[] x, double h = 0.0001)
        /* Function: jacobian
         * Purpose: 
         * Parameters: 
         * Returns: 
         */
        {
            Matrix M = new Matrix(fs.Count, x.Count());

            try
            {
                for (int r = 0; r < M.rows; r++)
                {
                    for (int c = 0; c < M.cols; c++)
                    {
                        M[r, c] = fs[r].partial(x, c);
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }

            return M;
        }

        public Matrix hessian(double[] x, double h = 0.001)
        /* Function: hessian
         * Purpose: 
         * Parameters: 
         * Returns: 
         */
        {
            Matrix M = new Matrix(x.Count(), x.Count());

            try
            {
                for (int r = 0; r < M.rows; r++)
                {
                    for (int c = 0; c < M.cols; c++)
                    {
                        double[] x_plus = (double[])x.Clone();
                        double[] x_minus = (double[])x.Clone();

                        x_plus[c] += h;
                        double F_plus = partial(x_plus, r);
                        x_minus[c] -= h;
                        double F_minus = partial(x_minus, r);

                        M[r, c] = (F_plus - F_minus) / (2.0 * h);
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }

            return M;
        }

        public Matrix gradient(double[] x, double h = 0.0001)
        /* Function: gradient
         * Purpose: 
         * Parameters: 
         * Returns: 
         */
        {
            Matrix M = new Matrix(x.Count());

            for (int r = 0; r < M.rows; r++)
            {
                for (int c = 0; c < M.cols; c++)
                {
                    M[r, c] = partial(x, r);
                }
            }

            return M;
        }

        public double[] solve_newton_multi(List<MultivariateFunction> fs, double[] x, double ap = 0.000001, double rp = 0.0001, int ns = 20)
        /* Function:    solve_newton_multi
         * Purpose:     Computes the root of a multidimensional function fs near point x 
         * Parameters:  fs - Multidimensional function that takes an array and returns a scalar
         *              x - Array
         *              ap - Absolute precision
         *              rp - Relative precision
         *              ns - Max number of iterations
         * Returns:     Solution of f(x)=0 as a array
         */
        {
            Matrix x_t = Matrix.from_list(new List<double[]> { x }).Transpose();

            try
            {
                for (int k = 0; k < ns; k++)
                {
                    int n = (x_t.data()).Length;

                    double[] fsx_list = new double[n];
                    //Console.WriteLine("x_t.data()=" + x_t.data()[0] + ", " + x_t.data()[1]);
                    for (int i = 0; i < fs.Count(); i++)
                    {
                        fsx_list[i] = fs[i].f(x_t.data());
                    }
                    Matrix fsx = Matrix.from_list(new List<double[]> { fsx_list }).Transpose();
                    //Console.WriteLine("k=" + k + " fsx=" + fsx);

                    Matrix J = jacobian(fs, x_t.data());

                    Matrix x_old = x_t;

                    x_t = x_t - (1.0 / J) * fsx;

                    if (Numeric.norm(x_t - x_old) < (Math.Max(ap, rp * Numeric.norm(x_t))))
                    {
                        return x_t.data();
                    }
                }
                
                throw new ArithmeticException("No Convergence");
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }

            return null;
        }

        public double[] optimize_newton_multi(double[] x, double ap = 0.000001, double rp = 0.0001, int ns = 20)
        /* Function:    solve_secant
         * Purpose:     Finds extreme of multidimensional function fs near point x 
         * Parameters:  fs - Multidimensional function that takes an array and returns scalar
         *              x - Array
         *              ap - Absolute precision
         *              rp - Relative precision
         *              ns - Max number of iterations
         * Returns:     Solution of f'(x) = 0 as an array and indicates if max or min
         */
        {
            Matrix x_t = Matrix.from_list(new List<double[]> { x }).Transpose();

            try
            {
                for (int k = 0; k < ns; k++)
                {
                    int n = x.Length;

                    double[] fsx_list = new double[n];

                    //Console.WriteLine("x_t.data()=" + x_t.data()[0] + ", " + x_t.data()[1]);
                    for (int i = 0; i < n; i++)
                    {
                        fsx_list[i] = partial(x_t.data(), i);
                    }
                    Matrix fsx = Matrix.from_list(new List<double[]> { fsx_list }).Transpose();
                    //Console.WriteLine("k=" + k + " fsx=" + fsx);

                    Matrix H = hessian(x_t.data());
                    //Console.WriteLine("k=" + k + " H=" + H);

                    Matrix x_old = x_t;

                    x_t = x_t - (1.0 / H) * fsx;

                    if (Numeric.norm(x_t - x_old) < (Math.Max(ap, rp * Numeric.norm(x_t))))
                    {
                        Numeric numeric = new Numeric();
                        try
                        {
                            if (numeric.Cholesky(H) == null)
                            {
                                if (numeric.Cholesky(-H) != null)
                                    Console.WriteLine("maximum");
                            }
                            else
                                Console.WriteLine("minimum");
                        }
                        catch (Exception ex) 
                        {
                            Console.WriteLine(ex.ToString());
                        }

                        return x_t.data();
                    }
                }

                throw new ArithmeticException("No Convergence");
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }

            return null;
        }
    }

    public class Function
    {
        public Function[] func;
        public Matrix c;

        public Function() { }
        public Function(Function[] func, Matrix c) { this.func = func; this.c = c;  }

        public virtual double f(double x) { return 0 * x; }

        public double eval_fitting_function(double x)
        {
            Matrix return_value;
            double sums = 0;

            if (func.Length == 1)
            {
                return_value = c * func[0].f(x);
                return return_value[0, 0];
            }
            else
            {
                for (int i = 0; i < func.Length; i++)
                    sums += func[i].f(x) * c[i, 0];
                return sums;
            }
        }

        public virtual double polynomial(double x, int n)
        {
            double return_val = 0;
            for (int i = 1; i <= n; i++)
                return_val += Math.Pow(x, i);
            return return_val;
        }

        public double Df(double x, double h = 1e-5) { return (f(x + h) - f(x - h)) / (2.0 * h); }

        public double DDf(double x, double h = 1e-5) { return (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h); }

        public double g(double x) { return f(x) + x; }

        public double Dg(double x, double h = 1e-5) { return (g(x + h) - g(x - h)) / (2.0 * h); }

        public double DDg(double x, double h = 1e-5) { return (g(x + h) - 2.0 * g(x) + g(x - h)) / (h * h); }

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

        public double solve_fixed_point(double x, double ap = 0.000001, double rp = 0.0001, int ns = 100)
        /* Function: solve_fixed_point
         * Purpose: 
         * Parameters: 
         * Returns: 
         */
        {
            double old_x;

            try
            {
                for (int k = 0; k < ns; k++)
                {
                    if (Math.Abs(Dg(x)) >= 1)
                        throw new InvalidOperationException("Error D(g)(x) >= 1");
                    
                    old_x = x;
                    x = g(x);

                    if (k > 2 && Math.Abs(old_x - x) < Math.Max(ap, Math.Abs(x) * rp))
                        return x;
                }
                throw new InvalidOperationException("No Convergence");
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }

            return 0.0;
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

        public static void fit_least_squares(double[,] points, Function[] f, out double[] fitting_coef, out double chi2, out Function fitting_f)
        /* 
         * Purpose:     Computes c_j for best linear fit of y[i] \pm dy[i] = fitting_f(x[i]) where fitting_f(x[i]) is \sum_j c_j f[j](x[i]).
         * Parameters:  points - list with points (x,y,dy)
         * Returns:     fitting_coef - column vector with fitting coefficients
         *              chi2 - the chi2 for the fit
         *              fitting_f - The fitting function.
         */
        {
            Matrix A = new Matrix(points.GetUpperBound(0)+1, f.Length);
            Matrix b = new Matrix(points.GetUpperBound(0)+1);
            double weight;

            for (int i = 0; i < A.rows; i++)
            {
                if (points.GetUpperBound(1)+1 > 2)
                    weight = 1.0 / points[i, 2];
                else
                    weight = 1.0;

                b[i, 0] = (double)weight * (float)points[i, 1];

                for (int j = 0; j < A.cols; j++)
                    A[i, j] = weight * f[j].f((float)points[i, 0]);
            }

            Matrix c = (1.0 / (A.Transpose() * A)) * (A.Transpose() * b);
            Matrix chi = A * c - b;
            fitting_coef = new double[c.cols];

            for (int j = 0; j < c.cols; j++)
                    fitting_coef[j] = c[0, j];

            chi2 = Math.Pow(Numeric.norm(chi, 2), 2);
            fitting_f = new Function(f, c);
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

        public static double norm(List<double> x, int p=1)
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

        public static double norm(Matrix A, int p=1)
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
