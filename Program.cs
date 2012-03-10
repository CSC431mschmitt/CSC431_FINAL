using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MatrixLib;

namespace MatrixTester
{
    class MyFunction : Function { public override double f(double x) { return (x - 2) * (x + 8); } };
    class MyFunction2 : Function { public override double f(double x) { return (x - 2) * (x - 2) * (x - 2) + x; } };
    class MyFunction3 : Function { public override double f(double x) { return (x * x) - (5.0 * x); } };
    class MyFunction4 : Function { public override double f(double x) { return (x - 2) * (x - 5); } };
    class MyFunction5 : Function { public override double f(double x) { return (x - 2) * (x - 5) / 10; } };
    class MyFunction6 : MultivariateFunction { public override double f(double[] x) { return (2.0 * x[0]) + (3.0 * x[1]) + (5.0 * x[1] * x[2]); } };
    class MyFunction7 : MultivariateFunction { public override double f(double[] x) { return x[0] * x[1] - 2; } };
    class MyFunction8 : MultivariateFunction { public override double f(double[] x) { return x[0] - 3.0 * x[1]*x[1]; } };
    class MyFunction9 : MultivariateFunction { public override double f(double[] x) { return 2.0 * x[0]; } };
    class MyFunction10 : MultivariateFunction { public override double f(double[] x) { return x[0] + x[1]; } };
    class MyFunction11 : MultivariateFunction { public override double f(double[] x) { return x[0] + (x[1] * x[1]) - 2; } };
    class MyFunction12 : MultivariateFunction { public override double f(double[] x) { return -1.0 * ( Math.Pow(x[0] - 2, 2) + Math.Pow(x[1] - 3, 2) ); } };
    class MyFunction13 : MultivariateFunction { public override double f(double[] x) { return Math.Pow(x[0] - 2, 2) + Math.Pow(x[1] - 3, 2); } };
    class Quadratic0 : Function { public override double f(double x) { return Math.Pow(x, 0); } };
    class Quadratic1 : Function { public override double f(double x) { return Math.Pow(x, 1); } };
    class Quadratic2 : Function { public override double f(double x) { return Math.Pow(x, 2); } };

    class Program
    {
        public static Matrix m_from_list;
        public static List<double[]> data;

        /* This will be for running tests. */
        static void Main(string[] args)
        {
            data = new List<double[]>() { new double[] { 1, 4, 7 }, new double[] { 3, 11, 19 } };
            m_from_list = Matrix.from_list(data);
            //Program.test_from_list();
            //Program.test_as_list();     ??????
            //Program.test_identity();
            //Program.test_diagonal();
            //Program.test_row();
            //Program.test_col();
            //Program.test_matrix_add();       ?????
            //Program.test_matrix_sub();       ?????
            //Program.test_matrix_mult();      ?????
            Console.ReadLine();
        }

        public static void test_from_list()
        {
            Console.WriteLine("\nTesting from_list(v) ...\n");
            Console.WriteLine("\tExpecting:\t[[1, 4, 7], [3, 11, 19]]");
            data = new List<double[]>() { new double[] { 1, 4, 7 }, new double[] { 3, 11, 19 } };
            m_from_list = Matrix.from_list(data);
            Console.WriteLine("\t   Result:\t" + m_from_list);
        }

        public static void test_as_list()
        {
            Console.WriteLine("\nTesting as_list(A) ...\n");
            Console.WriteLine("\tExpecting:\t[[1, 4, 7], [3, 11, 19]]");

             ////Test def as_list(A)
            //List<double[]> m_as_list = m_from_list.as_list();
            //List<double[]> m_as_list2 = (m_from_list.Transpose()).data;
            //Console.Write("[");
            //for (int j = 0; j < m_as_list2.Count; j++)
            //{
            //    if (j > 0)
            //        Console.Write(", ");

            //    Console.Write("[");
            //    double[] values = m_as_list2[j];
            //    for (int k = 0; k < values.Length; k++)
            //    {
            //        if (k > 0)
            //            Console.Write(", ");
            //        Console.Write(values[k]);
            //    }
            //    Console.Write("]");
            //}
            //Console.WriteLine("]");
            Console.WriteLine("\t   Result:\t" + m_from_list);
        }

        public static void test_identity()
        {
            Console.WriteLine("\nTesting identity(rows) ...\n");
            Console.WriteLine("\tExpecting:\t[[1, 0, 0], [0, 1, 0], [0, 0, 1]]");
            Matrix m_identity = Matrix.identity(3, 1, 0);
            Console.WriteLine("\t   Result:\t" + m_identity);
        }

        public static void test_diagonal()
        {
            Console.WriteLine("\nTesting diagonal(array[]) ...\n");
            Console.WriteLine("\tExpecting:\t[[5, 0, 0, 0], [0, 6, 0, 0], [0, 0, 7, 0], [0, 0, 0, 8]]");
            Matrix m_diagonal = Matrix.diagonal(new double[] { 5, 6, 7, 8 });
            Console.WriteLine("\t   Result:\t" + m_diagonal);
        }

        public static void test_row()
        {
            Console.WriteLine("\nTesting row(row_num) ...\n");
            Console.WriteLine("\tExpecting (print row 0 of matrix):\t[[1, 4, 7]]");
            data = new List<double[]>() { new double[] { 1, 4, 7 }, new double[] { 3, 11, 19 } };
            m_from_list = Matrix.from_list(data);
            Console.WriteLine("\t   Result (print row 0 of matrix):\t" + m_from_list.row(0));

            Console.WriteLine("\n\tExpecting (print row 1 of matrix):\t[[3, 11, 19]]");
            Console.WriteLine("\t   Result (print row 1 of matrix):\t" + m_from_list.row(1));
        }

        public static void test_col()
        {
            Console.WriteLine("\nTesting col(col_num) ...\n");
            Console.WriteLine("\tExpecting (print col 0 of matrix):\t[[1], [3]]");
            data = new List<double[]>() { new double[] { 1, 4, 7 }, new double[] { 3, 11, 19 } };
            m_from_list = Matrix.from_list(data);
            Console.WriteLine("\t   Result (print col 0 of matrix):\t" + m_from_list.col(0));

            Console.WriteLine("\n\tExpecting (print col 1 of matrix):\t[[4], [11]]");
            Console.WriteLine("\t   Result (print col 1 of matrix):\t" + m_from_list.col(1));
        }

        public static void test_matrix_add()
        {
            Console.WriteLine("\nTesting __add__(A, B) ...\n");
            Matrix A = m_from_list;
            List<double[]>  Blist = new List<double[]>() { new double[] { 4, 3 }, new double[] { 2, 1 } };
            Matrix B = Matrix.from_list(Blist);
            Console.WriteLine("\tA: " + A.ToString());
            Console.WriteLine("\tB: " + B.ToString());
            Console.WriteLine("\n\tExpecting A + B:\t[[???]]");
            Console.WriteLine("\t   Result A + B:\t" + (A + B));
        }

        public static void test_matrix_sub()
        {
            Console.WriteLine("\nTesting __sub__(A, B) ...\n");
            Matrix A = m_from_list;
            List<double[]> Blist = new List<double[]>() { new double[] { 1 }, new double[] { 2 } };
            Matrix B = Matrix.from_list(Blist);
            Console.WriteLine("\tA: " + A.ToString());
            Console.WriteLine("\tB: " + B.ToString());
            Console.WriteLine("\n\tExpecting B - A:\t[[???]]");
            Console.WriteLine("\t   Result B - A:\t" + (B - A));

            Console.WriteLine("\nTesting __neg__(A) ...\n");
            Console.WriteLine("\tA: " + A.ToString());
            Console.WriteLine("\n\tExpecting -A:\t[[-1, -4, -7], [-3, -11, -19]]");
            Console.WriteLine("\t   Result -A:\t" + (-A));
        }

        public static void test_matrix_mult()
        {
            Console.WriteLine("\nTesting __mult__(A, B) ...\n");
            Matrix A = m_from_list;
            List<double[]> Blist = new List<double[]>() { new double[] { 1 }, new double[] { 2 } };
            Matrix B = Matrix.from_list(Blist);
            Console.WriteLine("\tA: " + A.ToString());
            Console.WriteLine("\tB: " + B.ToString());
            Console.WriteLine("\n\tExpecting A * B:\t[[???]]");
            Console.WriteLine("\t   Result A * B:\t" + (A * B));

            Console.WriteLine("\nTesting __mult__(c1, c2) ...\n");
            Matrix c1 = Matrix.from_list( new List<double[]>() { new double[] {3}, new double[] {2}, new double[] {1} } );
            Matrix c2 = Matrix.from_list( new List<double[]>() { new double[] {4}, new double[] {5}, new double[] {6} });
            Console.WriteLine("\tc1: " + c1.ToString());
            Console.WriteLine("\tc2: " + c2.ToString());
            Console.WriteLine("\n\tExpecting c1 * c2:\t[[28]]");
            Console.WriteLine("\t   Result c1 * c2:\t" + (c1 * c2));

            Console.WriteLine("\nTesting __mult__(c3, x) ...\n");
            Matrix c3 = m_from_list;
            Matrix c4 = Matrix.from_list(new List<double[]>() { new double[] { 4 }, new double[] { 5 }, new double[] { 6 } });
            double x = 5.0;
            Console.WriteLine("\tc3: " + c3.ToString());
            Console.WriteLine("\tc4: " + c4.ToString());
            Console.WriteLine("\n\tExpecting c3 * x:\t[[5, 20, 35], [15, 55, 95]]");
            Console.WriteLine("\t   Result c3 * x:\t" + (c3 * x));
            Console.WriteLine("\n\tExpecting c4 * x:\t[[20], [25], [30]]");
            Console.WriteLine("\t   Result c4 * x:\t" + (c4 * x));
        }
            
            ////Test swap_rows(A, i, j)
            //Console.WriteLine("B: " + B.ToString());
            //B.swap_rows(0, 1);
            //Console.WriteLine("B after swap rows: " + B.ToString());

            ////test Matrix.data()
            //Matrix N2 = Matrix.from_list(new List<double[]>() { new double[] { 2, 4, 2, 1 }, new double[] { 3, 1, 5, 2 }, new double[] { 1, 2, 3, 3 }, new double[] { 0, 6, 1, 2 } });
            //double[] list = N2.data();
            //string t = "[";
            //for (int i = 0; i < list.Count(); i++)
            //{
            //    if (i > 0)
            //        t += ", ";

            //    t += list[i];
            //}
            //t += "]";
            //Console.WriteLine(t);

            ////Test Transpose property
            //Matrix A1 = Matrix.from_list(new List<double[]>() { new double[] { 1, 2, 3 }, new double[] { 4, 5, 6 } });
            //Console.WriteLine(A1.ToString());
            //Console.WriteLine(A1.Transpose());

            ////Test Clone Matrix
            //Matrix b1 = Matrix.from_list( new List<double[]>() { new double[] { 4, 3 }, new double[] { 2, 1 } } );
            //Matrix b1_copy = b1.Clone();
            //Console.WriteLine("b1: " + b1.ToString());
            //Console.WriteLine("b1_copy: " + b1_copy.ToString());
            //b1[0, 0] = 5.0;
            //Console.WriteLine("b1: " + b1.ToString());
            //Console.WriteLine("b1_copy: " + b1_copy.ToString());
            
            //Test Matrix Division
            //Matrix A = Matrix.from_list(new List<double[]>() { new double[] { 1, 2 }, new double[] { 4, 9 } });
            //Matrix B = Matrix.from_list(new List<double[]>() { new double[] { 5, 2 }, new double[] { 1, 1 } });

            //Console.WriteLine("1 / A: " + 1 / A);
            //Console.WriteLine("A / A: " + A / A);
            //Console.WriteLine("A / 2: " + A / 2);

            //A = Matrix.from_list(new List<double[]>() { new double[] { 1, 2, 2 }, new double[] { 4, 4, 2 }, new double[] { 4, 6, 4 } });
            //B = Matrix.from_list(new List<double[]>() { new double[] { 3 }, new double[] { 6 }, new double[] { 10 } });
            //Matrix x = (1 / A) * B;
            //Console.WriteLine("x: " + x);
            
            //Test Function Class
            //MyFunction mFunction = new MyFunction();
            //MyFunction2 mFunction2 = new MyFunction2();

            //Test Function Return
            //Console.WriteLine(mFunction.f(2)); // f(2) = 0
            //Console.WriteLine(mFunction.f(5)); // f(5) = 39

            //Test Function First Derivative
            //Console.WriteLine(mFunction.Df(2)); // Df(2) = 10
            //Console.WriteLine(mFunction.Df(5)); // Df(5) = 16

            //Test Function Second Derivative
            //Console.WriteLine(mFunction.DDf(2)); // DDf(2) ~ 2
            //Console.WriteLine(mFunction.DDf(5)); // DDf(5) ~ 2

            //Test Function Solve Newton
            //Console.WriteLine(mFunction.solve_newton(0.5)); //Should roughly equal 2.
            //Console.WriteLine(mFunction.optimize_newton(0.5)); //Should roughly equal -3.

            //Test Function Solve Newton Stabilized
            //Console.WriteLine(mFunction.solve_newton_stabilized(-0.5, 9)); //Should roughly equal 2.000000002707719.
            //Console.WriteLine(mFunction2.solve_newton_stabilized(-10, 9)); //Should roughly equal 0.9999999946201354.

            //Test Function Solve Bisection
            //Console.WriteLine(mFunction.solve_bisection(-0.5, 9)); //Should roughly equal 2.0000267028808594.
            //Console.WriteLine(mFunction2.solve_bisection(-10, 9)); //Should roughly equal 0.9999942779541016.
            

            //Test is_almost_symmetric, expect success
            //Numeric n = new Numeric();
            //Matrix B = Matrix.from_list(new List<double[]>() { new double[] { 1, 7, 3 }, new double[] { 7, 4, -5 }, new double[] { 3, -5, 6 } });
            //Console.WriteLine("A is_almost_symmetric (expect success): " + n.is_almost_symmetric(B));
            
            ////Test is_almost_symmetric, expect fail
            //Matrix A = Matrix.from_list(new List<double[]>() { new double[] { 1, 2 }, new double[] { 3, 4 } });
            //Console.WriteLine("A is_almost_symmetric (expect fail): " + n.is_almost_symmetric(A));
            
            //Test norm(List<double>x, p=1)
            //Numeric n = new Numeric();
            //List<double[]> list1 = new List<double[]>() { new double[] { 2 }, new double[] { 3 }, new double[] { 4 } };
            //List<double[]> list0 = new List<double[]>() { new double[] { 1, 2, 3 }, new double[] { 4, 5, 6 }, new double[] { 7, 8, 9 } };
            //List<double> list2 = new List<double>( new double[] { 2, 3, 4 } );
            //List<double[]> list3 = new List<double[]>() { new double[] { 2, 3, 4 } };
            //Matrix A = Matrix.from_list(list3);
            //try
            //{
            //    Console.WriteLine(Matrix.from_list(list1));
            //    Console.WriteLine(Matrix.from_list(list3));
            //    Console.WriteLine("1-norm of list2: " + n.norm(list2, 1));
            //    Console.WriteLine("2-norm of Matrix " + Matrix.from_list(list0) + ": " + n.norm(Matrix.from_list(list0), 2));
            //}
            //catch (Exception e)
            //{
            //    Console.WriteLine("{0} Exception caught.", e.ToString());
            //}
                                    
            ////Test condition_number(Matrix A)
            //Numeric n = new Numeric();
            //List<double[]> list1 = new List<double[]>() { new double[] { 1, 2 }, new double[] { 3, 4 } };
            //try
            //{
            //    Console.WriteLine("condition number of Matrix " + Matrix.from_list(list1) + ": " + n.condition_number(Matrix.from_list(list1)));
            //}
            //catch (Exception e)
            //{
            //    Console.WriteLine("{0} Exception caught.", e.ToString());
            //}
            ////Test condition_number(f, x=None, h=0.000001)
            //try
            //{
            //    MyFunction3 fn = new MyFunction3();
            //    Console.WriteLine("condition number of function " + fn + ": " + n.condition_number(fn, 1));
            //}
            //catch (Exception e)
            //{
            //    Console.WriteLine("{0} Exception caught.", e.ToString());
            //}


            ////Test exp(Matrix A)
            //Numeric n = new Numeric();
            //List<double[]> list1 = new List<double[]>() { new double[] { 1, 2 }, new double[] { 3, 4 } };
            //Console.WriteLine("exp(" + Matrix.from_list(list1) + "): " + exp(Matrix.from_list(list1)));

            //Test is_positive_definite(Matrix A)
            //Numeric n = new Numeric();
            //List<double[]> list1 = new List<double[]>() { new double[] { 1, 2 }, new double[] { 2, 1 } }; //false
            //List<double[]> list2 = new List<double[]>() { new double[] { 2, -1, 0 }, new double[] { -1, 2, -1 }, new double[] { 0, -1, 2 } }; //true
            //Console.WriteLine("is_positive_definite(" + Matrix.from_list(list1) + "), expect false: " + n.exp(Matrix.from_list(list1)));
            //Console.WriteLine("is_positive_definite(" + Matrix.from_list(list2) + "), expect true: " + n.exp(Matrix.from_list(list2)));

            ////Test Cholesky and is_almost_zero
            //Numeric n = new Numeric();
            //List<double[]> list3 = new List<double[]>() { new double[] { 4, 2, 1 }, new double[] { 2, 9, 3 }, new double[] { 1, 3, 16 } };
            //Matrix A = Matrix.from_list(list3);
            //Matrix L = n.Cholesky(A);
            //Console.WriteLine("Cholesky Matrix: " + L);
            //Console.WriteLine("is_almost_zero(A - L*L.t), expect true: " + n.is_almost_zero(A - L * L.Transpose()));

            //Test solvers/optimizers
            //MyFunction4 func = new MyFunction4();
            ////solve_secant
            //Console.WriteLine("solve_secant(f,1.0), f(x)=(x-2)*(x-5): " + func.solve_secant(1.0)); //2.0

            ////optimize_secant
            //Console.WriteLine("optimize_secant(f,3.0), f(x)=(x-2)*(x-5): " + func.optimize_secant(1.0)); //3.5

            ////optimize_bisection
            //Console.WriteLine("optimize_bisection(f,2.0,5.0), f(x)=(x-2)*(x-5): " + func.optimize_bisection(2.0, 5.0)); //3.5

            //optimize_golden_search
            //Console.WriteLine("optimize_golden_search(f,2.0,5.0), f(x)=(x-2)*(x-5): " + func.optimize_golden_search(2.0, 5.0)); //3.5
            
            ////Test Markowitz function
            //List<double[]> listMark = new List<double[]>() { new double[] { 0.04, 0.006, 0.02 }, new double[] { 0.006, 0.09, 0.06 }, new double[] { 0.02, 0.06, 0.16 } };
            //Matrix cov = Matrix.from_list(listMark);
            //List<double[]> listMu = new List<double[]>() { new double[] { 0.10 }, new double[] { 0.12 }, new double[] { 0.15 } };
            //Matrix mu = Matrix.from_list(listMu);
            //double r_free = 0.05, portfolio_return, portfolio_risk;
            //double[] portfolio;
            //Numeric.Markovitz(mu, cov, r_free, out portfolio, out portfolio_return, out portfolio_risk);
            
            //foreach (double s in portfolio)
            //    Console.WriteLine(s);
            //Console.WriteLine("\n" + portfolio_return + "\t" + portfolio_risk);

            //// TEST ComplexNumber object and SQRT function.
            //ComplexNumber c1 = new ComplexNumber(3, 0);
            //ComplexNumber c2 = new ComplexNumber(0, 4);
            //ComplexNumber c3 = new ComplexNumber(6, 7);
            //ComplexNumber c4 = Numeric.sqrt(-9.0);
            //Console.WriteLine(c1);
            //Console.WriteLine(c2);
            //Console.WriteLine(c3);
            //Console.WriteLine(c4);
            //Console.WriteLine(Numeric.sqrt(-45.0));
            //Console.WriteLine(Numeric.sqrt(45.0));
            //Console.WriteLine(Numeric.sqrt(45));

            ////TEST Norm of Matrix.
            //Matrix N = Matrix.from_list(new List<double[]>() { new double[] { 3, 5, 7 }, new double[] { 2, 6, 4 }, new double[] { 0, 2, 8 } });
            //Matrix N2 = Matrix.from_list(new List<double[]>() { new double[] { 2, 4, 2, 1 }, new double[] { 3, 1, 5, 2 }, new double[] { 1, 2, 3, 3 }, new double[] { 0, 6, 1, 2 } });
            //Console.WriteLine(Numeric.norm(N).ToString());
            //Console.WriteLine(Numeric.norm(N2).ToString());

            ///TEST solve_fixed_point
            //MyFunction5 f5 = new MyFunction5();
            //Console.WriteLine(Math.Round(f5.solve_fixed_point(1.0, 0.000001, 0), 4));  //Expecting 2.

            ////Test partial
            //MyFunction6 f6 = new MyFunction6();
            //double[] x = new double[] {1, 1, 1};
            //double df0 = f6.partial(x, 0);
            //double df1 = f6.partial(x, 1);
            //double df2 = f6.partial(x, 2);
            //Console.WriteLine("partial df0: " + Math.Round(df0, 4)); //2.0
            //Console.WriteLine("partial df1: " + Math.Round(df1, 4)); //8.0
            //Console.WriteLine("partial df2: " + Math.Round(df2, 4)); //5.0

            ////test jacobian
            //MultivariateFunction multi = new MultivariateFunction();
            //List<MultivariateFunction> fs = new List<MultivariateFunction>() { new MyFunction6(), new MyFunction9() } ;
            //double[] y = new double[] { 1, 1, 1 };
            //Console.WriteLine("jacobian(fs, x=[1,1,1]): " + multi.jacobian(fs, y) ); //[[1.9999999..., 7.999999..., 4.9999999...], [1.9999999..., 0.0, 0.0]]

            ////test gradient
            //MyFunction6 f6 = new MyFunction6();
            //double[] x = new double[] {1, 1, 1};
            //Console.WriteLine("gradient (f, x=[1, 1, 1]): " + f6.gradient(x) ); //[[1.999999...], [7.999999...], [4.999999...]]
            
            ////test hessian
            //MyFunction6 f6 = new MyFunction6();
            //double[] x = new double[] { 1, 1, 1 };
            //Console.WriteLine("hessian (f, x=[1, 1, 1]): " + f6.hessian(x) ); //[[0.0, 0.0, 0.0], [0.0, 0.0, 5.000000...], [0.0, 5.000000..., 0.0]]

            ////test solve_newton_multi
            //MultivariateFunction multi = new MultivariateFunction();
            //List<MultivariateFunction> fs1 = new List<MultivariateFunction>() { new MyFunction7(), new MyFunction8() };
            //double[] x = new double[] { 1, 1 };
            //double[] result = multi.solve_newton_multi(fs1, x);
            //Console.WriteLine("solve_newton_multi(fs, x=[1, 1]: [" + result[0] + ", " + result[1] + "]"); //[2.2894284851066793, 0.87358046473632422]

            //List<MultivariateFunction> fs2 = new List<MultivariateFunction>() { new MyFunction10(), new MyFunction11() };
            //double[] y = new double[] { 0, 0 };
            //double[] result2 = multi.solve_newton_multi(fs2, y);
            //Console.WriteLine("solve_newton_multi(fs2, x=[0, 0]: [" + result2[0] + ", " + result2[1] + "]");//[1.0000000006984919, -1.0000000006984919]
            //y = new double[] { 1, 1 };
            //result2 = multi.solve_newton_multi(fs2, y);
            //Console.WriteLine("solve_newton_multi(fs2, x=[1, 1]: [" + result2[0] + ", " + result2[1] + "]");//[-2.0000000006984919, 2.0000000006984919]     

            //test optimize_newton_multi
            //MyFunction12 f12 = new MyFunction12();
            //double[] x = new double[] { 0, 0 };
            //double[] result = f12.optimize_newton_multi(x);
            //Console.WriteLine("optimize_newton_multi(fs, x=[0, 0]: [" + result[0] + ", " + result[1] + "]"); //[2.0, 3.0] maximum

            //MyFunction13 f13 = new MyFunction13();
            //double[] y = new double[] { 0, 0 };
            //double[] result2 = f13.optimize_newton_multi(y);
            //Console.WriteLine("optimize_newton_multi(fs, x=[0, 0]: [" + result2[0] + ", " + result2[1] + "]"); //[2.0, 3.0] minimum

            //TEST fit_least_squares
            //  Expecting:
            //   90 2507.89 2506.98
            //   91 2562.21 2562.08
            //   92 2617.02 2617.78
            //   93 2673.15 2674.08
            //   94 2730.75 2730.98
            //   95 2789.18 2788.48
            //   96 2847.58 2846.58
            //   97 2905.68 2905.28
            //   98 2964.03 2964.58
            //   99 3023.5 3024.48
            //double[,] points = new double[100, 3];
            //Function[] f = new Function[3] { new Quadratic0(), new Quadratic1(), new Quadratic2() } ;
            //double[] fitting_coef;
            //double chi2;
            //Function fitting_f;
             
            //for (int i = 0; i < 100; i++)
            //{
            //    points[i, 0] = i;
            //    points[i, 1] = 5 + 0.8*i + 0.3*i*i + Math.Sin(i);
            //    points[i, 2] = 2;
            //}

            //Function.fit_least_squares(points, f, out fitting_coef, out chi2, out fitting_f);

            //for (int i = 90; i < 100; i++)
            //    Console.WriteLine(points[i, 0] + "\t" + Math.Round(points[i, 1], 2) + "\t" + Math.Round(fitting_f.eval_fitting_function(points[i, 0]), 2));


    }
















// ORIGINAL BEFORE MOVING TO TESTING FUNCTIONS


    /*
     * using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MatrixLib;

namespace MatrixTester
{

    class MyFunction : Function
    {
        public override double f(double x) { return (x - 2) * (x + 8); }
    };

    class MyFunction2 : Function
    {
        public override double f(double x) { return (x - 2) * (x - 2) * (x - 2) + x; }
    };

    class MyFunction3 : Function
    {
        public override double f(double x) { return (x * x) - (5.0 * x); }
    };

    class MyFunction4 : Function
    {
        public override double f(double x) { return (x - 2) * (x - 5); }
    };

    class MyFunction5 : Function
    {
        public override double f(double x) { return (x - 2) * (x - 5) / 10; }
    };

    class MyFunction6 : MultivariateFunction
    {
        public override double f(double[] x) { return (2.0 * x[0]) + (3.0 * x[1]) + (5.0 * x[1] * x[2]); }
    };

    class MyFunction7 : MultivariateFunction
    {
        public override double f(double[] x) { return x[0] * x[1] - 2; }
    };

    class MyFunction8 : MultivariateFunction
    {
        public override double f(double[] x) { return x[0] - 3.0 * x[1]*x[1]; }
    };

    class MyFunction9 : MultivariateFunction
    {
        public override double f(double[] x) { return 2.0 * x[0]; }
    };

    class MyFunction10 : MultivariateFunction
    {
        public override double f(double[] x) { return x[0] + x[1]; }
    };

    class MyFunction11 : MultivariateFunction
    {
        public override double f(double[] x) { return x[0] + (x[1] * x[1]) - 2; }
    };

    class MyFunction12 : MultivariateFunction
    {
        public override double f(double[] x) { return -1.0 * ( Math.Pow(x[0] - 2, 2) + Math.Pow(x[1] - 3, 2) ); }
    };

    class MyFunction13 : MultivariateFunction
    {
        public override double f(double[] x) { return Math.Pow(x[0] - 2, 2) + Math.Pow(x[1] - 3, 2); }
    };

    class Quadratic0 : Function
    {
        public override double f(double x) { return Math.Pow(x, 0); }
    };

    class Quadratic1 : Function
    {
        public override double f(double x) { return Math.Pow(x, 1); }
    };

    class Quadratic2 : Function
    {
        public override double f(double x) { return Math.Pow(x, 2); }
    };

    class Program
    {
        
        static void Main(string[] args)
        {
            ////Test def from_list(v)
            //List<double[]> data = new List<double[]>(){new double[]{1, 4, 7}, new double[]{3, 11, 19}};
            //Matrix m_from_list = Matrix.from_list(data);
            //Console.WriteLine("from list: " + m_from_list);

            ////Test def as_list(A)
            //List<double[]> m_as_list = m_from_list.as_list();
            //List<double[]> m_as_list2 = (m_from_list.Transpose()).data;
            //Console.Write("[");
            //for (int j = 0; j < m_as_list2.Count; j++)
            //{
            //    if (j > 0)
            //        Console.Write(", ");

            //    Console.Write("[");
            //    double[] values = m_as_list2[j];
            //    for (int k = 0; k < values.Length; k++)
            //    {
            //        if (k > 0)
            //            Console.Write(", ");
            //        Console.Write(values[k]);
            //    }
            //    Console.Write("]");
            //}
            //Console.WriteLine("]");

            ////Test def identity(rows)
            //Matrix m_identity = Matrix.identity(3, 1, 0);
            //Console.WriteLine("identity matrix: " + m_identity);

            ////Test def diagonol(array[])
            //Matrix m_diagonol = Matrix.diagonal(new double[] { 5, 6, 7, 8 });
            //Console.WriteLine("diagonol matrix : " + m_diagonol);

            ////Test def row(row_num)
            //Console.WriteLine("print row 0 of matrix: " + m_from_list.row(0));

            ////Test def row(row_num)
            //Console.WriteLine("print row 1 of matrix: " + m_from_list.row(1));

            ////Test def col(col_num)
            //Console.WriteLine("print col 0 of matrix: " + m_from_list.col(0));

            ////Test def col(col_num)
            //Console.WriteLine("print col 1 of matrix: " + m_from_list.col(1));

            ////Test __mult__(A, B)
            //Matrix A = m_from_list;
            //List<double[]> Blist = new List<double[]>() { new double[] { 1 }, new double[] { 2 } };
            //Matrix B = Matrix.from_list(Blist);
            //Console.WriteLine("A: " + A.ToString());
            //Console.WriteLine("B: " + B.ToString());
            //Console.WriteLine("A * B: " + (A*B) );

            ////Test __mult__(c1, c2)
            //Matrix c1 = Matrix.from_list( new List<double[]>() { new double[] {3}, new double[] {2}, new double[] {1} } );
            //Matrix c2 = Matrix.from_list( new List<double[]>() { new double[] {4}, new double[] {5}, new double[] {6} });
            //Console.WriteLine("c1: " + c1.ToString());
            //Console.WriteLine("c2: " + c2.ToString());
            //Console.WriteLine("c1 * c2: " + (c1 * c2));

            ////Test __mult__(c3, x)
            //Matrix c3 = m_from_list;
            //Matrix c4 = Matrix.from_list(new List<double[]>() { new double[] { 4 }, new double[] { 5 }, new double[] { 6 } });
            //double x = 5.0;
            //Console.WriteLine("c3: " + c3.ToString());
            //Console.WriteLine("c4: " + c4.ToString());
            //Console.WriteLine("c3 * x: " + (c3 * x));
            //Console.WriteLine("c4 * x: " + (c4 * x));

            ////Test __add__(A, B)
            //Blist = new List<double[]>() { new double[] { 4, 3 }, new double[] { 2, 1 } };
            //B = Matrix.from_list(Blist);
            //Console.WriteLine("A: " + A.ToString());
            //Console.WriteLine("B: " + B.ToString());
            //Console.WriteLine("A + B: " + (A + B));

            ////Test __sub__(A, B)
            //Console.WriteLine("A: " + A.ToString());
            //Console.WriteLine("B: " + B.ToString());
            //Console.WriteLine("B - A: " + (B - A));
            
            ////Test __neg__(A)
            //Console.WriteLine("-A: " + (-A));
            
            ////Test swap_rows(A, i, j)
            //Console.WriteLine("B: " + B.ToString());
            //B.swap_rows(0, 1);
            //Console.WriteLine("B after swap rows: " + B.ToString());

            ////test Matrix.data()
            //Matrix N2 = Matrix.from_list(new List<double[]>() { new double[] { 2, 4, 2, 1 }, new double[] { 3, 1, 5, 2 }, new double[] { 1, 2, 3, 3 }, new double[] { 0, 6, 1, 2 } });
            //double[] list = N2.data();
            //string t = "[";
            //for (int i = 0; i < list.Count(); i++)
            //{
            //    if (i > 0)
            //        t += ", ";

            //    t += list[i];
            //}
            //t += "]";
            //Console.WriteLine(t);

            ////Test Transpose property
            //Matrix A1 = Matrix.from_list(new List<double[]>() { new double[] { 1, 2, 3 }, new double[] { 4, 5, 6 } });
            //Console.WriteLine(A1.ToString());
            //Console.WriteLine(A1.Transpose());

            ////Test Clone Matrix
            //Matrix b1 = Matrix.from_list( new List<double[]>() { new double[] { 4, 3 }, new double[] { 2, 1 } } );
            //Matrix b1_copy = b1.Clone();
            //Console.WriteLine("b1: " + b1.ToString());
            //Console.WriteLine("b1_copy: " + b1_copy.ToString());
            //b1[0, 0] = 5.0;
            //Console.WriteLine("b1: " + b1.ToString());
            //Console.WriteLine("b1_copy: " + b1_copy.ToString());
            
            //Test Matrix Division
            //Matrix A = Matrix.from_list(new List<double[]>() { new double[] { 1, 2 }, new double[] { 4, 9 } });
            //Matrix B = Matrix.from_list(new List<double[]>() { new double[] { 5, 2 }, new double[] { 1, 1 } });

            //Console.WriteLine("1 / A: " + 1 / A);
            //Console.WriteLine("A / A: " + A / A);
            //Console.WriteLine("A / 2: " + A / 2);

            //A = Matrix.from_list(new List<double[]>() { new double[] { 1, 2, 2 }, new double[] { 4, 4, 2 }, new double[] { 4, 6, 4 } });
            //B = Matrix.from_list(new List<double[]>() { new double[] { 3 }, new double[] { 6 }, new double[] { 10 } });
            //Matrix x = (1 / A) * B;
            //Console.WriteLine("x: " + x);
            
            //Test Function Class
            //MyFunction mFunction = new MyFunction();
            //MyFunction2 mFunction2 = new MyFunction2();

            //Test Function Return
            //Console.WriteLine(mFunction.f(2)); // f(2) = 0
            //Console.WriteLine(mFunction.f(5)); // f(5) = 39

            //Test Function First Derivative
            //Console.WriteLine(mFunction.Df(2)); // Df(2) = 10
            //Console.WriteLine(mFunction.Df(5)); // Df(5) = 16

            //Test Function Second Derivative
            //Console.WriteLine(mFunction.DDf(2)); // DDf(2) ~ 2
            //Console.WriteLine(mFunction.DDf(5)); // DDf(5) ~ 2

            //Test Function Solve Newton
            //Console.WriteLine(mFunction.solve_newton(0.5)); //Should roughly equal 2.
            //Console.WriteLine(mFunction.optimize_newton(0.5)); //Should roughly equal -3.

            //Test Function Solve Newton Stabilized
            //Console.WriteLine(mFunction.solve_newton_stabilized(-0.5, 9)); //Should roughly equal 2.000000002707719.
            //Console.WriteLine(mFunction2.solve_newton_stabilized(-10, 9)); //Should roughly equal 0.9999999946201354.

            //Test Function Solve Bisection
            //Console.WriteLine(mFunction.solve_bisection(-0.5, 9)); //Should roughly equal 2.0000267028808594.
            //Console.WriteLine(mFunction2.solve_bisection(-10, 9)); //Should roughly equal 0.9999942779541016.
            

            //Test is_almost_symmetric, expect success
            //Numeric n = new Numeric();
            //Matrix B = Matrix.from_list(new List<double[]>() { new double[] { 1, 7, 3 }, new double[] { 7, 4, -5 }, new double[] { 3, -5, 6 } });
            //Console.WriteLine("A is_almost_symmetric (expect success): " + n.is_almost_symmetric(B));
            
            ////Test is_almost_symmetric, expect fail
            //Matrix A = Matrix.from_list(new List<double[]>() { new double[] { 1, 2 }, new double[] { 3, 4 } });
            //Console.WriteLine("A is_almost_symmetric (expect fail): " + n.is_almost_symmetric(A));
            
            //Test norm(List<double>x, p=1)
            //Numeric n = new Numeric();
            //List<double[]> list1 = new List<double[]>() { new double[] { 2 }, new double[] { 3 }, new double[] { 4 } };
            //List<double[]> list0 = new List<double[]>() { new double[] { 1, 2, 3 }, new double[] { 4, 5, 6 }, new double[] { 7, 8, 9 } };
            //List<double> list2 = new List<double>( new double[] { 2, 3, 4 } );
            //List<double[]> list3 = new List<double[]>() { new double[] { 2, 3, 4 } };
            //Matrix A = Matrix.from_list(list3);
            //try
            //{
            //    Console.WriteLine(Matrix.from_list(list1));
            //    Console.WriteLine(Matrix.from_list(list3));
            //    Console.WriteLine("1-norm of list2: " + n.norm(list2, 1));
            //    Console.WriteLine("2-norm of Matrix " + Matrix.from_list(list0) + ": " + n.norm(Matrix.from_list(list0), 2));
            //}
            //catch (Exception e)
            //{
            //    Console.WriteLine("{0} Exception caught.", e.ToString());
            //}
                                    
            ////Test condition_number(Matrix A)
            //Numeric n = new Numeric();
            //List<double[]> list1 = new List<double[]>() { new double[] { 1, 2 }, new double[] { 3, 4 } };
            //try
            //{
            //    Console.WriteLine("condition number of Matrix " + Matrix.from_list(list1) + ": " + n.condition_number(Matrix.from_list(list1)));
            //}
            //catch (Exception e)
            //{
            //    Console.WriteLine("{0} Exception caught.", e.ToString());
            //}
            ////Test condition_number(f, x=None, h=0.000001)
            //try
            //{
            //    MyFunction3 fn = new MyFunction3();
            //    Console.WriteLine("condition number of function " + fn + ": " + n.condition_number(fn, 1));
            //}
            //catch (Exception e)
            //{
            //    Console.WriteLine("{0} Exception caught.", e.ToString());
            //}


            ////Test exp(Matrix A)
            //Numeric n = new Numeric();
            //List<double[]> list1 = new List<double[]>() { new double[] { 1, 2 }, new double[] { 3, 4 } };
            //Console.WriteLine("exp(" + Matrix.from_list(list1) + "): " + exp(Matrix.from_list(list1)));

            //Test is_positive_definite(Matrix A)
            //Numeric n = new Numeric();
            //List<double[]> list1 = new List<double[]>() { new double[] { 1, 2 }, new double[] { 2, 1 } }; //false
            //List<double[]> list2 = new List<double[]>() { new double[] { 2, -1, 0 }, new double[] { -1, 2, -1 }, new double[] { 0, -1, 2 } }; //true
            //Console.WriteLine("is_positive_definite(" + Matrix.from_list(list1) + "), expect false: " + n.exp(Matrix.from_list(list1)));
            //Console.WriteLine("is_positive_definite(" + Matrix.from_list(list2) + "), expect true: " + n.exp(Matrix.from_list(list2)));

            ////Test Cholesky and is_almost_zero
            //Numeric n = new Numeric();
            //List<double[]> list3 = new List<double[]>() { new double[] { 4, 2, 1 }, new double[] { 2, 9, 3 }, new double[] { 1, 3, 16 } };
            //Matrix A = Matrix.from_list(list3);
            //Matrix L = n.Cholesky(A);
            //Console.WriteLine("Cholesky Matrix: " + L);
            //Console.WriteLine("is_almost_zero(A - L*L.t), expect true: " + n.is_almost_zero(A - L * L.Transpose()));

            //Test solvers/optimizers
            //MyFunction4 func = new MyFunction4();
            ////solve_secant
            //Console.WriteLine("solve_secant(f,1.0), f(x)=(x-2)*(x-5): " + func.solve_secant(1.0)); //2.0

            ////optimize_secant
            //Console.WriteLine("optimize_secant(f,3.0), f(x)=(x-2)*(x-5): " + func.optimize_secant(1.0)); //3.5

            ////optimize_bisection
            //Console.WriteLine("optimize_bisection(f,2.0,5.0), f(x)=(x-2)*(x-5): " + func.optimize_bisection(2.0, 5.0)); //3.5

            //optimize_golden_search
            //Console.WriteLine("optimize_golden_search(f,2.0,5.0), f(x)=(x-2)*(x-5): " + func.optimize_golden_search(2.0, 5.0)); //3.5
            
            ////Test Markowitz function
            //List<double[]> listMark = new List<double[]>() { new double[] { 0.04, 0.006, 0.02 }, new double[] { 0.006, 0.09, 0.06 }, new double[] { 0.02, 0.06, 0.16 } };
            //Matrix cov = Matrix.from_list(listMark);
            //List<double[]> listMu = new List<double[]>() { new double[] { 0.10 }, new double[] { 0.12 }, new double[] { 0.15 } };
            //Matrix mu = Matrix.from_list(listMu);
            //double r_free = 0.05, portfolio_return, portfolio_risk;
            //double[] portfolio;
            //Numeric.Markovitz(mu, cov, r_free, out portfolio, out portfolio_return, out portfolio_risk);
            
            //foreach (double s in portfolio)
            //    Console.WriteLine(s);
            //Console.WriteLine("\n" + portfolio_return + "\t" + portfolio_risk);

            //// TEST ComplexNumber object and SQRT function.
            //ComplexNumber c1 = new ComplexNumber(3, 0);
            //ComplexNumber c2 = new ComplexNumber(0, 4);
            //ComplexNumber c3 = new ComplexNumber(6, 7);
            //ComplexNumber c4 = Numeric.sqrt(-9.0);
            //Console.WriteLine(c1);
            //Console.WriteLine(c2);
            //Console.WriteLine(c3);
            //Console.WriteLine(c4);
            //Console.WriteLine(Numeric.sqrt(-45.0));
            //Console.WriteLine(Numeric.sqrt(45.0));
            //Console.WriteLine(Numeric.sqrt(45));

            ////TEST Norm of Matrix.
            //Matrix N = Matrix.from_list(new List<double[]>() { new double[] { 3, 5, 7 }, new double[] { 2, 6, 4 }, new double[] { 0, 2, 8 } });
            //Matrix N2 = Matrix.from_list(new List<double[]>() { new double[] { 2, 4, 2, 1 }, new double[] { 3, 1, 5, 2 }, new double[] { 1, 2, 3, 3 }, new double[] { 0, 6, 1, 2 } });
            //Console.WriteLine(Numeric.norm(N).ToString());
            //Console.WriteLine(Numeric.norm(N2).ToString());

            ///TEST solve_fixed_point
            //MyFunction5 f5 = new MyFunction5();
            //Console.WriteLine(Math.Round(f5.solve_fixed_point(1.0, 0.000001, 0), 4));  //Expecting 2.

            ////Test partial
            //MyFunction6 f6 = new MyFunction6();
            //double[] x = new double[] {1, 1, 1};
            //double df0 = f6.partial(x, 0);
            //double df1 = f6.partial(x, 1);
            //double df2 = f6.partial(x, 2);
            //Console.WriteLine("partial df0: " + Math.Round(df0, 4)); //2.0
            //Console.WriteLine("partial df1: " + Math.Round(df1, 4)); //8.0
            //Console.WriteLine("partial df2: " + Math.Round(df2, 4)); //5.0

            ////test jacobian
            //MultivariateFunction multi = new MultivariateFunction();
            //List<MultivariateFunction> fs = new List<MultivariateFunction>() { new MyFunction6(), new MyFunction9() } ;
            //double[] y = new double[] { 1, 1, 1 };
            //Console.WriteLine("jacobian(fs, x=[1,1,1]): " + multi.jacobian(fs, y) ); //[[1.9999999..., 7.999999..., 4.9999999...], [1.9999999..., 0.0, 0.0]]

            ////test gradient
            //MyFunction6 f6 = new MyFunction6();
            //double[] x = new double[] {1, 1, 1};
            //Console.WriteLine("gradient (f, x=[1, 1, 1]): " + f6.gradient(x) ); //[[1.999999...], [7.999999...], [4.999999...]]
            
            ////test hessian
            //MyFunction6 f6 = new MyFunction6();
            //double[] x = new double[] { 1, 1, 1 };
            //Console.WriteLine("hessian (f, x=[1, 1, 1]): " + f6.hessian(x) ); //[[0.0, 0.0, 0.0], [0.0, 0.0, 5.000000...], [0.0, 5.000000..., 0.0]]

            ////test solve_newton_multi
            //MultivariateFunction multi = new MultivariateFunction();
            //List<MultivariateFunction> fs1 = new List<MultivariateFunction>() { new MyFunction7(), new MyFunction8() };
            //double[] x = new double[] { 1, 1 };
            //double[] result = multi.solve_newton_multi(fs1, x);
            //Console.WriteLine("solve_newton_multi(fs, x=[1, 1]: [" + result[0] + ", " + result[1] + "]"); //[2.2894284851066793, 0.87358046473632422]

            //List<MultivariateFunction> fs2 = new List<MultivariateFunction>() { new MyFunction10(), new MyFunction11() };
            //double[] y = new double[] { 0, 0 };
            //double[] result2 = multi.solve_newton_multi(fs2, y);
            //Console.WriteLine("solve_newton_multi(fs2, x=[0, 0]: [" + result2[0] + ", " + result2[1] + "]");//[1.0000000006984919, -1.0000000006984919]
            //y = new double[] { 1, 1 };
            //result2 = multi.solve_newton_multi(fs2, y);
            //Console.WriteLine("solve_newton_multi(fs2, x=[1, 1]: [" + result2[0] + ", " + result2[1] + "]");//[-2.0000000006984919, 2.0000000006984919]     

            //test optimize_newton_multi
            //MyFunction12 f12 = new MyFunction12();
            //double[] x = new double[] { 0, 0 };
            //double[] result = f12.optimize_newton_multi(x);
            //Console.WriteLine("optimize_newton_multi(fs, x=[0, 0]: [" + result[0] + ", " + result[1] + "]"); //[2.0, 3.0] maximum

            //MyFunction13 f13 = new MyFunction13();
            //double[] y = new double[] { 0, 0 };
            //double[] result2 = f13.optimize_newton_multi(y);
            //Console.WriteLine("optimize_newton_multi(fs, x=[0, 0]: [" + result2[0] + ", " + result2[1] + "]"); //[2.0, 3.0] minimum

            //TEST fit_least_squares
            //  Expecting:
            //   90 2507.89 2506.98
            //   91 2562.21 2562.08
            //   92 2617.02 2617.78
            //   93 2673.15 2674.08
            //   94 2730.75 2730.98
            //   95 2789.18 2788.48
            //   96 2847.58 2846.58
            //   97 2905.68 2905.28
            //   98 2964.03 2964.58
            //   99 3023.5 3024.48
            //double[,] points = new double[100, 3];
            //Function[] f = new Function[3] { new Quadratic0(), new Quadratic1(), new Quadratic2() } ;
            //double[] fitting_coef;
            //double chi2;
            //Function fitting_f;
             
            //for (int i = 0; i < 100; i++)
            //{
            //    points[i, 0] = i;
            //    points[i, 1] = 5 + 0.8*i + 0.3*i*i + Math.Sin(i);
            //    points[i, 2] = 2;
            //}

            //Function.fit_least_squares(points, f, out fitting_coef, out chi2, out fitting_f);

            //for (int i = 90; i < 100; i++)
            //    Console.WriteLine(points[i, 0] + "\t" + Math.Round(points[i, 1], 2) + "\t" + Math.Round(fitting_f.eval_fitting_function(points[i, 0]), 2));

    */



}
