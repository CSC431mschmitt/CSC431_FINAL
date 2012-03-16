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
            Program.test_from_list();
            Program.test_as_list();
            Program.test_identity();
            Program.test_diagonal();
            Program.test_row();

            wait_to_continue();

            Program.test_col();
            Program.test_matrix_add();
            Program.test_matrix_sub();
            Program.test_matrix_mult();
            Program.test_matrix_div();

            wait_to_continue();

            Program.test_swap_rows();
            Program.test_data();
            Program.test_transpose();
            Program.test_clone();
            Program.test_function_class();

            wait_to_continue();
            
            Program.test_solve_newton();
            Program.test_optimize_newton();
            Program.test_solve_newton_stabilized();
            Program.test_solve_bisection();
            Program.test_is_almost_symmetric();

            wait_to_continue();

            Program.test_norm();
            Program.test_condition_number();
            Program.test_exp();
            Program.test_is_positive_definite();
            Program.test_cholesky();

            wait_to_continue();

            Program.test_is_almost_zero();
            Program.test_solve_secant();
            Program.test_optimize_secant();
            Program.test_optimize_bisection();

            wait_to_continue();

            Program.test_optimize_golden_search();
            Program.test_markowitz();
            Program.test_sqrt();
            Program.test_solve_fixed_point();
            Program.test_partial();

            wait_to_continue();

            Program.test_jacobian();
            Program.test_gradient();
            Program.test_hessian();
            Program.test_solve_newton_multi();
            Program.test_optimize_newton_multi();
            Program.fit_least_squares();

            wait_to_continue(true);
        }


        public static void wait_to_continue(bool end = false)
        {
            Console.ForegroundColor = ConsoleColor.Green;

            if (end)
                Console.WriteLine("\nTesting Completed!! Press any key to exit.");
            else
                Console.WriteLine("\nPress any key to continue showing test results...");

            Console.ForegroundColor = ConsoleColor.White;
            Console.ReadKey(false);
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
            Console.Write("\tExpecting:\t");

            List<double[]> m_as_list = m_from_list.as_list();
            Console.Write("[");
            int j = 0;
            foreach(double[] values in m_as_list)
            {
                if (j++ > 0)
                    Console.Write(", ");
                
                Console.Write("[" + string.Join(", ", values) + "]" );
            }
            Console.WriteLine("]");
            

            Console.Write("\t   Result:\t");
            int i = 0;
            Console.Write("[");
            foreach (double[] s in m_as_list)
            {
                if (i++ > 0) Console.Write(", "); 
                Console.Write("[" + string.Join(", ", s) + "]");
            }
            Console.Write("]");
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
            Matrix A = Matrix.from_list(new List<double[]>() { new double[] { 4, 3 }, new double[] { 2, 1 }});
            Matrix B = Matrix.from_list(new List<double[]>() { new double[] { 2, 8 }, new double[] { 4, 1 }});
            Console.WriteLine("\tA: " + A.ToString());
            Console.WriteLine("\tB: " + B.ToString());
            Console.WriteLine("\n\tExpecting A + B:\t[[6, 11], [6, 2]]");
            Console.WriteLine("\t   Result A + B:\t" + (A + B));
        }

        public static void test_matrix_sub()
        {
            Console.WriteLine("\nTesting __sub__(A, B) ...\n");
            Matrix A = Matrix.from_list(new List<double[]>() { new double[] { 4, 3 }, new double[] { 2, 1 }});
            Matrix B = Matrix.from_list(new List<double[]>() { new double[] { 2, 8 }, new double[] { 4, 1 }});
            Console.WriteLine("\tA: " + A.ToString());
            Console.WriteLine("\tB: " + B.ToString());
            Console.WriteLine("\n\tExpecting B - A:\t[[-2, 5], [2, 0]]");
            Console.WriteLine("\t   Result B - A:\t" + (B - A));

            A = Matrix.from_list(new List<double[]>() { new double[] { 1, 4, 7 }, new double[] { 3, 11, 19 } });
            Console.WriteLine("\nTesting __neg__(A) ...\n");
            Console.WriteLine("\tA: " + A.ToString());
            Console.WriteLine("\n\tExpecting -A:\t[[-1, -4, -7], [-3, -11, -19]]");
            Console.WriteLine("\t   Result -A:\t" + (-A));
        }

        public static void test_matrix_mult()
        {
            Console.WriteLine("\nTesting __mult__(Matrix A, Matrix B) ...\n");

            Matrix A = Matrix.from_list(new List<double[]>() { new double[] { 1, 4, 7 }, new double[] { 3, 11, 19 } });
            Matrix B = Matrix.from_list(new List<double[]>() { new double[] { 1 }, new double[] { 2 }, new double[] { 3 } });
                        
            Console.WriteLine("\tA: " + A.ToString());
            Console.WriteLine("\tB: " + B.ToString());
            Console.WriteLine("\n\tExpecting A * B:\t[[30.0], [82.0]]");
            Console.WriteLine("\t   Result A * B:\t" + (A * B));

            Console.WriteLine("\nTesting __mult__(Matrix c1, Matrix c2) ...\n");
            Matrix c1 = Matrix.from_list(new List<double[]>() { new double[] { 3 }, new double[] { 2 }, new double[] { 1 } });
            Matrix c2 = Matrix.from_list(new List<double[]>() { new double[] { 4 }, new double[] { 5 }, new double[] { 6 } });
            Console.WriteLine("\tc1: " + c1.ToString());
            Console.WriteLine("\tc2: " + c2.ToString());
            Console.WriteLine("\n\tExpecting c1 * c2:\t[[28]]");
            Console.WriteLine("\t   Result c1 * c2:\t" + (c1 * c2));

            Console.WriteLine("\nTesting __mult__(Matrix c3, scalar x) ...\n");
            Matrix c3 = m_from_list;
            Matrix c4 = Matrix.from_list(new List<double[]>() { new double[] { 4 }, new double[] { 5 }, new double[] { 6 } });
            double x = 5.0;
            Console.WriteLine("\tc3: " + c3.ToString());
            Console.WriteLine("\tc4: " + c4.ToString());
            Console.WriteLine("\tx: " + x.ToString());
            Console.WriteLine("\n\tExpecting c3 * x:\t[[5, 20, 35], [15, 55, 95]]");
            Console.WriteLine("\t   Result c3 * x:\t" + (c3 * x));
            Console.WriteLine("\n\tExpecting c4 * x:\t[[20], [25], [30]]");
            Console.WriteLine("\t   Result c4 * x:\t" + (c4 * x));
        }

        public static void test_matrix_div()
        {

            Console.WriteLine("\nTesting __div__(,) ...\n");
            Matrix A = Matrix.from_list(new List<double[]>() { new double[] { 1, 2 }, new double[] { 4, 9 } });
            Matrix B = Matrix.from_list(new List<double[]>() { new double[] { 5, 2 }, new double[] { 1, 1 } });
            Console.WriteLine("\tA: " + A.ToString());
            Console.WriteLine("\tB: " + B.ToString());

            Console.WriteLine("\n\tTesting __div__(scalar x, Matrix A) ...\n");
            Console.WriteLine("\t\tExpecting 1/A: [[9, -2], [-4, 1]]");
            Console.WriteLine("\t\tResult of 1/A: " + (1 / A));

            Console.WriteLine("\n\tTesting __div__(Matria A, Matrix A) ...\n");
            Console.WriteLine("\t\tExpecting A/A: [[1, 0], [0, 1]]");
            Console.WriteLine("\t\tResult of A/A: " + (A / A));

            Console.WriteLine("\n\tTesting __div__(Matrix A, scalar x) ...\n");
            Console.WriteLine("\t\tExpecting A/2: [[0.5, 1], [2, 4.5]]");
            Console.WriteLine("\t\tResult of A/2: " + (A / 2));

            A = Matrix.from_list(new List<double[]>() { new double[] { 1, 2, 2 }, new double[] { 4, 4, 2 }, new double[] { 4, 6, 4 } });
            B = Matrix.from_list(new List<double[]>() { new double[] { 3 }, new double[] { 6 }, new double[] { 10 } });
            Console.WriteLine("\n\n\tA: " + A.ToString());
            Console.WriteLine("\tB: " + B.ToString());

            Console.WriteLine("\n\t\tExpecting (1/A)*B: [[-1], [3], [-1]]");
            Console.WriteLine("\t\tResult of (1/A)*B: " + ((1 / A) * B));
        }

        public static void test_swap_rows()
        {
            Console.WriteLine("\nTesting swap_rows(A, i, j) ...\n");
            Matrix A = Matrix.from_list(new List<double[]>() { new double[] { 3, 4, 5 }, new double[] { 8, 8, 8 } });
            Console.WriteLine("A: " + A.ToString());
            A.swap_rows(0, 1);
            Console.WriteLine("\n\tExpecting:\t[[8, 8, 8], [3, 4, 5]]");
            Console.WriteLine("\t   Result:\t" + A.ToString());
        }

        public static void test_data()
        {
            Console.WriteLine("\nTesting data ...\n");
            Matrix A = Matrix.from_list(new List<double[]>() { new double[] { 2, 4, 2, 1 }, new double[] { 3, 1, 5, 2 }, new double[] { 1, 2, 3, 3 }, new double[] { 0, 6, 1, 2 } });
            double[] list = A.data();
            
            Console.WriteLine("A: " + A.ToString());

            Console.WriteLine("\n\tExpecting:\t[2, 4, 2, 1, 3, 1, 5, 2, 1, 2, 3, 3, 0, 6, 1, 2]");

            Console.WriteLine("\t   Result:\t[" + string.Join(", ", list) + "]");
        }

        public static void test_transpose()
        {
            Console.WriteLine("\nTesting Transpose() ...\n");
            Matrix A = Matrix.from_list(new List<double[]>() { new double[] { 1, 2, 3 }, new double[] { 4, 5, 6 } });
            Console.WriteLine("A: " + A.ToString());
            A.swap_rows(0, 1);
            Console.WriteLine("\n\tExpecting:\t[[4, 1], [5, 2], [6, 3]]");
            Console.WriteLine("\t   Result:\t" + A.Transpose());
        }

        public static void test_clone()
        {
            Console.WriteLine("\nTesting Clone() ...\n");
            Matrix b1 = Matrix.from_list(new List<double[]>() { new double[] { 4, 3 }, new double[] { 2, 1 } });
            Matrix b1_copy = b1.Clone();
            Console.WriteLine("\n\t    tb1: " + b1.ToString());
            Console.WriteLine("\tb1_copy: " + b1_copy.ToString());
            b1[0, 0] = 5.0;
            Console.WriteLine("\n\tFollowing update of b1[0, 0] = 5.0, copy should remain unchanged:");
            Console.WriteLine("\n\t     b1: " + b1.ToString());
            Console.WriteLine("\tb1_copy: " + b1_copy.ToString()); 
        }

        public static void test_function_class()
        {
            Console.WriteLine("\nTesting Function Class ...\n");
            MyFunction mFunction = new MyFunction();

            Console.WriteLine("\n\tf(x) = (x - 2) * (x + 8)");

            Console.WriteLine("\n\tExpecting:\t f(2) = 0");
            Console.WriteLine("\t   Result:\t f(2) = " + mFunction.f(2));
            Console.WriteLine("\n\tExpecting:\t f(5) = 39");
            Console.WriteLine("\t   Result:\t f(5) = " + mFunction.f(5));
            Console.WriteLine("\n\tExpecting:\t Df(2) ~ 10");
            Console.WriteLine("\t   Result:\t Df(2) = " + mFunction.Df(2));
            Console.WriteLine("\n\tExpecting:\t Df(5) ~ 16");
            Console.WriteLine("\t   Result:\t Df(5) = " + mFunction.Df(5));
            Console.WriteLine("\n\tExpecting:\t DDf(2) ~ 2");
            Console.WriteLine("\t   Result:\t Df(2) = " + mFunction.DDf(2));
            Console.WriteLine("\n\tExpecting:\t DDf(5) ~ 2");
            Console.WriteLine("\t   Result:\t Df(5) = " + mFunction.DDf(5));
        }

        public static void test_solve_newton()
        {
            Console.WriteLine("\nTesting solve_newton() ...\n");
            MyFunction mFunction = new MyFunction();

            Console.WriteLine("\n\tf(x) = (x - 2) * (x + 8)");

            Console.WriteLine("\n\tExpecting:\t solve_newton(0.5) ~ 2");
            Console.WriteLine("\t   Result:\t solve_newton(0.5) = " + mFunction.solve_newton(0.5));
        }

        public static void test_optimize_newton()
        {
            Console.WriteLine("\nTesting optimize_newton() ...\n");
            MyFunction mFunction = new MyFunction();

            Console.WriteLine("\n\tf(x) = (x - 2) * (x + 8)");

            Console.WriteLine("\n\tExpecting:\t optimize_newton(0.5) ~ -3");
            Console.WriteLine("\t   Result:\t optimize_newton(0.5) = " + mFunction.optimize_newton(0.5));
        }

        public static void test_solve_newton_stabilized()
        {
            Console.WriteLine("\nTesting solve_newton_stabilized() ...\n");
            MyFunction mFunction = new MyFunction(); 
            MyFunction2 mFunction2 = new MyFunction2();

            Console.WriteLine("\n\tf(x) = (x - 2) * (x + 8)");

            Console.WriteLine("\n\tExpecting:\t solve_newton_stabilized(-0.5, 9) = 2.000000002707719");
            Console.WriteLine("\t   Result:\t solve_newton_stabilized(-0.5, 9) = " + mFunction.solve_newton_stabilized(-0.5, 9));

            Console.WriteLine("\n\tf(x) = (x - 2) * (x - 2) * (x - 2) + x");

            Console.WriteLine("\n\tExpecting:\t solve_newton_stabilized(-10, 9) = 0.9999999946201354");
            Console.WriteLine("\t   Result:\t solve_newton_stabilized(-10, 9) = " + mFunction2.solve_newton_stabilized(-10, 9));
        }

        public static void test_solve_bisection()
        {
            Console.WriteLine("\nTesting solve_bisection() ...\n");
            MyFunction mFunction = new MyFunction();
            MyFunction2 mFunction2 = new MyFunction2();

            Console.WriteLine("\n\tf(x) = (x - 2) * (x + 8)");

            Console.WriteLine("\n\tExpecting:\t solve_bisection(-0.5, 9) = 2.0000267028808594");
            Console.WriteLine("\t   Result:\t solve_bisection(-0.5, 9) = " + mFunction.solve_bisection(-0.5, 9));

            Console.WriteLine("\n\tf(x) = (x - 2) * (x - 2) * (x - 2) + x");

            Console.WriteLine("\n\tExpecting:\t solve_bisection(-10, 9) = 0.9999942779541016");
            Console.WriteLine("\t   Result:\t solve_bisection(-10, 9) = " + mFunction2.solve_bisection(-10, 9));
        }

        public static void test_is_almost_symmetric()
        {
            Console.WriteLine("\nTesting is_almost_symmetric() ...\n");
            Numeric n = new Numeric();
            
            Matrix A = Matrix.from_list(new List<double[]>() { new double[] { 1, 7, 3 }, new double[] { 7, 4, -5 }, new double[] { 3, -5, 6 } });
            Console.WriteLine("\n\tA: " + A.ToString());
            Console.WriteLine("\n\t\tA is_almost_symmetric (expect success): " + n.is_almost_symmetric(A));

            Matrix B = Matrix.from_list(new List<double[]>() { new double[] { 1, 2 }, new double[] { 3, 4 } });
            Console.WriteLine("\n\tB: " + B.ToString());
            Console.WriteLine("\n\t\tB is_almost_symmetric (expect fail): " + n.is_almost_symmetric(B));
        }

        public static void test_norm()
        {
            Console.WriteLine("\nTesting norm(List<double>x, p=1) ...\n");

            List<double> list = new List<double>(new double[] { 2, 3, 4 });
            Matrix A = Matrix.from_list(new List<double[]>() { new double[] { 1, 2, 3 }, new double[] { 4, 5, 6 }, new double[] { 7, 8, 9 } });

            try
            {
                Console.WriteLine("\n\tlist = [2, 3, 4]");
                Console.WriteLine("\n\tExpecting: \t1-norm of list = 9");
                Console.WriteLine("\t   Result: \t1-norm of list = " + Numeric.norm(list, 1));

                Console.WriteLine("\n\tExpecting: \t2-norm of list = 5.385164807");
                Console.WriteLine("\t   Result: \t2-norm of list = " + Numeric.norm(list, 2));

                Console.WriteLine("\n\tA = [[1, 2, 3], [4, 5, 6],  [7, 8, 9]]");
                Console.WriteLine("\n\tExpecting: \t1-norm of Matrix A = 18");
                Console.WriteLine("\t   Result: \t1-norm of Matrix " + A.ToString() + ": " + Numeric.norm(A, 1));

                Console.WriteLine("\n\tExpecting: \t2-norm of Matrix " + A.ToString() + ": EXCEPTION");
                Numeric.norm(A, 2);
            }
            catch (Exception e)
            {
                Console.WriteLine("\t   Result:");
                Console.WriteLine("\t\t{0} Exception caught.", e.ToString());
            }
        }

        public static void test_condition_number()
        {
            Numeric n = new Numeric();
            List<double[]> list1 = new List<double[]>() { new double[] { 1, 2 }, new double[] { 3, 4 } };
            
            Console.WriteLine("\nTesting test_condition_number(Matrix A) ...\n");
            try
            {
                Console.WriteLine("\tExpecting:\tCondition number of Matrix [[1, 2], [3, 4]] = 21");
                Console.WriteLine("\t   Result:\tCondition number of Matrix " + Matrix.from_list(list1) + " = " + n.condition_number(Matrix.from_list(list1)));
            }
            catch (Exception e)
            {
                Console.WriteLine("{0} Exception caught.", e.ToString());
            }

            Console.WriteLine("\nTesting test_condition_number(f, x=None, h=0.000001) ...\n");
            try
            {
                MyFunction3 fn = new MyFunction3();
                Console.WriteLine("\tExpecting:\tCondition number of function f(x) = (x * x) - (5.0 * x) = 0.75");
                Console.WriteLine("\t   Result:\tCondition number of function f(x) = (x * x) - (5.0 * x) = " + n.condition_number(fn, 1));
            }
            catch (Exception e)
            {
                Console.WriteLine("{0} Exception caught.", e.ToString());
            }
        }

        public static void test_exp()
        {
            Console.WriteLine("\nTesting exp(Matrix A) ...\n");
            Matrix A = Matrix.from_list(new List<double[]>() { new double[] { 1, 2 }, new double[] { 3, 4 } });

            Console.WriteLine("\n\tExpecting:\t exp(" + A.ToString() + ") = [[51.9682385150117, 74.7355185953338], [112.1032778930007, 164.07151640801243]]");
            Console.WriteLine("\t   Result:\t exp(" + A.ToString() + ") = " + Numeric.exp(A).ToString());
        }


        public static void test_is_positive_definite()
        {
            Console.WriteLine("\nTesting is_positive_definite(Matrix A) ...\n");
            Numeric n = new Numeric();
            List<double[]> list1 = new List<double[]>() { new double[] { 1, 2 }, new double[] { 2, 1 } }; //false
            List<double[]> list2 = new List<double[]>() { new double[] { 2, -1, 0 }, new double[] { -1, 2, -1 }, new double[] { 0, -1, 2 } }; //true

            Console.Write("\n\tExpect Fail: is_positive_definite(" + Matrix.from_list(list1) + ") = ");
            Console.WriteLine(n.is_positive_definite(Matrix.from_list(list1)));
            Console.WriteLine("\tExpect True: is_positive_definite(" + Matrix.from_list(list2) + ") = " + n.is_positive_definite(Matrix.from_list(list2)));
        }

        public static void test_cholesky()
        {
            Console.WriteLine("\nTesting Cholesky(Matrix A) ...\n");
            Numeric n = new Numeric();
            List<double[]> list3 = new List<double[]>() { new double[] { 4, 2, 1 }, new double[] { 2, 9, 3 }, new double[] { 1, 3, 16 } };
            Matrix A = Matrix.from_list(list3);
            Matrix L = n.Cholesky(A);
            Console.WriteLine("\n\tA: " + A.ToString());
            Console.WriteLine("\n\tExpect: Cholesky(A) = [[2.0, 0, 0], [1.0, 2.8284271247461903, 0], [0.5, 0.88388347648318433, 3.8689468851355402]]");
            Console.WriteLine("\t Result:Cholesky(A) = " + L.ToString());
        }

        public static void test_is_almost_zero()
        {
            Console.WriteLine("\nTesting is_almost_zero(Matrix A) ...\n");
            Numeric n = new Numeric();
            List<double[]> list3 = new List<double[]>() { new double[] { 4, 2, 1 }, new double[] { 2, 9, 3 }, new double[] { 1, 3, 16 } };
            Matrix A = Matrix.from_list(list3);
            Matrix L = n.Cholesky(A);
            Console.WriteLine("\n\tA: " + A.ToString());
            Console.WriteLine("\n\tExpect True: is_almost_zero(A - L*L.t) = " + n.is_almost_zero(A - L * L.Transpose()));
        }

        public static void test_solve_secant()
        {
            Console.WriteLine("\nTesting solve_secant(x) ...\n");
            MyFunction4 func = new MyFunction4();

            Console.WriteLine("\n\tf(x) = (x-2) * (x-5)");

            Console.WriteLine("\n\tExpecting:\t solve_secant(1.0) = 1.9999999552965158");
            Console.WriteLine("\t   Result:\t solve_secant(1.0) = " + func.solve_secant(1.0));
        }

        public static void test_optimize_secant()
        {
            Console.WriteLine("\nTesting optimize_secant(x) ...\n");
            MyFunction4 func = new MyFunction4();

            Console.WriteLine("\n\tf(x) = (x-2) * (x-5)");

            Console.WriteLine("\n\tExpecting:\t optimize_secant(1.0) = 3.4999999999402016");
            Console.WriteLine("\t   Result:\t optimize_secant(1.0) = " + func.optimize_secant(1.0));
        }

        public static void test_optimize_bisection()
        {
            Console.WriteLine("\nTesting optimize_bisection(a, b) ...\n");
            MyFunction4 func = new MyFunction4();

            Console.WriteLine("\n\tf(x) = (x-2) * (x-5)");

            Console.WriteLine("\n\tExpecting:\t optimize_bisection(2.0, 5.0) = 3.5");
            Console.WriteLine("\t   Result:\t optimize_bisection(2.0, 5.0) = " + func.optimize_bisection(2.0, 5.0));
        }

        public static void test_optimize_golden_search()
        {
            Console.WriteLine("\nTesting optimize_golden_search(a, b) ...\n");
            MyFunction4 func = new MyFunction4();

            Console.WriteLine("\n\tf(x) = (x-2) * (x-5)");

            Console.WriteLine("\n\tExpecting:\t optimize_golden_search(2.0, 5.0) = 3.500061284523513");
            Console.WriteLine("\t   Result:\t optimize_golden_search(2.0, 5.0) = " + func.optimize_golden_search(2.0, 5.0));
        }

        public static void test_markowitz()
        {
            Console.WriteLine("\nTesting Markovitz(mu, A, r_free)");
            List<double[]> listMark = new List<double[]>() { new double[] { 0.04, 0.006, 0.02 }, new double[] { 0.006, 0.09, 0.06 }, new double[] { 0.02, 0.06, 0.16 } };
            Matrix cov = Matrix.from_list(listMark);
            List<double[]> listMu = new List<double[]>() { new double[] { 0.10 }, new double[] { 0.12 }, new double[] { 0.15 } };
            Matrix mu = Matrix.from_list(listMu);
            double r_free = 0.05, portfolio_return, portfolio_risk;
            double[] portfolio;

            Console.WriteLine("\n\tmu: " + mu.ToString());
            Console.WriteLine("\tcov: " + cov.ToString());
            Console.WriteLine("\tr_free: " + r_free.ToString());

            Numeric.Markovitz(mu, cov, r_free, out portfolio, out portfolio_return, out portfolio_risk);

            Console.WriteLine("\n\tExpecting:\tNumeric.Markovitz(mu, cov, r_free):");
            Console.WriteLine("\t\tx = [0.55663430420711979, 0.27508090614886727, 0.16828478964401297]");
            Console.WriteLine("\t\tret = 0.113915857605, risk = 0.186747095412");
            Console.WriteLine("\tResult set:\t x = [" + string.Join(", ", portfolio) + "]");
            Console.WriteLine("\n\treturn = " + portfolio_return + ", risk = " + portfolio_risk);
        }

        public static void test_sqrt()
        {
            Console.WriteLine("\nTesting sqrt(x) ...\n");

            Console.WriteLine("\n\tExpecting:\t sqrt(-45) = 6.708203932499369j");
            Console.WriteLine("\t   Result:\t sqrt(-45) = " + Numeric.sqrt(-45.0));

            Console.WriteLine("\n\tExpecting:\t sqrt(10.0) = 3.1622776601683795");
            Console.WriteLine("\t   Result:\t sqrt(10.0) = " + Numeric.sqrt(10.0));

            Console.WriteLine("\n\tExpecting:\t sqrt(9) = 3.0");
            Console.WriteLine("\t   Result:\t sqrt(9) = " + Numeric.sqrt(9));
        }

        public static void test_solve_fixed_point()
        {
            Console.WriteLine("\nTesting solve_fixed_point(x) ...\n");
            MyFunction5 f5 = new MyFunction5();

            Console.WriteLine("\n\tf(x) = (x - 2) * (x - 5) / 10");

            Console.WriteLine("\n\tExpecting:\t solve_fixed_point(1.0) = 1.9996317277774924");
            Console.WriteLine("\t   Result:\t solve_fixed_point(1.0) = " + f5.solve_fixed_point(1.0));
        }

        public static void test_partial()
        {
            Console.WriteLine("\nTesting partial(x, i) ...\n");
            MyFunction6 f6 = new MyFunction6();
            double[] x = new double[] {1, 1, 1};

            Console.WriteLine("\n\tf(x[]) =  (2.0 * x[0]) + (3.0 * x[1]) + (5.0 * x[1] * x[2])");

            Console.WriteLine("\n\tExpecting:\t partial(x, 0) = 2.0");
            Console.WriteLine("\t   Result:\t partial(x, 0) = " + f6.partial(x, 0));

            Console.WriteLine("\n\tExpecting:\t partial(x, 2) = 8.0");
            Console.WriteLine("\t   Result:\t partial(x, 1) = " + f6.partial(x, 1));

            Console.WriteLine("\n\tExpecting:\t partial(x, 2) = 5.0");
            Console.WriteLine("\t   Result:\t partial(x, 2) = " + f6.partial(x, 2));
        }

        public static void test_jacobian()
        {
            Console.WriteLine("\nTesting jacobian(fs, x=[1,1,1]) ...\n");
            MultivariateFunction multi = new MultivariateFunction();
            List<MultivariateFunction> fs = new List<MultivariateFunction>() { new MyFunction6(), new MyFunction9() } ;
            double[] x = new double[] { 1, 1, 1 };

            Console.WriteLine("\n\tfs =  [ (2.0 * x[0]) + (3.0 * x[1]) + (5.0 * x[1] * x[2]), 2.0 * x[0] ]");

            Console.WriteLine("\n\tExpecting:\tmulti.jacobian(fs, x) = [[1.9999999..., 7.999999..., 4.9999999...], [1.9999999..., 0.0, 0.0]]");
            Console.WriteLine("\tResult:\t\tmulti.jacobian(fs, x=[1, 1, 1]) = " + multi.jacobian(fs, x) );
        }

        public static void test_gradient()
        {
            Console.WriteLine("\nTesting gradient(x) ...\n");
            MyFunction6 f6 = new MyFunction6();
            double[] x = new double[] {1, 1, 1};

            Console.WriteLine("\n\tx = [1, 1, 1]");

            Console.WriteLine("\n\tExpecting:\tgradient(x) = [[1.999999...], [7.999999...], [4.999999...]]" );
            Console.WriteLine("\tResult:\t\tgradient(x=[1, 1, 1]) = " + f6.gradient(x) );
        }

        public static void test_hessian()
        {
            Console.WriteLine("\nTesting hessian(x) ...\n");
            MyFunction6 f6 = new MyFunction6();
            double[] x = new double[] { 1, 1, 1 };

            Console.WriteLine("\n\tx = [1, 1, 1]");

            Console.WriteLine("\n\tExpecting:\thessian(x) = [[0.0, 0.0, 0.0], [0.0, 0.0, 5.000000...], [0.0, 5.000000..., 0.0]]");
            Console.WriteLine("\tResult:\t\thessian(x=[1, 1, 1]) = " + f6.hessian(x));   
        }

        public static void test_solve_newton_multi()
        {
            Console.WriteLine("\nTesting solve_newton_multi(fs, x) ...\n");
            MultivariateFunction multi = new MultivariateFunction();
            List<MultivariateFunction> fs = new List<MultivariateFunction>() { new MyFunction7(), new MyFunction8() };
            double[] x = new double[] { 1, 1 };
            
            Console.WriteLine("\n\tfs =  [ x[0] * x[1] - 2, x[0] - 3.0 * x[1] * x[1] ]");

            Console.WriteLine("\n\tExpecting:\tsolve_newton_multi(fs, x=[1, 1]) = [2.2894284851066793, 0.87358046473632422]" );
            double[] result = multi.solve_newton_multi(fs, x);
            Console.WriteLine("\tResult:\t\tsolve_newton_multi(fs, x=[1, 1]) =  [" + result[0] + ", " + result[1] + "]");

            fs = new List<MultivariateFunction>() { new MyFunction10(), new MyFunction11() };
            x = new double[] { 0, 0 };
            Console.WriteLine("\n\tfs =  [ x[0] + x[1], x[0] + (x[1] * x[1]) - 2 ]");

            Console.WriteLine("\n\tExpecting:\tsolve_newton_multi(fs, x=[0, 0]) = [1.0000000006984919, -1.0000000006984919]");
            result = multi.solve_newton_multi(fs, x);
            Console.WriteLine("\tResult:\t\tsolve_newton_multi(fs, x=[0, 0]) =  [" + result[0] + ", " + result[1] + "]");
        }

        public static void test_optimize_newton_multi()
        {
            Console.WriteLine("\nTesting test_optimize_newton_multi(x) ...\n");
            MyFunction12 f12 = new MyFunction12();
            double[] x = new double[] { 0, 0 };

            Console.WriteLine("\n\tf(x[]) =  -1.0 * ( Math.Pow(x[0] - 2, 2) + Math.Pow(x[1] - 3, 2) )");

            Console.WriteLine("\n\tExpecting:\toptimize_newton_multi(x=[0, 0]) = [2.0, 3.0] maximum" );
            string min_max = string.Empty;
            double[] result = f12.optimize_newton_multi(x, ref min_max);
            Console.WriteLine("\tResult:\t\toptimize_newton_multi(x=[0, 0]) = [" + result[0] + ", " + result[1] + "] " + min_max);

            MyFunction13 f13 = new MyFunction13();
            Console.WriteLine("\n\tf(x[]) =  Math.Pow(x[0] - 2, 2) + Math.Pow(x[1] - 3, 2)");

            Console.WriteLine("\n\tExpecting:\toptimize_newton_multi(x=[0, 0]) = [2.0, 3.0] minimum");
            result = f13.optimize_newton_multi(x, ref min_max);
            Console.WriteLine("\tResult:\t\toptimize_newton_multi(x=[0, 0]) = [" + result[0] + ", " + result[1] + "] " + min_max);
        }

        public static void fit_least_squares()
        {
            Console.WriteLine("\nTesting fit_least_squares(points, f, out fitting_coef, out chi2, out fitting_f) ...\n");
            
            double[,] points = new double[100, 3];
            Function[] f = new Function[3] { new Quadratic0(), new Quadratic1(), new Quadratic2() } ;
            double[] fitting_coef;
            double chi2;
            Function fitting_f;
            
            for (int i = 0; i < 100; i++)
            {
                points[i, 0] = i;
                points[i, 1] = 5 + 0.8*i + 0.3*i*i + Math.Sin(i);
                points[i, 2] = 2;
            }

            Console.WriteLine("\tdouble[,] points = new double[100, 3]");
            Console.WriteLine("\tPopulate points[,] ...");
            Console.WriteLine("\t\tfor (int i = 0; i < 100; i++)");
            Console.WriteLine("\t\t{");
            Console.WriteLine("\t\tpoints[i, 0] = i;");
            Console.WriteLine("\t\tpoints[i, 1] = 5 + 0.8*i + 0.3*i*i + Math.Sin(i);");
            Console.WriteLine("\t\tpoints[i, 2] = 2;");
            Console.WriteLine("\t\t}");

            Console.WriteLine("\tf= [ Math.Pow(x, 0), Math.Pow(x, 1), Math.Pow(x, 2) ]" );

            Function.fit_least_squares(points, f, out fitting_coef, out chi2, out fitting_f);
            Console.WriteLine("\n\tExpecting:\tfit_least_squares(points, f, out fitting_coef, out chi2, out fitting_f) = "); 
            Console.WriteLine("\t\t90  2507.89\t2506.98");
            Console.WriteLine("\t\t91  2562.21\t2562.08");
            Console.WriteLine("\t\t92  2617.02\t2617.78");
            Console.WriteLine("\t\t93  2673.15\t2674.08");
            Console.WriteLine("\t\t94  2730.75\t2730.98");
            Console.WriteLine("\t\t95  2789.18\t2788.48");
            Console.WriteLine("\t\t96  2847.58\t2846.58");
            Console.WriteLine("\t\t97  2905.68\t2905.28");
            Console.WriteLine("\t\t98  2964.03\t2964.58");
            Console.WriteLine("\t\t99  3023.5 \t3024.48");

            Console.WriteLine("\tResult:\t\tfit_least_squares(points, f, out fitting_coef, out chi2, out fitting_f) = ");
            for (int i = 90; i < 100; i++)
                Console.WriteLine("\t\t" + points[i, 0] + "  " + Math.Round(points[i, 1], 2) + "\t" + Math.Round(fitting_f.eval_fitting_function(points[i, 0]), 2));
        }
    }
}
