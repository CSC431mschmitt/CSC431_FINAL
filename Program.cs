using System;
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

    class Program
    {
        /* This will be for running tests. */
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
            //Console.WriteLine("exp(" + Matrix.from_list(list1) + "): " + n.exp(Matrix.from_list(list1)));

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

            Console.ReadLine();
        }
    }
}
