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

    class Program
    {
        /* This will be for running tests. */
        static void Main(string[] args)
        {
            /*
            //Test def from_list(v)
            List<double[]> data = new List<double[]>(){new double[]{1,2}, new double[]{3,4}};
            Matrix m_from_list = Matrix.from_list(data);
            Console.WriteLine("from list: " + m_from_list);

            //Test def as_list(A)
            List<double[]> m_as_list = m_from_list.as_list();
            Console.WriteLine("as list: ");
            Console.Write("\t[");
            for (int j=0; j < m_as_list.Count; j++)
            {
                for (int k=0; k < m_as_list[j].Length; k++)
                {
                    if (j > 0)
                        Console.Write(", ");
                    Console.Write( String.Format("[{0}, {1}]", m_as_list[j][k], m_as_list[j][k+1]) );
                    k++;
                }
            }
            Console.WriteLine("]");
            
            //Test def identity(rows)
            Matrix m_identity = Matrix.identity(3, 1, 0);
            Console.WriteLine("identity matrix: " + m_identity);

            //Test def diagonol(array[])
            Matrix m_diagonol = Matrix.diagonal(new double[] { 5, 6, 7, 8 });
            Console.WriteLine("diagonol matrix : " +  m_diagonol);

            //Test def row(row_num)
            Console.WriteLine( "print row 0 of matrix: " + m_from_list.row(0) );

            //Test def row(row_num)
            Console.WriteLine("print row 1 of matrix: " + m_from_list.row(1));

            //Test def col(col_num)
            Console.WriteLine("print col 0 of matrix: " + m_from_list.col(0));

            //Test def col(col_num)
            Console.WriteLine("print col 1 of matrix: " + m_from_list.col(1));

            //Test __mult__(A, B)
            Matrix A = m_from_list;
            List<double[]> Blist = new List<double[]>() { new double[] { 1 }, new double[] { 2 } };
            Matrix B = Matrix.from_list(Blist);
            Console.WriteLine("A: " + A.ToString());
            Console.WriteLine("B: " + B.ToString());
            Console.WriteLine("A * B: " + (A*B) );

            //Test __mult__(c1, c2)
            Matrix c1 = Matrix.from_list( new List<double[]>() { new double[] {3}, new double[] {2}, new double[] {1} } );
            Matrix c2 = Matrix.from_list( new List<double[]>() { new double[] {4}, new double[] {5}, new double[] {6} });
            Console.WriteLine("c1: " + c1.ToString());
            Console.WriteLine("c2: " + c2.ToString());
            Console.WriteLine("c1 * c2: " + (c1 * c2));

            //Test __mult__(c3, x)
            Matrix c3 = m_from_list;
            Matrix c4 = Matrix.from_list(new List<double[]>() { new double[] { 4 }, new double[] { 5 }, new double[] { 6 } });
            double x = 5.0;
            Console.WriteLine("c3: " + c3.ToString());
            Console.WriteLine("c4: " + c4.ToString());
            Console.WriteLine("c3 * x: " + (c3 * x));
            Console.WriteLine("c4 * x: " + (c4 * x));

            //Test __add__(A, B)
            Blist = new List<double[]>() { new double[] { 4, 3 }, new double[] { 2, 1 } };
            B = Matrix.from_list(Blist);
            Console.WriteLine("A: " + A.ToString());
            Console.WriteLine("B: " + B.ToString());
            Console.WriteLine("A + B: " + (A + B));

            //Test __sub__(A, B)
            Console.WriteLine("A: " + A.ToString());
            Console.WriteLine("B: " + B.ToString());
            Console.WriteLine("B - A: " + (B - A));
            
            //Test __neg__(A)
            Console.WriteLine("-A: " + (-A));
            
            //Test swap_rows(A, i, j)
            Console.WriteLine("B: " + B.ToString());
            B.swap_rows(0, 1);
            Console.WriteLine("B after swap rows: " + B.ToString());

            //Test Transpose property
            Matrix A1 = Matrix.from_list( new List<double[]>() { new double[] { 1,2,3 }, new double[] { 4,5,6 } } );
            Console.WriteLine(A1.ToString());
            Console.WriteLine(A1.Transpose());

            //Test Clone Matrix
            Matrix b1 = Matrix.from_list( new List<double[]>() { new double[] { 4, 3 }, new double[] { 2, 1 } } );
            Matrix b1_copy = b1.Clone();
            Console.WriteLine("b1: " + b1.ToString());
            Console.WriteLine("b1_copy: " + b1_copy.ToString());
            b1[0, 0] = 5.0;
            Console.WriteLine("b1: " + b1.ToString());
            Console.WriteLine("b1_copy: " + b1_copy.ToString());
            

            ////Test Matrix Division
            //Matrix A = Matrix.from_list(new List<double[]>() { new double[] { 1, 2 }, new double[] { 3, 4 } });
            //Matrix B = Matrix.from_list(new List<double[]>() { new double[] { 5, 2 }, new double[] { 1, 1 } });

            //Console.WriteLine("A / B: " + A / B);
            //Console.WriteLine("B / A: " + B / A);
            */
            ////Test Function Class
            MyFunction mFunction = new MyFunction();
            MyFunction2 mFunction2 = new MyFunction2();

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
            

            /*
            Numeric n = new Numeric();

            //Test is_almost_symmetric, expect success
            Matrix B = Matrix.from_list(new List<double[]>() { new double[] { 1, 7, 3 }, new double[] { 7, 4, -5 }, new double[] { 3, -5, 6 } });
            Console.WriteLine("A is_almost_symmetric (expect success): " + n.is_almost_symmetric(B));
            
            //Test is_almost_symmetric, expect fail
            Matrix A = Matrix.from_list(new List<double[]>() { new double[] { 1, 2 }, new double[] { 3, 4 } });
            Console.WriteLine("A is_almost_symmetric (expect fail): " + n.is_almost_symmetric(A));

            Console.WriteLine(Math.Pow(10.0, -6) * 7.0);
            */

            Console.ReadLine();
        }
    }
}
