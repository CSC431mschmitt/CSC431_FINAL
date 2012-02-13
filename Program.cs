using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MatrixLib;

namespace MatrixTester
{
    class Program
    {
        /* This will be for running tests. */
        static void Main(string[] args)
        {
            //Numeric n = new Numeric();
            
            //Matrix m1 = new Matrix(4, 3, 2);
            //Matrix m2 = new Matrix();
            //Matrix m3 = new Matrix(3, 3, 4.5);
            //Matrix m4 = new Matrix(6, 2, 7);
            //Matrix m5 = new Identity(3, 4, 7); 
            //double[] v = new double[] {4, 3, 2, 1};
            //double[] v1 = new double[] {1, 2};
            //double[] v2 = new double[] {3, 4};
            //Matrix m6 = new Diagonal(v);

            //List <double[]> m7 = new List<double[]>();

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

            //Test def diagnol(array[])
            Matrix m_diagnol = Matrix.diagonal(new double[] { 5, 6, 7, 8 });
            Console.WriteLine("diagnol matrix : " +  m_diagnol);

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
            //m7 = m6.as_list(m6);
            //Console.WriteLine(m7[0][0]);
            //Console.WriteLine(m7[0][1]);
            //Console.WriteLine(m7[0][2]);
            //Console.WriteLine(m7[0][3]);
            //Console.WriteLine(m7[1][0]);
            //Console.WriteLine(m7[1][1]);
            //Console.WriteLine(m7[1][2]);
            //Console.WriteLine(m7[1][3]);
            //Console.WriteLine(m7[2][0]);
            //Console.WriteLine(m7[2][1]);
            //Console.WriteLine(m7[2][2]);
            //Console.WriteLine(m7[2][3]);
            //Console.WriteLine(m7[3][0]);
            //Console.WriteLine(m7[3][1]);
            //Console.WriteLine(m7[3][2]);
            //Console.WriteLine(m7[3][3]);
            Console.ReadLine();
        }
    }
}
